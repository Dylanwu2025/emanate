import { EventEmitter } from 'node:events';
import {
  AGENT_PROFILES,
  buildAgentSystemPrompt,
} from './agents.js';
import type {
  AgentId,
  EmanateStore,
  MessageRole,
  MessageRow,
  MessageStatus,
} from './db.js';
import type { ChatMessage, LLMProvider } from './llm.js';
import {
  TurnRouter,
  type SpeakerPlan,
  type WaveRoutingPlan,
} from './router.js';

export const MAX_A2A_WAVES = 3;

export interface PublicWireMessage {
  seq: number;
  id: string;
  role: MessageRole;
  agentId: AgentId | null;
  agentName: string | null;
  content: string;
  status: MessageStatus;
  createdAt: number;
}

export type OrchestratorSSEEvent =
  | {
      type: 'message.created';
      message: PublicWireMessage;
    }
  | {
      type: 'message.delta';
      id: string;
      seq: number;
      agentId: AgentId;
      delta: string;
    }
  | {
      type: 'message.done';
      message: PublicWireMessage;
    }
  | {
      type: 'turn.idle';
    };

export function toPublicWireMessage(row: MessageRow): PublicWireMessage {
  return {
    seq: row.seq,
    id: row.id,
    role: row.role,
    agentId: row.agent_id,
    agentName: row.agent_id ? AGENT_PROFILES[row.agent_id].name : null,
    content: row.content,
    status: row.status,
    createdAt: row.created_at,
  };
}

export function buildConversationHistoryForAgent(
  targetAgentId: AgentId,
  transcript: MessageRow[],
  excludeMessageIds: Set<string> = new Set(),
): ChatMessage[] {
  const history: ChatMessage[] = [];
  for (const row of transcript) {
    if (excludeMessageIds.has(row.id)) continue;
    if (!row.content && row.status === 'streaming') continue;

    if (row.role === 'user') {
      history.push({
        role: 'user',
        content: row.content,
      });
    } else if (row.agent_id === targetAgentId) {
      history.push({
        role: 'assistant',
        content: row.content,
      });
    } else if (row.agent_id) {
      const peerName = AGENT_PROFILES[row.agent_id].name;
      history.push({
        role: 'user',
        content: `[${peerName}]: ${row.content}`,
      });
    }
  }
  return history;
}

export interface ActiveAgentStreamState {
  messageId: string;
  seq: number;
  agentId: AgentId;
  waveIndex: number;
  abortController: AbortController;
  generationEpoch: number;
  flushedContent: string;
}

export interface RunTurnResult {
  userMessage: PublicWireMessage;
  wavePlans: WaveRoutingPlan[];
  agentMessages: PublicWireMessage[];
}

export interface OrchestratorOptions {
  store: EmanateStore;
  llm: LLMProvider;
  maxA2AWaves?: number;
}

export class GroupChatOrchestrator extends EventEmitter {
  public readonly store: EmanateStore;
  public readonly llm: LLMProvider;
  public readonly router: TurnRouter;
  public readonly maxA2AWaves: number;

  private readonly activeStreams = new Map<AgentId, ActiveAgentStreamState>();
  private turnAbortController: AbortController | null = null;

  constructor(options: OrchestratorOptions) {
    super();
    this.store = options.store;
    this.llm = options.llm;
    this.router = new TurnRouter(options.llm);
    this.maxA2AWaves = options.maxA2AWaves ?? MAX_A2A_WAVES;
  }

  public onEvent(listener: (event: OrchestratorSSEEvent) => void): () => void {
    this.on('sse', listener);
    return () => {
      this.off('sse', listener);
    };
  }

  private emitSSE(event: OrchestratorSSEEvent): void {
    this.emit('sse', event);
  }

  public getActiveStreams(): ReadonlyMap<AgentId, ActiveAgentStreamState> {
    return this.activeStreams;
  }

  public isStreaming(): boolean {
    return this.activeStreams.size > 0;
  }

  public getPublicMessages(): PublicWireMessage[] {
    return this.store.listMessages().map(toPublicWireMessage);
  }

  public async handleUserMessage(content: string): Promise<RunTurnResult> {
    const trimmed = content.trim();
    if (!trimmed) {
      throw new Error('User message content cannot be empty.');
    }

    const userRow = this.store.reserveMessage({
      role: 'user',
      content: trimmed,
      status: 'completed',
    });
    const userMessage = toPublicWireMessage(userRow);
    this.emitSSE({ type: 'message.created', message: userMessage });

    this.turnAbortController = new AbortController();
    const turnSignal = this.turnAbortController.signal;

    const wavePlans: WaveRoutingPlan[] = [];
    const agentMessages: PublicWireMessage[] = [];

    try {
      for (let waveIndex = 0; waveIndex <= this.maxA2AWaves; waveIndex++) {
        if (turnSignal.aborted) break;

        const transcriptSnapshot = this.store.listMessages();
        const activeRules = this.store.listActiveStandingRules();

        const plan = await this.router.routeWave({
          waveIndex,
          transcript: transcriptSnapshot,
          activeRules,
          latestUserMessage:
            waveIndex === 0
              ? trimmed
              : transcriptSnapshot[transcriptSnapshot.length - 1]?.content,
          signal: turnSignal,
        });
        wavePlans.push(plan);

        if (waveIndex === 0 && plan.standingRule) {
          this.store.addStandingRule({
            scope: plan.standingRule.scope,
            rule: plan.standingRule.rule,
          });
        }

        if (plan.isResolved || plan.speakers.length === 0) {
          break;
        }

        const isFinalWave = waveIndex === this.maxA2AWaves;
        const completedInWave = await this.executeWave(
          waveIndex,
          plan.speakers,
          isFinalWave,
          turnSignal,
        );
        agentMessages.push(...completedInWave);

        if (turnSignal.aborted) break;
      }
    } finally {
      this.turnAbortController = null;
      this.emitSSE({ type: 'turn.idle' });
    }

    return {
      userMessage,
      wavePlans,
      agentMessages,
    };
  }

  private async executeWave(
    waveIndex: number,
    speakers: SpeakerPlan[],
    isFinalWave: boolean,
    turnSignal: AbortSignal,
  ): Promise<PublicWireMessage[]> {
    // 1. Up-front seq slot reservation in deterministic speaker order BEFORE any LLM stream starts
    const reservedSlots: Array<{
      speaker: SpeakerPlan;
      row: MessageRow;
      streamState: ActiveAgentStreamState;
    }> = [];

    const reservedIds = new Set<string>();

    for (const speaker of speakers) {
      const row = this.store.reserveMessage({
        role: 'agent',
        agentId: speaker.agentId,
        content: '',
        status: 'streaming',
      });
      reservedIds.add(row.id);

      const abortController = new AbortController();
      const onTurnAbort = () => abortController.abort();
      turnSignal.addEventListener('abort', onTurnAbort, { once: true });
      abortController.signal.addEventListener(
        'abort',
        () => turnSignal.removeEventListener('abort', onTurnAbort),
        { once: true },
      );

      const streamState: ActiveAgentStreamState = {
        messageId: row.id,
        seq: row.seq,
        agentId: speaker.agentId,
        waveIndex,
        abortController,
        generationEpoch: 1,
        flushedContent: '',
      };

      this.activeStreams.set(speaker.agentId, streamState);
      reservedSlots.push({ speaker, row, streamState });

      this.emitSSE({
        type: 'message.created',
        message: toPublicWireMessage(row),
      });
    }

    const preWaveTranscript = this.store.listMessages();

    // 2. Launch all speaking agents in this wave concurrently and wait at the wave barrier
    const waveResults = await Promise.all(
      reservedSlots.map(({ speaker, row, streamState }) =>
        this.streamSingleAgentInWave({
          speaker,
          row,
          streamState,
          preWaveTranscript,
          reservedIds,
          isFinalWave,
        }),
      ),
    );

    return waveResults;
  }

  private async streamSingleAgentInWave(params: {
    speaker: SpeakerPlan;
    row: MessageRow;
    streamState: ActiveAgentStreamState;
    preWaveTranscript: MessageRow[];
    reservedIds: Set<string>;
    isFinalWave: boolean;
  }): Promise<PublicWireMessage> {
    const {
      speaker,
      row,
      streamState,
      preWaveTranscript,
      reservedIds,
      isFinalWave,
    } = params;
    const { agentId, effort, maxTokens, thesis, peerTheses, priorSpeakerTheses } = speaker;
    const activeEpoch = streamState.generationEpoch;

    const activeRules = this.store.listActiveStandingRules(agentId);
    const systemPrompt = buildAgentSystemPrompt(agentId, activeRules, {
      effort,
      thesis,
      peerTheses,
      priorSpeakerTheses,
      isFinalWave,
    });

    const conversationHistory = buildConversationHistoryForAgent(
      agentId,
      preWaveTranscript,
      reservedIds,
    );

    const messages: ChatMessage[] = [
      { role: 'system', content: systemPrompt },
      ...(conversationHistory.length > 0
        ? conversationHistory
        : [{ role: 'user' as const, content: 'Please share your perspective.' }]),
    ];

    try {
      for await (const delta of this.llm.streamChat({
        messages,
        maxTokens,
        temperature: 0.4,
        signal: streamState.abortController.signal,
      })) {
        if (
          streamState.abortController.signal.aborted ||
          streamState.generationEpoch !== activeEpoch
        ) {
          break;
        }
        if (!delta) continue;

        streamState.flushedContent += delta;
        this.store.appendMessageDelta(row.id, delta);
        this.emitSSE({
          type: 'message.delta',
          id: row.id,
          seq: row.seq,
          agentId,
          delta,
        });
      }
    } catch (err) {
      if (!streamState.abortController.signal.aborted) {
        throw err;
      }
    } finally {
      if (this.activeStreams.get(agentId) === streamState) {
        this.activeStreams.delete(agentId);
      }
    }

    const completedRow = this.store.markMessageStatus(row.id, 'completed');
    const wireMsg = toPublicWireMessage(completedRow);
    this.emitSSE({
      type: 'message.done',
      message: wireMsg,
    });
    return wireMsg;
  }
}
