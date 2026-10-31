import { EventEmitter } from 'node:events';
import {
  AGENT_PROFILES,
  EFFORT_MAX_TOKENS,
  buildAgentSystemPrompt,
  type EffortLevel,
} from './agents.js';
import type {
  AgentId,
  EmanateStore,
  MessageRole,
  MessageRow,
  MessageStatus,
} from './db.js';
import { sleepAbortable, type ChatMessage, type LLMProvider } from './llm.js';
import {
  TurnRouter,
  extractMentionedAgents,
  isExplicitStopCommand,
  type SpeakerPlan,
  type SteeringClassification,
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

const LEADING_META_DISCLAIMER_RE =
  /^(?:I\s+(?:can(?:'t|’t|not)|only)\s+(?:provide|summarize|share|help\s+with|cover)\s+[^,.!\n]+(?:,\s*but\s+|\s*[.!]\s*)|I(?:'m|’m|\s+am)\s+focused\s+on\s+[^,.\n]+,\s*so\s+(?:I(?:'ll|’ll|\s+will)\s+(?:share|note|mention)\s+that\s+)?)/i;

export function stripLeadingMetaDisclaimer(text: string): string {
  const match = text.match(LEADING_META_DISCLAIMER_RE);
  if (!match) return text;
  const remainder = text.slice(match[0].length).trimStart();
  if (!remainder) return text;
  return remainder[0].toUpperCase() + remainder.slice(1);
}

const LEFT_BOUNDARY_RE = /[\s,.;:!?—–\-(\[{"'/$¥€£]/;
const RIGHT_BOUNDARY_RE = /[\s,.;:!?—–\-)\]}"']/;
const INLINE_LEADING_PUNCTUATION_RE = /^[,.;:!?—–)\]}]/;
const LEADING_BULLET_RE = /^[-*•]\s+/;
const DANGLING_BULLET_AT_END_RE = /(?:^|\n)\s*[-*•]\s*$/;

export function normalizeContinuationSeam(
  flushedContent: string,
  continuationChunk: string,
): string {
  if (!flushedContent || !continuationChunk) {
    return continuationChunk;
  }

  const flushedTrimmed = flushedContent.trimEnd();
  if (!flushedTrimmed) {
    return continuationChunk;
  }

  let contTrimmed = continuationChunk.trimStart();
  if (!contTrimmed) {
    return '';
  }

  const flushedFullTrim = flushedTrimmed.trimStart();
  if (
    flushedFullTrim.length > 0 &&
    contTrimmed.toLowerCase().startsWith(flushedFullTrim.toLowerCase())
  ) {
    contTrimmed = contTrimmed.slice(flushedFullTrim.length).trimStart();
    if (!contTrimmed) {
      return '';
    }
  }

  const maxOverlap = Math.min(flushedTrimmed.length, contTrimmed.length);
  let matchedOverlapLen = 0;

  for (let len = maxOverlap; len >= 1; len--) {
    const suffix = flushedTrimmed.slice(flushedTrimmed.length - len);
    const prefix = contTrimmed.slice(0, len);

    if (suffix.toLowerCase() !== prefix.toLowerCase()) {
      continue;
    }

    const leftIdx = flushedTrimmed.length - len - 1;
    const isLeftBoundary = leftIdx < 0 || LEFT_BOUNDARY_RE.test(flushedTrimmed[leftIdx]);
    const isRightBoundary =
      len === contTrimmed.length || RIGHT_BOUNDARY_RE.test(contTrimmed[len]);

    if (isLeftBoundary && isRightBoundary && (len >= 2 || /^\w+$/.test(suffix))) {
      matchedOverlapLen = len;
      break;
    }
  }

  let stripped =
    matchedOverlapLen > 0
      ? contTrimmed.slice(matchedOverlapLen).trimStart()
      : contTrimmed;

  if (!stripped) {
    return '';
  }

  // If flushedContent ended with a dangling list bullet (e.g. "\n-" or "\n   -")
  // and the continuation starts with a list bullet ("- ") OR an em-dash/en-dash/hyphen ("—", "–", "-"),
  // strip the leading marker so we never produce "-- **Flight A" or "   -—in JPY"
  if (
    DANGLING_BULLET_AT_END_RE.test(flushedContent) &&
    (LEADING_BULLET_RE.test(stripped) || /^[—–\-:]+\s*/.test(stripped))
  ) {
    stripped = stripped.replace(/^(?:[-*•]\s+|[—–\-:]+\s*)/, '');
    if (!stripped) return '';
    return /\s$/.test(flushedContent) ? stripped : ` ${stripped}`;
  }

  // If the continuation starts with a markdown bullet ("- item") after a completed line/sentence,
  // place it on a clean newline instead of gluing it inline
  if (LEADING_BULLET_RE.test(stripped)) {
    return /\n\s*$/.test(flushedContent) ? stripped : `\n${stripped}`;
  }

  const flushedEndsWithSpace = /\s$/.test(flushedContent);
  if (flushedEndsWithSpace) {
    return stripped;
  }

  if (INLINE_LEADING_PUNCTUATION_RE.test(stripped)) {
    return stripped;
  }

  // Allow em-dash/hyphen clause connectors like "—which is..." or "- which is..." when not a list bullet
  if (/^[—–]/.test(stripped)) {
    return stripped;
  }

  return ` ${stripped}`;
}

function isPrefixOfWordBoundedSuffix(flushedTrimmed: string, candidate: string): boolean {
  const normCandidate = candidate.trimStart().toLowerCase();
  if (!normCandidate) return false;

  const lowerFlushed = flushedTrimmed.toLowerCase();
  if (lowerFlushed.startsWith(normCandidate)) {
    return true;
  }

  const maxSuffixLen = Math.min(lowerFlushed.length, 64);
  const startIdx = Math.max(0, lowerFlushed.length - maxSuffixLen);

  for (let i = startIdx; i < lowerFlushed.length; i++) {
    const isBoundary = i === 0 || LEFT_BOUNDARY_RE.test(lowerFlushed[i - 1]);
    if (!isBoundary) continue;
    const suffix = lowerFlushed.slice(i);
    if (suffix.startsWith(normCandidate)) {
      return true;
    }
  }

  return false;
}

function createTurnLinkedAbortController(turnSignal: AbortSignal): AbortController {
  const abortController = new AbortController();
  if (turnSignal.aborted) {
    abortController.abort(turnSignal.reason);
    return abortController;
  }
  const onTurnAbort = () => abortController.abort(turnSignal.reason);
  turnSignal.addEventListener('abort', onTurnAbort, { once: true });
  abortController.signal.addEventListener(
    'abort',
    () => turnSignal.removeEventListener('abort', onTurnAbort),
    { once: true },
  );
  return abortController;
}

export interface PendingContinuation {
  directive: string;
  flushedSnapshot: string;
}

interface DeferredGate {
  promise: Promise<'continue' | 'stop'>;
  resolve: (decision: 'continue' | 'stop') => void;
}

function createDeferredGate(): DeferredGate {
  let resolve!: (decision: 'continue' | 'stop') => void;
  const promise = new Promise<'continue' | 'stop'>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

export interface ActiveAgentStreamState {
  messageId: string;
  seq: number;
  agentId: AgentId;
  waveIndex: number;
  epoch: number;
  abortController: AbortController;
  flushedContent: string;
  effort: Exclude<EffortLevel, 'silent'>;
  thesis: string;
  peerTheses: string[];
  priorSpeakerTheses: Array<{ agentId: AgentId; thesis: string }>;
  pendingContinuation: PendingContinuation | null;
  classificationGate: DeferredGate | null;
  stopped: boolean;
}

export type ActiveAgentStream = ActiveAgentStreamState;

function createActiveStreamState(init: {
  messageId: string;
  seq: number;
  agentId: AgentId;
  waveIndex: number;
  abortController: AbortController;
  effort: Exclude<EffortLevel, 'silent'>;
  thesis: string;
  peerTheses: string[];
  priorSpeakerTheses: Array<{ agentId: AgentId; thesis: string }>;
}): ActiveAgentStreamState {
  return {
    messageId: init.messageId,
    seq: init.seq,
    agentId: init.agentId,
    waveIndex: init.waveIndex,
    epoch: 1,
    abortController: init.abortController,
    flushedContent: '',
    effort: init.effort,
    thesis: init.thesis,
    peerTheses: init.peerTheses,
    priorSpeakerTheses: init.priorSpeakerTheses,
    pendingContinuation: null,
    classificationGate: null,
    stopped: false,
  };
}

export interface RunTurnResult {
  userMessage: PublicWireMessage;
  wavePlans: WaveRoutingPlan[];
  agentMessages: PublicWireMessage[];
  steeringClassification?: SteeringClassification;
}

export interface OrchestratorOptions {
  store: EmanateStore;
  llm: LLMProvider;
  maxA2AWaves?: number;
  streamPaceMs?: number;
}

export class GroupChatOrchestrator extends EventEmitter {
  public readonly store: EmanateStore;
  public readonly llm: LLMProvider;
  public readonly router: TurnRouter;
  public readonly maxA2AWaves: number;
  public readonly streamPaceMs: number;

  private readonly activeStreams = new Map<AgentId, ActiveAgentStreamState>();
  private turnAbortController: AbortController | null = null;
  private activeTurnPromise: Promise<RunTurnResult> | null = null;
  private steeringChain: Promise<void> = Promise.resolve();
  private currentWaveDynamicPromises: Array<Promise<PublicWireMessage>> = [];
  private currentWaveReservedIds: Set<string> = new Set();
  private steeredInCurrentTurn = false;

  constructor(options: OrchestratorOptions) {
    super();
    this.store = options.store;
    this.llm = options.llm;
    this.router = new TurnRouter(options.llm);
    this.maxA2AWaves = options.maxA2AWaves ?? MAX_A2A_WAVES;
    const envPace = Number(process.env.STREAM_PACE_MS ?? 0);
    this.streamPaceMs =
      options.streamPaceMs ?? (Number.isFinite(envPace) && envPace > 0 ? envPace : 0);
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

  /**
   * Epoch-guarded delta emission. Drops any zombie token whose epoch does not match
   * the active stream's current epoch.
   */
  public emitDeltaForEpoch(agentId: AgentId, epoch: number, delta: string): boolean {
    const streamState = this.activeStreams.get(agentId);
    if (!streamState) return false;
    if (streamState.stopped || streamState.epoch !== epoch) {
      return false;
    }
    if (!delta) return false;

    streamState.flushedContent += delta;
    this.store.appendMessageDelta(streamState.messageId, delta);
    this.emitSSE({
      type: 'message.delta',
      id: streamState.messageId,
      seq: streamState.seq,
      agentId,
      delta,
    });
    return true;
  }

  public async stopAll(): Promise<PublicWireMessage[]> {
    this.turnAbortController?.abort();

    for (const streamState of this.activeStreams.values()) {
      streamState.epoch += 1;
      streamState.pendingContinuation = null;
      streamState.stopped = true;
      streamState.classificationGate?.resolve('stop');
      streamState.classificationGate = null;
      streamState.abortController.abort();
    }

    if (this.activeTurnPromise) {
      try {
        await this.activeTurnPromise;
      } catch {
        // Ignore abort errors during stopAll
      }
    }

    return this.getPublicMessages();
  }

  public async reset(): Promise<void> {
    await this.stopAll();
    this.store.reset();
  }

  public async handleUserMessage(content: string): Promise<RunTurnResult> {
    const trimmed = content.trim();
    if (!trimmed) {
      throw new Error('User message content cannot be empty.');
    }

    // If streams are currently active, route through the In-Flight Steering Classifier
    if (this.isStreaming()) {
      return this.handleMidStreamUserMessage(trimmed);
    }

    // Bug 7 fix: If a turn is currently in its pre-stream routing phase (awaiting routeWave before
    // activeStreams is populated), immediately abort the pre-stream routing and start/steer with this new message
    // instead of blocking for the entire previous turn!
    if (this.activeTurnPromise && this.turnAbortController && !this.turnAbortController.signal.aborted) {
      await this.stopAll();
    } else if (this.activeTurnPromise) {
      await this.activeTurnPromise.catch(() => {});
    }

    const userRow = this.store.reserveMessage({
      role: 'user',
      content: trimmed,
      status: 'completed',
    });
    const userMessage = toPublicWireMessage(userRow);
    this.emitSSE({ type: 'message.created', message: userMessage });

    const turnPromise = this.runTurnLoop(userMessage, trimmed);
    this.activeTurnPromise = turnPromise;
    try {
      return await turnPromise;
    } finally {
      if (this.activeTurnPromise === turnPromise) {
        this.activeTurnPromise = null;
      }
    }
  }

  private async handleMidStreamUserMessage(trimmed: string): Promise<RunTurnResult> {
    const userRow = this.store.reserveMessage({
      role: 'user',
      content: trimmed,
      status: 'completed',
    });
    const userMessage = toPublicWireMessage(userRow);
    this.emitSSE({ type: 'message.created', message: userMessage });

    if (isExplicitStopCommand(trimmed)) {
      const activeAgentIds = Array.from(this.activeStreams.keys());
      await this.stopAll();
      return {
        userMessage,
        wavePlans: [],
        agentMessages: this.getPublicMessages().filter((m) => m.role === 'agent'),
        steeringClassification: {
          action: 'stop_all',
          targetAgents: activeAgentIds,
          directive: '',
          standingRule: null,
          clearRuleScope: null,
        },
      };
    }

    // Immediately abort explicitly @mentioned active streams (0ms latency)
    // before awaiting any serialized steering classification.
    const mentioned = extractMentionedAgents(trimmed);
    const preAbortedStreams = new Map<
      AgentId,
      { stream: ActiveAgentStreamState; snapshot: string }
    >();

    for (const agentId of mentioned) {
      const existing = this.activeStreams.get(agentId);
      if (existing && !existing.stopped) {
        const snapshot = existing.flushedContent;
        existing.epoch += 1;
        existing.classificationGate?.resolve('continue');
        existing.classificationGate = createDeferredGate();
        existing.abortController.abort();
        preAbortedStreams.set(agentId, { stream: existing, snapshot });
      }
    }

    let releaseLock!: () => void;
    const prevChain = this.steeringChain;
    this.steeringChain = new Promise<void>((res) => {
      releaseLock = res;
    });

    let classification!: SteeringClassification;
    let freshTurnPromise: Promise<RunTurnResult> | null = null;

    await prevChain;
    try {
      const activeAgentIds = Array.from(this.activeStreams.keys());
      const pausedOnlyStreams = new Map<
        AgentId,
        { stream: ActiveAgentStreamState; snapshot: string }
      >();

      // When no explicit @mention was given, pause token emission on active streams
      // via classificationGate without aborting their upstream HTTP stream yet.
      if (mentioned.length === 0) {
        for (const agentId of activeAgentIds) {
          const existing = this.activeStreams.get(agentId);
          if (existing && !existing.stopped) {
            existing.classificationGate?.resolve('continue');
            existing.classificationGate = createDeferredGate();
            pausedOnlyStreams.set(agentId, {
              stream: existing,
              snapshot: existing.flushedContent,
            });
          }
        }
      }

      classification = await this.router.classifySteering({
        userMessage: trimmed,
        activeAgents: activeAgentIds,
        transcript: this.store.listMessages(),
        activeRules: this.store.listActiveStandingRules(),
        signal: this.turnAbortController?.signal,
      });

      if (!this.turnAbortController || this.turnAbortController.signal.aborted) {
        for (const { stream } of preAbortedStreams.values()) {
          stream.classificationGate?.resolve('stop');
          stream.classificationGate = null;
        }
        for (const { stream } of pausedOnlyStreams.values()) {
          stream.classificationGate?.resolve('stop');
          stream.classificationGate = null;
        }
        return {
          userMessage,
          wavePlans: [],
          agentMessages: this.getPublicMessages().filter((m) => m.role === 'agent'),
          steeringClassification: classification,
        };
      }

      if (classification.clearRuleScope) {
        const cleared = this.store.clearStandingRules(classification.clearRuleScope);
        if (cleared === 0 && classification.clearRuleScope === 'global') {
          this.store.clearStandingRules('all');
        }
      }
      if (classification.standingRule) {
        this.store.addStandingRule({
          scope: classification.standingRule.scope,
          rule: classification.standingRule.rule,
        });
      }

      if (classification.action === 'stop_all') {
        await this.stopAll();
        return {
          userMessage,
          wavePlans: [],
          agentMessages: this.getPublicMessages().filter((m) => m.role === 'agent'),
          steeringClassification: classification,
        };
      }

      if (classification.action === 'new_turn') {
        await this.stopAll();
        freshTurnPromise = this.runTurnLoop(userMessage, trimmed);
        this.activeTurnPromise = freshTurnPromise;
      } else {
        // action === 'steer': Selective Abort & In-Place Continuation
        this.steeredInCurrentTurn = true;
        const directive = classification.directive || trimmed;
        const finalTargetSet = new Set<AgentId>(classification.targetAgents);

        for (const [agentId, { stream, snapshot }] of preAbortedStreams.entries()) {
          if (!finalTargetSet.has(agentId)) {
            stream.pendingContinuation = {
              directive: 'Continue your original thought naturally.',
              flushedSnapshot: snapshot,
            };
            stream.classificationGate?.resolve('continue');
            stream.classificationGate = null;
          }
        }

        // Unpause non-targeted streams without ever having aborted their upstream request
        for (const [agentId, { stream }] of pausedOnlyStreams.entries()) {
          if (!finalTargetSet.has(agentId)) {
            stream.classificationGate?.resolve('continue');
            stream.classificationGate = null;
          }
        }

        for (const targetAgentId of classification.targetAgents) {
          const preAborted = preAbortedStreams.get(targetAgentId);
          if (preAborted) {
            preAborted.stream.pendingContinuation = {
              directive,
              flushedSnapshot: preAborted.snapshot,
            };
            preAborted.stream.classificationGate?.resolve('continue');
            preAborted.stream.classificationGate = null;
            continue;
          }

          const paused = pausedOnlyStreams.get(targetAgentId);
          if (paused && !paused.stream.stopped) {
            paused.stream.epoch += 1;
            paused.stream.pendingContinuation = {
              directive,
              flushedSnapshot: paused.snapshot,
            };
            paused.stream.abortController.abort();
            paused.stream.classificationGate?.resolve('continue');
            paused.stream.classificationGate = null;
            continue;
          }

          const existingStream = this.activeStreams.get(targetAgentId);
          if (existingStream && !existingStream.stopped) {
            existingStream.epoch += 1;
            existingStream.pendingContinuation = {
              directive,
              flushedSnapshot: existingStream.flushedContent,
            };
            existingStream.abortController.abort();
          } else if (
            !existingStream &&
            this.turnAbortController &&
            !this.turnAbortController.signal.aborted
          ) {
            this.spawnDynamicAgentInCurrentWave(targetAgentId, directive);
          }
        }
      }
    } finally {
      releaseLock();
    }

    if (freshTurnPromise) {
      try {
        const res = await freshTurnPromise;
        return { ...res, steeringClassification: classification };
      } finally {
        if (this.activeTurnPromise === freshTurnPromise) {
          this.activeTurnPromise = null;
        }
      }
    }

    if (this.activeTurnPromise) {
      const turnRes = await this.activeTurnPromise;
      return {
        userMessage,
        wavePlans: turnRes.wavePlans,
        agentMessages: this.getPublicMessages().filter((m) => m.role === 'agent'),
        steeringClassification: classification,
      };
    }

    return {
      userMessage,
      wavePlans: [],
      agentMessages: this.getPublicMessages().filter((m) => m.role === 'agent'),
      steeringClassification: classification,
    };
  }

  private spawnDynamicAgentInCurrentWave(agentId: AgentId, directive: string): void {
    const turnSignal = this.turnAbortController?.signal;
    if (!turnSignal || turnSignal.aborted) return;

    const row = this.store.reserveMessage({
      role: 'agent',
      agentId,
      content: '',
      status: 'streaming',
    });
    this.currentWaveReservedIds.add(row.id);

    const abortController = createTurnLinkedAbortController(turnSignal);

    const speaker: SpeakerPlan = {
      agentId,
      effort: 'standard',
      maxTokens: EFFORT_MAX_TOKENS.standard,
      thesis: directive,
      peerTheses: [],
      priorSpeakerTheses: [],
    };

    const streamState = createActiveStreamState({
      messageId: row.id,
      seq: row.seq,
      agentId,
      waveIndex: 0,
      abortController,
      effort: speaker.effort,
      thesis: speaker.thesis,
      peerTheses: [],
      priorSpeakerTheses: [],
    });

    this.activeStreams.set(agentId, streamState);
    this.emitSSE({
      type: 'message.created',
      message: toPublicWireMessage(row),
    });

    const transcriptSnapshot = this.store.listMessages();
    const promise = this.streamSingleAgentInWave({
      speaker,
      row,
      streamState,
      preWaveTranscript: transcriptSnapshot,
      reservedIds: this.currentWaveReservedIds,
      isFinalWave: false,
      turnSignal,
    });
    this.currentWaveDynamicPromises.push(promise);
  }

  private async runTurnLoop(
    userMessage: PublicWireMessage,
    trimmedUserText: string,
  ): Promise<RunTurnResult> {
    this.turnAbortController = new AbortController();
    this.steeredInCurrentTurn = false;
    const turnSignal = this.turnAbortController.signal;

    const wavePlans: WaveRoutingPlan[] = [];
    const agentMessages: PublicWireMessage[] = [];
    let previousWaveSpeakers: AgentId[] = [];
    const turnThesesByAgent: Partial<Record<AgentId, string[]>> = {
      maya: [],
      liam: [],
      zoe: [],
    };

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
              ? trimmedUserText
              : transcriptSnapshot[transcriptSnapshot.length - 1]?.content,
          previousWaveSpeakers,
          turnThesesByAgent,
          signal: turnSignal,
        });
        wavePlans.push(plan);

        if (waveIndex === 0) {
          if (plan.clearRuleScope) {
            const cleared = this.store.clearStandingRules(plan.clearRuleScope);
            if (cleared === 0 && plan.clearRuleScope === 'global') {
              this.store.clearStandingRules('all');
            }
          }
          if (plan.standingRule) {
            this.store.addStandingRule({
              scope: plan.standingRule.scope,
              rule: plan.standingRule.rule,
            });
          }
        }

        if (plan.isResolved || plan.speakers.length === 0) {
          break;
        }

        for (const sp of plan.speakers) {
          turnThesesByAgent[sp.agentId] = [
            ...(turnThesesByAgent[sp.agentId] ?? []),
            sp.thesis,
          ];
        }

        const isFinalWave = waveIndex === this.maxA2AWaves;
        const completedInWave = await this.executeWave(
          waveIndex,
          plan.speakers,
          isFinalWave,
          turnSignal,
        );
        agentMessages.push(...completedInWave);
        previousWaveSpeakers = completedInWave
          .map((m) => m.agentId)
          .filter((id): id is AgentId => Boolean(id));

        if (turnSignal.aborted || this.steeredInCurrentTurn) {
          break;
        }
      }
    } finally {
      this.turnAbortController = null;
      this.currentWaveDynamicPromises = [];
      this.currentWaveReservedIds.clear();
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
    const reservedSlots: Array<{
      speaker: SpeakerPlan;
      row: MessageRow;
      streamState: ActiveAgentStreamState;
    }> = [];

    const reservedIds = new Set<string>();
    this.currentWaveReservedIds = reservedIds;
    this.currentWaveDynamicPromises = [];

    for (const speaker of speakers) {
      const row = this.store.reserveMessage({
        role: 'agent',
        agentId: speaker.agentId,
        content: '',
        status: 'streaming',
      });
      reservedIds.add(row.id);

      const abortController = createTurnLinkedAbortController(turnSignal);

      const streamState = createActiveStreamState({
        messageId: row.id,
        seq: row.seq,
        agentId: speaker.agentId,
        waveIndex,
        abortController,
        effort: speaker.effort,
        thesis: speaker.thesis,
        peerTheses: speaker.peerTheses,
        priorSpeakerTheses: speaker.priorSpeakerTheses,
      });

      this.activeStreams.set(speaker.agentId, streamState);
      reservedSlots.push({ speaker, row, streamState });

      this.emitSSE({
        type: 'message.created',
        message: toPublicWireMessage(row),
      });
    }

    const preWaveTranscript = this.store.listMessages();

    const initialWavePromises = reservedSlots.map(({ speaker, row, streamState }) =>
      this.streamSingleAgentInWave({
        speaker,
        row,
        streamState,
        preWaveTranscript,
        reservedIds,
        isFinalWave,
        turnSignal,
      }),
    );

    const results = await Promise.all(initialWavePromises);
    while (this.currentWaveDynamicPromises.length > 0) {
      const extraPromises = this.currentWaveDynamicPromises.splice(0);
      const extraResults = await Promise.all(extraPromises);
      results.push(...extraResults);
    }

    return results;
  }

  private async streamSingleAgentInWave(params: {
    speaker: SpeakerPlan;
    row: MessageRow;
    streamState: ActiveAgentStreamState;
    preWaveTranscript: MessageRow[];
    reservedIds: Set<string>;
    isFinalWave: boolean;
    turnSignal: AbortSignal;
  }): Promise<PublicWireMessage> {
    const {
      speaker,
      row,
      streamState,
      preWaveTranscript,
      reservedIds,
      isFinalWave,
      turnSignal,
    } = params;
    const { agentId, effort, maxTokens, thesis, peerTheses, priorSpeakerTheses } = speaker;

    let currentContinuation: PendingContinuation | null = null;

    try {
      while (!streamState.stopped && !turnSignal.aborted) {
        const activeEpoch = streamState.epoch;
        const activeSignal = streamState.abortController.signal;

        const activeRules = this.store.listActiveStandingRules(agentId);
        const systemPrompt = buildAgentSystemPrompt(agentId, activeRules, {
          effort,
          thesis,
          peerTheses,
          priorSpeakerTheses,
          isFinalWave,
          ...(currentContinuation
            ? {
                continuation: {
                  flushedContent: currentContinuation.flushedSnapshot,
                  steeringDirective: currentContinuation.directive,
                },
              }
            : {}),
        });

        const transcriptForPrompt = currentContinuation
          ? this.store.listMessages()
          : preWaveTranscript;

        const conversationHistory = buildConversationHistoryForAgent(
          agentId,
          transcriptForPrompt,
          reservedIds,
        );

        const messages: ChatMessage[] = [
          { role: 'system', content: systemPrompt },
          ...(conversationHistory.length > 0
            ? conversationHistory
            : [{ role: 'user' as const, content: 'Please share your perspective.' }]),
          ...(currentContinuation
            ? [
                {
                  role: 'assistant' as const,
                  content: currentContinuation.flushedSnapshot,
                },
                {
                  role: 'user' as const,
                  content: `[Live Redirect: "${currentContinuation.directive}"] Continue directly from the exact end of your partial message above to fulfill "${currentContinuation.directive}". Do NOT repeat or re-list any bullet headers already written in your partial message above, and use ONLY the exact currency/format/length requested in "${currentContinuation.directive}". Output ONLY the continuation suffix.`,
                },
              ]
            : []),
        ];

        const flushedBeforeContinuation = currentContinuation
          ? currentContinuation.flushedSnapshot
          : '';
        const flushedTrimmed = flushedBeforeContinuation.trimEnd();
        let needsSeamNormalization = Boolean(
          currentContinuation && flushedBeforeContinuation.length > 0,
        );
        let seamBuffer = '';
        let disclaimerCheckDone = Boolean(currentContinuation);
        let disclaimerBuffer = '';

        try {
          for await (const rawDelta of this.llm.streamChat({
            messages,
            maxTokens,
            temperature: 0.4,
            signal: activeSignal,
          })) {
            if (
              activeSignal.aborted ||
              streamState.stopped ||
              streamState.epoch !== activeEpoch
            ) {
              break;
            }
            if (!rawDelta) continue;

            if (this.streamPaceMs > 0) {
              await sleepAbortable(this.streamPaceMs, activeSignal);
              if (
                activeSignal.aborted ||
                streamState.stopped ||
                streamState.epoch !== activeEpoch
              ) {
                break;
              }
            }

            if (streamState.classificationGate) {
              const gateDecision = await streamState.classificationGate.promise;
              if (
                gateDecision === 'stop' ||
                activeSignal.aborted ||
                streamState.stopped ||
                streamState.epoch !== activeEpoch
              ) {
                break;
              }
            }

            let deltaToEmit = rawDelta;

            if (!disclaimerCheckDone) {
              disclaimerBuffer += rawDelta;
              if (
                /^I(?:'m|’m|\s+am|\s+can|\s+only)\b/i.test(disclaimerBuffer.trimStart()) &&
                !/[.!]\s/.test(disclaimerBuffer) &&
                disclaimerBuffer.length < 110
              ) {
                continue;
              }
              deltaToEmit = stripLeadingMetaDisclaimer(disclaimerBuffer);
              disclaimerBuffer = '';
              disclaimerCheckDone = true;
              if (!deltaToEmit) continue;
            }

            if (needsSeamNormalization) {
              seamBuffer += deltaToEmit;
              if (
                seamBuffer.length < 48 &&
                isPrefixOfWordBoundedSuffix(flushedTrimmed, seamBuffer)
              ) {
                continue;
              }

              deltaToEmit = normalizeContinuationSeam(flushedBeforeContinuation, seamBuffer);
              seamBuffer = '';
              if (!deltaToEmit) {
                continue;
              }
              needsSeamNormalization = false;
            }

            this.emitDeltaForEpoch(agentId, activeEpoch, deltaToEmit);
          }

          if (
            !disclaimerCheckDone &&
            disclaimerBuffer.length > 0 &&
            !activeSignal.aborted &&
            !streamState.stopped &&
            streamState.epoch === activeEpoch
          ) {
            const cleaned = stripLeadingMetaDisclaimer(disclaimerBuffer);
            disclaimerBuffer = '';
            disclaimerCheckDone = true;
            if (cleaned) {
              this.emitDeltaForEpoch(agentId, activeEpoch, cleaned);
            }
          }

          if (
            needsSeamNormalization &&
            seamBuffer.length > 0 &&
            !activeSignal.aborted &&
            !streamState.stopped &&
            streamState.epoch === activeEpoch
          ) {
            const finalSeamDelta = normalizeContinuationSeam(
              flushedBeforeContinuation,
              seamBuffer,
            );
            seamBuffer = '';
            needsSeamNormalization = false;
            if (finalSeamDelta) {
              this.emitDeltaForEpoch(agentId, activeEpoch, finalSeamDelta);
            }
          }
        } catch (err) {
          if (!activeSignal.aborted && !streamState.stopped && !turnSignal.aborted) {
            throw err;
          }
        }

        // If a mid-stream classification gate is in flight for this stream, wait for it to resolve
        if (!streamState.stopped && !turnSignal.aborted && streamState.classificationGate) {
          const gateDecision = await streamState.classificationGate.promise;
          if (gateDecision === 'stop') {
            break;
          }
        }

        // Check if this stream was aborted specifically for an in-place steering continuation
        if (!streamState.stopped && !turnSignal.aborted && streamState.pendingContinuation) {
          currentContinuation = streamState.pendingContinuation;
          streamState.pendingContinuation = null;
          streamState.abortController = createTurnLinkedAbortController(turnSignal);
          continue;
        }

        break;
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
