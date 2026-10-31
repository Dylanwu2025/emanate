import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { EmanateStore } from '../src/server/db.js';
import {
  createLLMProvider,
  ScriptableMockLLMProvider,
} from '../src/server/llm.js';
import {
  classifySteeringDeterministic,
  detectRuleClearScope,
  isCasualSocialMessage,
  isExplicitStopCommand,
  sanitizeThesisForAgentDomain,
} from '../src/server/router.js';
import {
  GroupChatOrchestrator,
  normalizeContinuationSeam,
  stripLeadingMetaDisclaimer,
  type OrchestratorSSEEvent,
} from '../src/server/orchestrator.js';
import { createAppServer } from '../src/server/index.js';

function waitForCondition(
  predicate: () => boolean,
  timeoutMs = 5000,
  intervalMs = 5,
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (predicate()) {
      resolve();
      return;
    }
    const start = Date.now();
    const timer = setInterval(() => {
      if (predicate()) {
        clearInterval(timer);
        resolve();
      } else if (Date.now() - start > timeoutMs) {
        clearInterval(timer);
        reject(new Error('Timed out waiting for condition'));
      }
    }, intervalMs);
  });
}

describe('Part 3: In-Flight Mid-Stream Steering, Selective Abort, Stop-All, Standing Rules & HTTP/SSE Server', () => {
  let store: EmanateStore;
  let mockLLM: ScriptableMockLLMProvider;

  beforeEach(() => {
    store = new EmanateStore(':memory:');
    mockLLM = new ScriptableMockLLMProvider();
  });

  afterEach(() => {
    store.close();
  });

  describe('a) Selective Per-Agent Mid-Stream Steering (Single Bubble Continuation)', () => {
    it('steers Maya mid-stream into JPY within the exact same messageId and seq while Liam continues uninterrupted', async () => {
      const orchestrator = new GroupChatOrchestrator({ store, llm: mockLLM });
      const sseEvents: OrchestratorSSEEvent[] = [];
      orchestrator.onEvent((ev) => sseEvents.push(ev));

      // Wave 0 routing: Maya and Liam both speak
      mockLLM.enqueueJSON({
        response: {
          maya: { effort: 'standard', thesis: 'Compare Flight A and Flight B in USD' },
          liam: { effort: 'standard', thesis: 'Describe KIX to Gion transit and tofu kaiseki' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      // Maya initial stream (slow enough to be steered after 2 chunks)
      mockLLM.enqueueStream({
        match: (opts) =>
          opts.messages[0].content.includes('You are Maya') &&
          !opts.messages[0].content.includes('Mid-Stream Steering Continuation'),
        chunks: [
          'Looking at the flight options, ',
          'Flight A is $650 and Flight B is $820 ',
          'which leaves plenty of room in our $4,800 USD budget.',
          ' We can also book Ryokan Gion for $380 USD.',
        ],
        delayMs: 30,
      });

      // Liam stream (should NEVER be aborted)
      mockLLM.enqueueStream({
        match: (opts) => opts.messages[0].content.includes('You are Liam'),
        chunks: [
          'For Day 1 dinner, ',
          'Arashiyama has a wonderful tofu kaiseki spot ',
          'that stays open until 7:30 PM and is 100% shellfish-free.',
        ],
        delayMs: 25,
      });

      // Maya surgical in-place replacement stream after mid-stream steering redirect
      mockLLM.enqueueStream({
        match: (opts) =>
          opts.messages[0].content.includes('You are Maya') &&
          opts.messages[0].content.includes('Mid-Stream Steering Continuation'),
        chunks: [
          'Looking at the flight options, ',
          'Flight A is ¥97,500 and Flight B is ¥123,000 ',
          'which leaves plenty of room in our ¥720,000 JPY budget.',
        ],
        delayMs: 10,
      });

      // Start Turn 1 (do not await yet so we can steer mid-stream)
      const turn1Promise = orchestrator.handleUserMessage(
        'What are our flight options and Day 1 dinner plan?',
      );

      // Wait until Maya has emitted 2 chunks and Liam has emitted at least 1 chunk
      await waitForCondition(() => {
        const mayaStream = orchestrator.getActiveStreams().get('maya');
        const liamStream = orchestrator.getActiveStreams().get('liam');
        return (
          Boolean(mayaStream && mayaStream.flushedContent.includes('Flight B is $820 ')) &&
          Boolean(liamStream && liamStream.flushedContent.includes('For Day 1 dinner, '))
        );
      });

      const mayaPreSteerState = orchestrator.getActiveStreams().get('maya')!;
      const liamPreSteerState = orchestrator.getActiveStreams().get('liam')!;
      const mayaInitialId = mayaPreSteerState.messageId;
      const mayaInitialSeq = mayaPreSteerState.seq;
      const liamInitialAbortSignal = liamPreSteerState.abortController.signal;

      expect(mayaInitialSeq).toBe(2);
      expect(liamPreSteerState.seq).toBe(3);

      // Send mid-stream steering message targeting Maya ONLY
      const steerPromise = orchestrator.handleUserMessage(
        'Maya, convert all prices to Japanese Yen instead',
      );

      // Verify Liam's original AbortSignal was NEVER aborted!
      expect(liamInitialAbortSignal.aborted).toBe(false);

      await Promise.all([turn1Promise, steerPromise]);

      // Verify Liam's original AbortSignal remained un-aborted through completion
      expect(liamInitialAbortSignal.aborted).toBe(false);

      const allMessages = orchestrator.getPublicMessages();
      // Expected rows:
      // seq=1: User initial question
      // seq=2: Maya (single bubble containing pre-steer USD prefix + post-steer JPY continuation!)
      // seq=3: Liam (uninterrupted full message)
      // seq=4: User steering message ("Maya, convert all prices to Japanese Yen instead")
      expect(allMessages).toHaveLength(4);
      expect(allMessages.map((m) => ({ seq: m.seq, role: m.role, agentId: m.agentId }))).toEqual([
        { seq: 1, role: 'user', agentId: null },
        { seq: 2, role: 'agent', agentId: 'maya' },
        { seq: 3, role: 'agent', agentId: 'liam' },
        { seq: 4, role: 'user', agentId: null },
      ]);

      const mayaFinalMsg = allMessages[1];
      expect(mayaFinalMsg.id).toBe(mayaInitialId);
      expect(mayaFinalMsg.seq).toBe(mayaInitialSeq);
      expect(mayaFinalMsg.status).toBe('completed');
      expect(mayaFinalMsg.content).toBe(
        'Looking at the flight options, Flight A is ¥97,500 and Flight B is ¥123,000 which leaves plenty of room in our ¥720,000 JPY budget.',
      );
      // Verify the aborted tail chunks from stream #1 never leaked into Maya's message
      expect(mayaFinalMsg.content).not.toContain('$4,800 USD');
      expect(mayaFinalMsg.content).not.toContain('$650');

      const liamFinalMsg = allMessages[2];
      expect(liamFinalMsg.status).toBe('completed');
      expect(liamFinalMsg.content).toBe(
        'For Day 1 dinner, Arashiyama has a wonderful tofu kaiseki spot that stays open until 7:30 PM and is 100% shellfish-free.',
      );

      // Verify message.created was emitted only ONCE for Maya (no duplicate bubble created)
      const mayaCreatedEvents = sseEvents.filter(
        (e) => e.type === 'message.created' && e.message.agentId === 'maya',
      );
      expect(mayaCreatedEvents).toHaveLength(1);
    });

    it('spawns a non-streaming agent mid-wave when addressed by name without interrupting active streams', async () => {
      const orchestrator = new GroupChatOrchestrator({ store, llm: mockLLM });

      // Wave 0: Only Maya is streaming initially
      mockLLM.enqueueJSON({
        response: {
          maya: { effort: 'standard', thesis: 'Share Flight A and Flight B pricing' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      mockLLM.enqueueStream({
        match: /You are Maya/,
        chunks: ['Flight A is $650 ', 'and Flight B is $820 ', 'for the group.'],
        delayMs: 25,
      });

      mockLLM.enqueueStream({
        match: /You are Zoe/,
        chunks: ['Sam has a severe shellfish allergy and no one wakes before 7:30 AM.'],
        delayMs: 10,
      });

      const turnPromise = orchestrator.handleUserMessage('Tell me about the flights.');

      await waitForCondition(() => {
        const maya = orchestrator.getActiveStreams().get('maya');
        return Boolean(maya && maya.flushedContent.includes('Flight A is $650 '));
      });

      const mayaSignal = orchestrator.getActiveStreams().get('maya')!.abortController.signal;

      // User addresses @Zoe while Maya is actively streaming
      const callZoePromise = orchestrator.handleUserMessage('@Zoe what about allergies and wakeup times?');

      expect(mayaSignal.aborted).toBe(false);

      await Promise.all([turnPromise, callZoePromise]);

      const messages = orchestrator.getPublicMessages();
      expect(messages.map((m) => ({ seq: m.seq, agentId: m.agentId }))).toEqual([
        { seq: 1, agentId: null },
        { seq: 2, agentId: 'maya' },
        { seq: 3, agentId: null },
        { seq: 4, agentId: 'zoe' },
      ]);
      expect(messages[1].content).toBe('Flight A is $650 and Flight B is $820 for the group.');
      expect(messages[3].content).toBe(
        'Sam has a severe shellfish allergy and no one wakes before 7:30 AM.',
      );
    });
  });

  describe('b) Global Mid-Stream Steering', () => {
    it('pivots all currently active streams in-place when given a global redirect', async () => {
      const orchestrator = new GroupChatOrchestrator({ store, llm: mockLLM });

      mockLLM.enqueueJSON({
        response: {
          maya: { effort: 'standard', thesis: 'Cover flights' },
          liam: { effort: 'standard', thesis: 'Cover transit' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      mockLLM.enqueueStream({
        match: (opts) =>
          opts.messages[0].content.includes('You are Maya') &&
          !opts.messages[0].content.includes('Mid-Stream Steering Continuation'),
        chunks: ['We have two flights: ', 'Flight A and Flight B ', 'with different schedules...'],
        delayMs: 30,
      });

      mockLLM.enqueueStream({
        match: (opts) =>
          opts.messages[0].content.includes('You are Liam') &&
          !opts.messages[0].content.includes('Mid-Stream Steering Continuation'),
        chunks: ['Transit from KIX ', 'takes 2 hours 15 minutes ', 'by train and taxi...'],
        delayMs: 30,
      });

      // Continuation streams for both Maya and Liam in uppercase
      mockLLM.enqueueStream({
        match: (opts) =>
          opts.messages[0].content.includes('You are Maya') &&
          opts.messages[0].content.includes('Mid-Stream Steering Continuation'),
        chunks: ['—BOOK FLIGHT B AT $820!'],
        delayMs: 5,
      });

      mockLLM.enqueueStream({
        match: (opts) =>
          opts.messages[0].content.includes('You are Liam') &&
          opts.messages[0].content.includes('Mid-Stream Steering Continuation'),
        chunks: ['—TAKE THE HARUKA EXPRESS!'],
        delayMs: 5,
      });

      const turnPromise = orchestrator.handleUserMessage('How do we get to Kyoto?');

      await waitForCondition(() => {
        const maya = orchestrator.getActiveStreams().get('maya');
        const liam = orchestrator.getActiveStreams().get('liam');
        return (
          Boolean(maya && maya.flushedContent.includes('We have two flights: ')) &&
          Boolean(liam && liam.flushedContent.includes('Transit from KIX '))
        );
      });

      const mayaId = orchestrator.getActiveStreams().get('maya')!.messageId;
      const liamId = orchestrator.getActiveStreams().get('liam')!.messageId;

      const globalSteerPromise = orchestrator.handleUserMessage(
        'Actually, wrap up in one short uppercase clause',
      );

      await Promise.all([turnPromise, globalSteerPromise]);

      const messages = orchestrator.getPublicMessages();
      const mayaMsg = messages.find((m) => m.agentId === 'maya')!;
      const liamMsg = messages.find((m) => m.agentId === 'liam')!;

      expect(mayaMsg.id).toBe(mayaId);
      expect(mayaMsg.seq).toBe(2);
      expect(mayaMsg.content).toContain('—BOOK FLIGHT B AT $820!');
      expect(mayaMsg.content).not.toContain('with different schedules');

      expect(liamMsg.id).toBe(liamId);
      expect(liamMsg.seq).toBe(3);
      expect(liamMsg.content).toContain('—TAKE THE HARUKA EXPRESS!');
      expect(liamMsg.content).not.toContain('by train and taxi');
    });
  });

  describe('c) Post-Abort Zombie Token Guard & Seam Normalizer', () => {
    it('drops late zombie tokens from a stale epoch and deduplicates boundary words at the splice seam', async () => {
      // 1. Unit test normalizeContinuationSeam directly
      expect(
        normalizeContinuationSeam(
          'Looking at the flight options, Flight A is $650 and Flight B is $820 ',
          'Flight B is $820 — which is ¥97,500 and ¥123,000.',
        ),
      ).toBe('— which is ¥97,500 and ¥123,000.');

      expect(
        normalizeContinuationSeam(
          'Flight A costs ',
          'costs ¥97,500 in Japanese Yen.',
        ),
      ).toBe('¥97,500 in Japanese Yen.');

      expect(
        normalizeContinuationSeam(
          'Flight B is $820',
          'in Japanese Yen that is ¥123,000.',
        ),
      ).toBe(' in Japanese Yen that is ¥123,000.');

      expect(
        normalizeContinuationSeam(
          'Flight B is $820',
          '—which is ¥123,000 in JPY.',
        ),
      ).toBe('—which is ¥123,000 in JPY.');

      // Dangling bullet marker at end of flushedContent + leading bullet in continuation -> never "-- **"
      expect(
        normalizeContinuationSeam(
          '- **Ryokan Gion**: $380/night, strict 6:00 PM check-in cutoff.\n-',
          '- **Hotel Granvia Kyoto**: ¥43,500/night with 24-hour check-in.',
        ),
      ).toBe(' **Hotel Granvia Kyoto**: ¥43,500/night with 24-hour check-in.');

      // Completed sentence + leading bullet in continuation -> placed on a clean newline
      expect(
        normalizeContinuationSeam(
          'In JPY, Flight A is ¥97,500 and Flight B is ¥123,000.',
          '- **Hotel Granvia Kyoto**: ¥43,500/night.',
        ),
      ).toBe('\n- **Hotel Granvia Kyoto**: ¥43,500/night.');

      // 2. Test epoch guard rejecting zombie tokens from an older epoch
      const orchestrator = new GroupChatOrchestrator({ store, llm: mockLLM });
      mockLLM.enqueueJSON({
        response: {
          maya: { effort: 'standard', thesis: 'Share flight prices' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      mockLLM.enqueueStream({
        chunks: ['Chunk 1 ', 'Chunk 2 ', 'Chunk 3'],
        delayMs: 30,
      });

      const turnPromise = orchestrator.handleUserMessage('Flight prices?');
      await waitForCondition(() => {
        const maya = orchestrator.getActiveStreams().get('maya');
        return Boolean(maya && maya.flushedContent.includes('Chunk 1 '));
      });

      const mayaStream = orchestrator.getActiveStreams().get('maya')!;
      const staleEpoch = mayaStream.epoch; // 1
      mayaStream.epoch = 2; // Simulate epoch increment on abort

      // Attempt to emit a zombie token with staleEpoch = 1
      const acceptedZombie = orchestrator.emitDeltaForEpoch('maya', staleEpoch, 'ZOMBIE_TOKEN ');
      expect(acceptedZombie).toBe(false);
      expect(store.getMessageById(mayaStream.messageId)!.content).not.toContain('ZOMBIE_TOKEN');

      // Valid epoch = 2 token IS accepted
      const acceptedValid = orchestrator.emitDeltaForEpoch('maya', 2, 'Valid Epoch 2.');
      expect(acceptedValid).toBe(true);
      expect(store.getMessageById(mayaStream.messageId)!.content).toBe('Chunk 1 Valid Epoch 2.');

      await orchestrator.stopAll();
      await turnPromise;
    });
  });

  describe('d) Instant Stop-All & A2A Wave Cancellation', () => {
    it('recognizes stop commands in <5ms and immediately halts all active streams and pending A2A waves', async () => {
      const t0 = performance.now();
      expect(isExplicitStopCommand('stop')).toBe(true);
      expect(isExplicitStopCommand('/stop')).toBe(true);
      expect(isExplicitStopCommand('everyone stop')).toBe(true);
      expect(isExplicitStopCommand('Stop talking!')).toBe(true);
      const stopCheckMs = performance.now() - t0;
      expect(stopCheckMs).toBeLessThan(5);

      const orchestrator = new GroupChatOrchestrator({ store, llm: mockLLM });
      const sseEvents: OrchestratorSSEEvent[] = [];
      orchestrator.onEvent((ev) => sseEvents.push(ev));

      // Wave 0: Maya and Liam start long streams
      mockLLM.enqueueJSON({
        response: {
          maya: { effort: 'deep', thesis: 'Detailed flight breakdown' },
          liam: { effort: 'deep', thesis: 'Detailed food itinerary' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      // Also enqueue a Wave 1 A2A JSON that would trigger Zoe if Wave 1 were ever reached
      mockLLM.enqueueJSON({
        response: {
          isResolved: false,
          maya: { effort: 'silent', thesis: '' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'standard', thesis: 'Should never be reached after stop' },
        },
      });

      mockLLM.enqueueStream({
        match: /You are Maya/,
        chunks: ['Maya part 1 ', 'Maya part 2 ', 'Maya part 3 ', 'Maya part 4'],
        delayMs: 35,
      });
      mockLLM.enqueueStream({
        match: /You are Liam/,
        chunks: ['Liam part 1 ', 'Liam part 2 ', 'Liam part 3 ', 'Liam part 4'],
        delayMs: 35,
      });

      const turnPromise = orchestrator.handleUserMessage('Give us a full breakdown.');

      await waitForCondition(() => {
        const maya = orchestrator.getActiveStreams().get('maya');
        const liam = orchestrator.getActiveStreams().get('liam');
        return (
          Boolean(maya && maya.flushedContent.includes('Maya part 1 ')) &&
          Boolean(liam && liam.flushedContent.includes('Liam part 1 '))
        );
      });

      // Send "stop" while both streams are in flight
      const stopTurnPromise = orchestrator.handleUserMessage('stop');
      await Promise.all([turnPromise, stopTurnPromise]);

      expect(orchestrator.isStreaming()).toBe(false);

      const messages = orchestrator.getPublicMessages();
      // No Zoe message from Wave 1 was ever created!
      expect(messages.some((m) => m.agentId === 'zoe')).toBe(false);

      const mayaMsg = messages.find((m) => m.agentId === 'maya')!;
      const liamMsg = messages.find((m) => m.agentId === 'liam')!;
      expect(mayaMsg.status).toBe('completed');
      expect(liamMsg.status).toBe('completed');
      expect(mayaMsg.content).not.toContain('Maya part 4');
      expect(liamMsg.content).not.toContain('Liam part 4');
      expect(sseEvents.at(-1)?.type).toBe('turn.idle');
    });
  });

  describe('e) Standing Instructions Mid-Stream + Persistence Across Reload', () => {
    it('applies standing instructions mid-stream, persists them in SQLite across orchestrator reload, and supports superseding/clearing', async () => {
      const orchestrator1 = new GroupChatOrchestrator({ store, llm: mockLLM });

      // Turn 1: Maya streams flight prices; user steers her mid-stream with a standing JPY instruction
      mockLLM.enqueueJSON({
        response: {
          maya: { effort: 'standard', thesis: 'Quote flight prices' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      mockLLM.enqueueStream({
        match: (opts) =>
          opts.messages[0].content.includes('You are Maya') &&
          !opts.messages[0].content.includes('Mid-Stream Steering Continuation'),
        chunks: ['Flight A is $650 ', 'and Flight B is $820 ', 'in USD.'],
        delayMs: 30,
      });

      mockLLM.enqueueStream({
        match: (opts) =>
          opts.messages[0].content.includes('You are Maya') &&
          opts.messages[0].content.includes('Mid-Stream Steering Continuation'),
        chunks: ['—which is ¥97,500 and ¥123,000 in JPY.'],
        delayMs: 5,
      });

      const t1 = orchestrator1.handleUserMessage('How much are the flights?');
      await waitForCondition(() => {
        const maya = orchestrator1.getActiveStreams().get('maya');
        return Boolean(maya && maya.flushedContent.includes('Flight A is $650 '));
      });

      const steer = orchestrator1.handleUserMessage(
        'Maya, always convert all prices to Japanese Yen from now on',
      );
      await Promise.all([t1, steer]);

      // Verify the standing rule was persisted in SQLite for 'maya'
      const activeRulesForMaya = store.listActiveStandingRules('maya');
      expect(activeRulesForMaya).toHaveLength(1);
      expect(activeRulesForMaya[0].scope).toBe('maya');
      expect(activeRulesForMaya[0].rule).toMatch(/Japanese Yen/i);

      // Verify the mid-stream continuation prompt itself already included the newly persisted standing rule!
      const continuationCall = mockLLM.streamCalls[1];
      expect(continuationCall.messages[0].content).toContain('Active Standing Instructions');
      expect(continuationCall.messages[0].content).toContain('Japanese Yen');

      // Simulate browser / server reload by creating a brand-new GroupChatOrchestrator backed by the same SQLite store
      const mockLLM2 = new ScriptableMockLLMProvider();
      const orchestrator2 = new GroupChatOrchestrator({ store, llm: mockLLM2 });

      expect(orchestrator2.getPublicMessages()).toEqual(orchestrator1.getPublicMessages());

      // Turn 2 after reload: Ask Maya about hotel prices without mentioning Yen
      mockLLM2.enqueueJSON({
        response: {
          maya: { effort: 'light', thesis: 'Quote Ryokan Gion and Hotel Granvia prices' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });
      mockLLM2.enqueueStream({
        match: /You are Maya/,
        chunks: ['Ryokan Gion is ¥57,000/night and Hotel Granvia is ¥43,500/night.'],
      });
      mockLLM2.enqueueJSON({
        response: {
          isResolved: true,
          maya: { effort: 'silent', thesis: '' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      await orchestrator2.handleUserMessage('What do the Kyoto hotels cost per night?');

      // Verify Maya's Turn 2 prompt after reload still contains the standing rule from Turn 1!
      const turn2MayaPrompt = mockLLM2.streamCalls[0].messages[0].content;
      expect(turn2MayaPrompt).toContain('Active Standing Instructions');
      expect(turn2MayaPrompt).toContain('Japanese Yen');

      // Also verify deterministic role-alias targeting ("budget person")
      const roleClassified = classifySteeringDeterministic(
        'Budget person, always include per-person totals',
        ['maya', 'liam'],
      );
      expect(roleClassified.targetAgents).toEqual(['maya']);

      // Verify Bug 1 fix: clearing haiku or prefix rules ("Stop using haikus...", "Stop starting replies with [KYOTO]...")
      expect(
        detectRuleClearScope('Stop using haikus, answer in 1 normal sentence.', []),
      ).toBe('global');
      expect(
        detectRuleClearScope('Stop starting replies with [KYOTO], answer normally.', []),
      ).toBe('global');

      // Verify Bug 2 & 6 seam and disclaimer helpers
      expect(
        normalizeContinuationSeam('   - **Notes**\n   -', '—in Euros, that is €600.'),
      ).toBe(' in Euros, that is €600.');
      expect(
        stripLeadingMetaDisclaimer(
          "I can't provide hotel options, but transit from KIX to Gion takes 2 hours 15 minutes.",
        ),
      ).toBe('Transit from KIX to Gion takes 2 hours 15 minutes.');
      expect(isCasualSocialMessage('hey team!')).toBe(true);
      expect(isCasualSocialMessage('thanks!')).toBe(true);
      expect(
        sanitizeThesisForAgentDomain(
          'maya',
          'Flight A is $600 departing at 10:00 AM',
          false,
        ),
      ).not.toContain('$600');
    });
  });

  describe('f) HTTP + SSE Server Integration & Live OpenAI Mid-Stream Steering', () => {
    it('serves GET /api/state, GET /api/stream SSE, POST /api/messages, POST /api/stop, and POST /api/reset with zero leaked internal fields', async () => {
      const app = createAppServer({ store, llm: mockLLM });
      const { url } = await app.listen(0);

      try {
        // Connect an SSE listener to GET /api/stream
        const sseAbort = new AbortController();
        const receivedSSE: OrchestratorSSEEvent[] = [];

        const sseRes = await fetch(`${url}/api/stream`, { signal: sseAbort.signal });
        expect(sseRes.status).toBe(200);
        expect(sseRes.headers.get('content-type')).toContain('text/event-stream');

        const reader = sseRes.body!.getReader();
        const decoder = new TextDecoder();
        const readLoop = (async () => {
          let buf = '';
          try {
            while (true) {
              const { value, done } = await reader.read();
              if (done) break;
              buf += decoder.decode(value, { stream: true });
              const lines = buf.split('\n');
              buf = lines.pop() ?? '';
              for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed.startsWith('data:')) {
                  receivedSSE.push(JSON.parse(trimmed.slice(5).trim()));
                }
              }
            }
          } catch {
            // SSE closed
          }
        })();

        mockLLM.enqueueJSON({
          response: {
            maya: { effort: 'light', thesis: 'State budget' },
            liam: { effort: 'silent', thesis: '' },
            zoe: { effort: 'silent', thesis: '' },
          },
        });
        mockLLM.enqueueStream({
          chunks: ['Total budget is ', '$4,800 USD.'],
          delayMs: 5,
        });
        mockLLM.enqueueJSON({
          response: {
            isResolved: true,
            maya: { effort: 'silent', thesis: '' },
            liam: { effort: 'silent', thesis: '' },
            zoe: { effort: 'silent', thesis: '' },
          },
        });

        const postRes = await fetch(`${url}/api/messages`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: 'What is our total budget?' }),
        });
        expect(postRes.status).toBe(200);
        const postData = (await postRes.json()) as {
          ok: boolean;
          messages: Array<Record<string, unknown>>;
        };
        expect(postData.ok).toBe(true);
        expect(postData.messages).toHaveLength(2);

        // Verify GET /api/state returns exact seq order and zero internal fields
        const stateRes = await fetch(`${url}/api/state`);
        const stateData = (await stateRes.json()) as {
          messages: Array<Record<string, unknown>>;
        };
        expect(stateData.messages).toEqual(postData.messages);
        for (const msg of stateData.messages) {
          expect(Object.keys(msg).sort()).toEqual(
            ['agentId', 'agentName', 'content', 'createdAt', 'id', 'role', 'seq', 'status'].sort(),
          );
        }

        await waitForCondition(() => receivedSSE.some((e) => e.type === 'turn.idle'));
        expect(receivedSSE.some((e) => e.type === 'message.created')).toBe(true);
        expect(receivedSSE.some((e) => e.type === 'message.delta')).toBe(true);
        expect(receivedSSE.some((e) => e.type === 'message.done')).toBe(true);

        // Verify POST /api/stop and POST /api/reset
        const stopRes = await fetch(`${url}/api/stop`, { method: 'POST' });
        expect(stopRes.status).toBe(200);

        const resetRes = await fetch(`${url}/api/reset`, { method: 'POST' });
        expect(resetRes.status).toBe(200);
        const afterReset = (await (await fetch(`${url}/api/state`)).json()) as {
          messages: unknown[];
        };
        expect(afterReset.messages).toEqual([]);

        sseAbort.abort();
        await readLoop;
      } finally {
        await app.close();
      }
    });

    it(
      'steers a live OpenAI gpt-4o-mini stream mid-sentence for Maya while Liam continues streaming uninterrupted',
      async () => {
        const liveLLM = createLLMProvider();
        if (!liveLLM.config.apiKey) {
          return;
        }

        // Use streamPaceMs: 35 so live OpenAI tokens flush at a steady human reading pace
        // and we can deterministically intercept Maya mid-sentence after she states USD prices
        const orchestrator = new GroupChatOrchestrator({
          store,
          llm: liveLLM,
          streamPaceMs: 35,
          maxA2AWaves: 1,
        });

        const turnPromise = orchestrator.handleUserMessage(
          '@Maya and @Liam: Maya, give a detailed breakdown of all flight and Kyoto/Osaka hotel prices in USD. Liam, explain the KIX to Gion transit time and Nishiki Market hours.',
        );

        // Wait until Maya has started streaming her USD breakdown (at least 40 chars flushed)
        // and Liam is also actively streaming
        await waitForCondition(
          () => {
            const maya = orchestrator.getActiveStreams().get('maya');
            const liam = orchestrator.getActiveStreams().get('liam');
            return (
              Boolean(maya && maya.flushedContent.length >= 40) &&
              Boolean(liam && liam.flushedContent.length >= 15)
            );
          },
          20000,
          15,
        );

        const mayaStreamBefore = orchestrator.getActiveStreams().get('maya')!;
        const liamStreamBefore = orchestrator.getActiveStreams().get('liam')!;
        const mayaIdBefore = mayaStreamBefore.messageId;
        const mayaSeqBefore = mayaStreamBefore.seq;
        const mayaPrefixSnapshot = mayaStreamBefore.flushedContent;
        const liamAbortSignalBefore = liamStreamBefore.abortController.signal;

        // Steer ONLY Maya mid-stream to Japanese Yen (JPY)
        const steerPromise = orchestrator.handleUserMessage(
          'Maya, convert all prices to Japanese Yen (JPY) instead!',
        );

        // Liam's stream must NOT be aborted by Maya's selective redirect
        expect(liamAbortSignalBefore.aborted).toBe(false);

        await Promise.all([turnPromise, steerPromise]);

        expect(liamAbortSignalBefore.aborted).toBe(false);

        const finalMessages = orchestrator.getPublicMessages();
        const mayaMessages = finalMessages.filter((m) => m.agentId === 'maya');
        const liamMessages = finalMessages.filter((m) => m.agentId === 'liam');

        // Maya must still have only ONE message bubble with the exact same id and seq!
        expect(mayaMessages).toHaveLength(1);
        expect(mayaMessages[0].id).toBe(mayaIdBefore);
        expect(mayaMessages[0].seq).toBe(mayaSeqBefore);
        expect(mayaMessages[0].status).toBe('completed');

        // Maya's bubble was surgically updated in-place to JPY (¥ / JPY / 97,500 / 123,000 / 57,000 / 43,500)
        expect(mayaMessages[0].content.length).toBeGreaterThan(20);
        expect(mayaMessages[0].content).toMatch(/¥|JPY|Yen|97,500|123,000|57,000|43,500/i);
        expect(mayaMessages[0].content).not.toContain('-- **');

        // Liam completed his message uninterrupted
        expect(liamMessages).toHaveLength(1);
        expect(liamMessages[0].status).toBe('completed');
        expect(liamMessages[0].content).toMatch(/2\s*hours?\s*15|2h\s*15m|Nishiki|11:00/i);
      },
      45000,
    );
  });
});
