import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { EmanateStore } from '../src/server/db.js';
import { EFFORT_MAX_TOKENS } from '../src/server/agents.js';
import {
  createLLMProvider,
  ScriptableMockLLMProvider,
} from '../src/server/llm.js';
import {
  applyAsymmetricDedup,
  computeDeterministicSpeakerOrder,
  TurnRouter,
} from '../src/server/router.js';
import {
  GroupChatOrchestrator,
  MAX_A2A_WAVES,
  type OrchestratorSSEEvent,
} from '../src/server/orchestrator.js';

describe('Part 2: Turn Router, Asymmetric Dedup, Concurrent Streaming & Bounded A2A Waves', () => {
  let store: EmanateStore;
  let mockLLM: ScriptableMockLLMProvider;

  beforeEach(() => {
    store = new EmanateStore(':memory:');
    mockLLM = new ScriptableMockLLMProvider();
  });

  afterEach(() => {
    store.close();
  });

  describe('1. Silence / Non-Repetition Gate & Asymmetric Priority', () => {
    it('keeps unrelated agents silent (0 message bubbles created) and orders @mentioned agents first', async () => {
      expect(computeDeterministicSpeakerOrder('Hey @Zoe, does anyone have allergies?')).toEqual([
        'zoe',
        'maya',
        'liam',
      ]);
      expect(computeDeterministicSpeakerOrder('@Liam and @Maya what do you think?')).toEqual([
        'liam',
        'maya',
        'zoe',
      ]);

      const orchestrator = new GroupChatOrchestrator({ store, llm: mockLLM });
      const events: OrchestratorSSEEvent[] = [];
      orchestrator.onEvent((e) => events.push(e));

      // Wave 0: Only Zoe speaks; Maya and Liam are silent
      mockLLM.enqueueJSON({
        response: {
          maya: { effort: 'silent', thesis: '' },
          liam: { effort: 'silent', thesis: '' },
          zoe: {
            effort: 'standard',
            thesis: 'Share Sam shellfish allergy and Priya vegetarian diet',
          },
        },
      });

      mockLLM.enqueueStream({
        match: /You are Zoe/,
        chunks: ['Sam has a severe shellfish allergy ', 'and Priya is strictly vegetarian.'],
        delayMs: 5,
      });

      // Wave 1 A2A check: resolved
      mockLLM.enqueueJSON({
        response: {
          isResolved: true,
          maya: { effort: 'silent', thesis: '' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      const result = await orchestrator.handleUserMessage(
        'Are there any dietary restrictions in our group?',
      );

      expect(result.agentMessages).toHaveLength(1);
      expect(result.agentMessages[0].agentId).toBe('zoe');
      expect(result.agentMessages[0].seq).toBe(2);

      const dbRows = store.listMessages();
      expect(dbRows).toHaveLength(2); // 1 user + 1 zoe (0 bubbles for maya/liam)
      expect(events.at(-1)?.type).toBe('turn.idle');
    });

    it('applies asymmetric speaker priority and silences later speakers with redundant theses', async () => {
      const { decisions, speakers } = applyAsymmetricDedup(
        {
          maya: {
            effort: 'standard',
            thesis: 'Compare Flight A ($650) and Flight B ($820) arrival times',
          },
          liam: {
            effort: 'standard',
            thesis: 'Provide KIX to Gion 2h 15m airport transit duration',
          },
          zoe: {
            effort: 'light',
            // Redundant with Maya's thesis -> should be downgraded to 'silent'
            thesis: 'Compare Flight A ($650) and Flight B ($820) arrival times',
          },
        },
        ['maya', 'liam', 'zoe'],
      );

      expect(decisions.zoe.effort).toBe('silent');
      expect(speakers).toHaveLength(2);

      // Speaker 1 (Maya) has empty peerTheses
      expect(speakers[0].agentId).toBe('maya');
      expect(speakers[0].peerTheses).toEqual([]);

      // Speaker 2 (Liam) receives Speaker 1's thesis so he adds only his distinct angle
      expect(speakers[1].agentId).toBe('liam');
      expect(speakers[1].peerTheses).toEqual([
        'Maya is covering: Compare Flight A ($650) and Flight B ($820) arrival times',
      ]);

      // Verify the orchestrator injects Speaker 1's thesis into Speaker 2's system prompt
      const orchestrator = new GroupChatOrchestrator({ store, llm: mockLLM });
      mockLLM.enqueueJSON({
        response: {
          maya: {
            effort: 'standard',
            thesis: 'Cover Flight B $820 price and Ryokan Gion 6:00 PM check-in cutoff',
          },
          liam: {
            effort: 'standard',
            thesis: 'Cover 2h 15m KIX to Gion transit duration',
          },
          zoe: { effort: 'silent', thesis: '' },
        },
      });
      mockLLM.enqueueStream({
        match: /You are Maya/,
        chunks: ['Flight B is $820 and Ryokan Gion cuts off at 6:00 PM.'],
      });
      mockLLM.enqueueStream({
        match: /You are Liam/,
        chunks: ['KIX to Gion takes 2 hours 15 minutes.'],
      });
      mockLLM.enqueueJSON({
        response: {
          isResolved: true,
          maya: { effort: 'silent', thesis: '' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      await orchestrator.handleUserMessage('How do Flight B and Ryokan Gion work on Day 1?');

      expect(mockLLM.streamCalls).toHaveLength(2);
      const mayaSysPrompt = mockLLM.streamCalls[0].messages[0].content;
      const liamSysPrompt = mockLLM.streamCalls[1].messages[0].content;

      expect(mayaSysPrompt).not.toContain('Asymmetric Anti-Repetition Notice');
      expect(liamSysPrompt).toContain('Asymmetric Anti-Repetition Notice');
      expect(liamSysPrompt).toContain(
        'Maya is covering: Cover Flight B $820 price and Ryokan Gion 6:00 PM check-in cutoff',
      );
      expect(liamSysPrompt).toContain(
        'Do not list, summarize, or repeat options or prices from your peers’ domains',
      );
    });

    it('silences generic duplicate recap theses on summary questions while keeping distinct domain conflict theses', () => {
      const { decisions, speakers } = applyAsymmetricDedup(
        {
          maya: {
            effort: 'standard',
            thesis: 'Summarize the plan so far and what we need to decide for flights and hotels',
          },
          liam: {
            effort: 'standard',
            // Generic duplicate recap thesis without a specific new constraint -> silenced!
            thesis: 'Summarize the current travel plan so far and pending decisions',
          },
          zoe: {
            effort: 'light',
            // Specific domain safety constraint -> kept!
            thesis: 'Warn that Nishiki Market shared grills conflict with Sam shellfish allergy',
          },
        },
        ['maya', 'liam', 'zoe'],
      );

      expect(decisions.liam.effort).toBe('silent');
      expect(decisions.zoe.effort).toBe('light');
      expect(speakers.map((s) => s.agentId)).toEqual(['maya', 'zoe']);
    });
  });

  describe('2. Effort Calibration (Stretch)', () => {
    it('maps light, standard, and deep effort levels to calibrated maxTokens and prompt instructions', async () => {
      const orchestrator = new GroupChatOrchestrator({ store, llm: mockLLM });

      mockLLM.enqueueJSON({
        response: {
          maya: { effort: 'light', thesis: 'State total budget cap' },
          liam: { effort: 'standard', thesis: 'Explain Nishiki Market hours and food' },
          zoe: { effort: 'deep', thesis: 'Detail all traveler dietary and pacing rules' },
        },
      });
      mockLLM.enqueueStream({
        match: /You are Maya/,
        chunks: ['Our total group budget is $4,800 ($1,200 per person).'],
      });
      mockLLM.enqueueStream({
        match: /You are Liam/,
        chunks: ['Nishiki Market runs 11 AM to 2 PM with grilled seafood skewers.'],
      });
      mockLLM.enqueueStream({
        match: /You are Zoe/,
        chunks: ['Sam has a severe shellfish allergy and motion sickness, while Priya is vegetarian.'],
      });
      mockLLM.enqueueJSON({
        response: {
          isResolved: true,
          maya: { effort: 'silent', thesis: '' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      await orchestrator.handleUserMessage('Give me a quick overview of our constraints.');

      expect(mockLLM.streamCalls).toHaveLength(3);

      const [mayaCall, liamCall, zoeCall] = mockLLM.streamCalls;
      expect(mayaCall.maxTokens).toBe(EFFORT_MAX_TOKENS.light); // 80
      expect(mayaCall.messages[0].content).toContain(
        'Keep your response to 1 short, crisp sentence',
      );

      expect(liamCall.maxTokens).toBe(EFFORT_MAX_TOKENS.standard); // 250
      expect(liamCall.messages[0].content).toContain('2 to 4 clear sentences');

      expect(zoeCall.maxTokens).toBe(EFFORT_MAX_TOKENS.deep); // 600
      expect(zoeCall.messages[0].content).toContain('thorough, well-structured breakdown');
    });
  });

  describe('3. Concurrent Streaming & Deterministic seq Ordering Across Reload', () => {
    it('reserves seq slots up front, interleaves concurrent deltas, and preserves identical seq order on reload', async () => {
      const orchestrator = new GroupChatOrchestrator({ store, llm: mockLLM });
      const sseEvents: OrchestratorSSEEvent[] = [];
      orchestrator.onEvent((ev) => sseEvents.push(ev));

      mockLLM.enqueueJSON({
        response: {
          maya: { effort: 'standard', thesis: 'Compare Flight A and B' },
          liam: { effort: 'standard', thesis: 'Share KIX transit time' },
          zoe: { effort: 'standard', thesis: 'Note 7:30 AM wakeup rule' },
        },
      });

      // Maya is slower (25ms/chunk), Liam is medium (10ms/chunk), Zoe is fastest (2ms/chunk)
      // So Zoe finishes first, Liam second, Maya last — yet seq MUST remain Maya=2, Liam=3, Zoe=4!
      mockLLM.enqueueStream({
        match: /You are Maya/,
        chunks: ['Maya-1 ', 'Maya-2 ', 'Maya-3'],
        delayMs: 25,
      });
      mockLLM.enqueueStream({
        match: /You are Liam/,
        chunks: ['Liam-1 ', 'Liam-2 ', 'Liam-3'],
        delayMs: 10,
      });
      mockLLM.enqueueStream({
        match: /You are Zoe/,
        chunks: ['Zoe-1 ', 'Zoe-2 ', 'Zoe-3'],
        delayMs: 2,
      });

      mockLLM.enqueueJSON({
        response: {
          isResolved: true,
          maya: { effort: 'silent', thesis: '' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      await orchestrator.handleUserMessage('Tell us about Day 1 arrival.');

      const createdEvents = sseEvents.filter(
        (e): e is Extract<OrchestratorSSEEvent, { type: 'message.created' }> =>
          e.type === 'message.created',
      );
      expect(createdEvents.map((e) => ({ seq: e.message.seq, agentId: e.message.agentId }))).toEqual([
        { seq: 1, agentId: null },
        { seq: 2, agentId: 'maya' },
        { seq: 3, agentId: 'liam' },
        { seq: 4, agentId: 'zoe' },
      ]);

      // Verify deltas actually interleaved (Zoe's or Liam's deltas arrived before Maya's final delta)
      const deltaAgentOrder = sseEvents
        .filter(
          (e): e is Extract<OrchestratorSSEEvent, { type: 'message.delta' }> =>
            e.type === 'message.delta',
        )
        .map((e) => e.agentId);
      const firstZoeDeltaIdx = deltaAgentOrder.indexOf('zoe');
      const lastMayaDeltaIdx = deltaAgentOrder.lastIndexOf('maya');
      expect(firstZoeDeltaIdx).toBeLessThan(lastMayaDeltaIdx);

      // Verify done events arrived in completion order (zoe first, maya last)
      const doneAgentOrder = sseEvents
        .filter(
          (e): e is Extract<OrchestratorSSEEvent, { type: 'message.done' }> =>
            e.type === 'message.done',
        )
        .map((e) => e.message.agentId);
      expect(doneAgentOrder[0]).toBe('zoe');
      expect(doneAgentOrder[2]).toBe('maya');

      // Simulated page reload: getPublicMessages() / store.listMessages() MUST match created order (seq 1, 2, 3, 4)
      const reloadedMessages = orchestrator.getPublicMessages();
      expect(reloadedMessages.map((m) => ({ seq: m.seq, agentId: m.agentId, status: m.status }))).toEqual([
        { seq: 1, agentId: null, status: 'completed' },
        { seq: 2, agentId: 'maya', status: 'completed' },
        { seq: 3, agentId: 'liam', status: 'completed' },
        { seq: 4, agentId: 'zoe', status: 'completed' },
      ]);
      expect(reloadedMessages[1].content).toBe('Maya-1 Maya-2 Maya-3');
      expect(reloadedMessages[2].content).toBe('Liam-1 Liam-2 Liam-3');
      expect(reloadedMessages[3].content).toBe('Zoe-1 Zoe-2 Zoe-3');

      // Verify zero leaked internal fields on public wire messages
      for (const msg of reloadedMessages) {
        expect(Object.keys(msg).sort()).toEqual(
          ['agentId', 'agentName', 'content', 'createdAt', 'id', 'role', 'seq', 'status'].sort(),
        );
      }
    });
  });

  describe('4. Cross-Agent Synthesis & Bounded A2A Follow-Up Waves', () => {
    it('runs multi-wave A2A synthesis across private vaults and terminates naturally when isResolved is true', async () => {
      const orchestrator = new GroupChatOrchestrator({ store, llm: mockLLM });

      // Wave 0: Maya and Liam speak; Maya asks Liam about transit time for Flight B (3:45 PM) + Ryokan Gion (6:00 PM cutoff)
      mockLLM.enqueueJSON({
        response: {
          maya: {
            effort: 'standard',
            thesis: 'State Flight B lands at 3:45 PM ($820) and Ryokan Gion has a 6:00 PM cutoff; ask @Liam about transit',
          },
          liam: {
            effort: 'standard',
            thesis: 'Explain KIX to Gion takes 2 hours 15 minutes, arriving at 6:00 PM sharp with zero buffer',
          },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      mockLLM.enqueueStream({
        match: (opts) =>
          opts.messages[0].content.includes('You are Maya') &&
          opts.messages.length === 2,
        chunks: [
          'Flight B ($820) lands at 3:45 PM and Ryokan Gion ($380/night) has a strict 6:00 PM check-in cutoff—@Liam how long does airport transit take?',
        ],
      });

      mockLLM.enqueueStream({
        match: (opts) =>
          opts.messages[0].content.includes('You are Liam') &&
          opts.messages.length === 2,
        chunks: [
          'Clearing KIX customs and taking the Haruka Express plus taxi to Gion takes 2 hours 15 minutes, putting us there at 6:00 PM with zero buffer.',
        ],
      });

      // Wave 1 (A2A follow-up after Wave 0 barrier):
      // Maya pivots to Hotel Granvia ($290/night, 24h check-in) or Flight A (6:45 AM), and Zoe vetoes 6:45 AM Flight A due to the 7:30 AM group sleep rule!
      mockLLM.enqueueJSON({
        response: {
          isResolved: false,
          maya: {
            effort: 'light',
            thesis: 'Recommend Hotel Granvia ($290/night, 24h check-in) since 2h 15m cuts Ryokan Gion too close',
          },
          liam: { effort: 'silent', thesis: '' },
          zoe: {
            effort: 'standard',
            thesis: 'Veto 6:45 AM Flight A because the group rule forbids waking before 7:30 AM, confirming Flight B + Hotel Granvia',
          },
        },
      });

      mockLLM.enqueueStream({
        match: (opts) =>
          opts.messages[0].content.includes('You are Maya') &&
          opts.messages.length > 2,
        chunks: [
          'Since 2h 15m cuts the 6:00 PM Ryokan Gion cutoff too close, let us book Hotel Granvia ($290/night, 24-hour check-in) on Night 1.',
        ],
      });

      mockLLM.enqueueStream({
        match: (opts) => opts.messages[0].content.includes('You are Zoe'),
        chunks: [
          'Agreed on Flight B + Hotel Granvia—our group has a strict no-waking-before-7:30-AM rule, so the 6:45 AM Flight A is off the table.',
        ],
      });

      // Wave 2 (A2A check): Exchange is resolved!
      mockLLM.enqueueJSON({
        response: {
          isResolved: true,
          maya: { effort: 'silent', thesis: '' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      const turnResult = await orchestrator.handleUserMessage(
        'Should we take Flight B and stay at Ryokan Gion on Day 1?',
      );

      expect(turnResult.wavePlans).toHaveLength(3); // Wave 0, Wave 1, and Wave 2 (resolved check)
      expect(turnResult.agentMessages).toHaveLength(4);
      expect(turnResult.agentMessages.map((m) => ({ seq: m.seq, agentId: m.agentId }))).toEqual([
        { seq: 2, agentId: 'maya' },
        { seq: 3, agentId: 'liam' },
        { seq: 4, agentId: 'maya' },
        { seq: 5, agentId: 'zoe' },
      ]);

      // Verify Wave 1 agents saw Wave 0 messages in their conversation history
      const wave1ZoeCall = mockLLM.streamCalls[3];
      const historyText = wave1ZoeCall.messages.map((m) => m.content).join('\n');
      expect(historyText).toContain('[Maya]: Flight B ($820) lands at 3:45 PM');
      expect(historyText).toContain('[Liam]: Clearing KIX customs');
    });

    it('enforces the MAX_A2A_WAVES = 3 safety ceiling and marks the final wave with isFinalWave', async () => {
      const orchestrator = new GroupChatOrchestrator({ store, llm: mockLLM });

      // Queue 4 waves (Wave 0 + 3 follow-up waves) where Maya and Liam alternate back-and-forth
      // with distinct cross-agent points while isResolved stays false
      const distinctWaveTheses = [
        { maya: 'Propose Flight B ($820) landing at 3:45 PM', liam: '' },
        { maya: '', liam: 'Note KIX customs and transit to Gion takes 2 hours 15 minutes' },
        { maya: 'Pivot lodging to Hotel Granvia ($290/night, 24h check-in)', liam: '' },
        { maya: '', liam: 'Confirm Haruka Express arrives directly inside Kyoto Station' },
      ];

      for (let w = 0; w <= MAX_A2A_WAVES; w++) {
        const step = distinctWaveTheses[w];
        mockLLM.enqueueJSON({
          response: {
            isResolved: false,
            maya: step.maya
              ? { effort: 'light', thesis: step.maya }
              : { effort: 'silent', thesis: '' },
            liam: step.liam
              ? { effort: 'light', thesis: step.liam }
              : { effort: 'silent', thesis: '' },
            zoe: { effort: 'silent', thesis: '' },
          },
        });
        mockLLM.enqueueStream({
          chunks: [`Agent response in wave ${w}: ${step.maya || step.liam}.`],
          delayMs: 1,
        });
      }

      const res = await orchestrator.handleUserMessage('Keep debating.');

      // Wave 0 + 3 follow-up waves = 4 waves total, then stops at safety ceiling
      expect(res.wavePlans).toHaveLength(MAX_A2A_WAVES + 1);
      expect(res.agentMessages).toHaveLength(MAX_A2A_WAVES + 1);

      // The final wave (Wave 3) must include the Final Synthesis Wave wrap-up instruction
      const finalWaveSystemPrompt =
        mockLLM.streamCalls[MAX_A2A_WAVES].messages[0].content;
      expect(finalWaveSystemPrompt).toContain('## Final Synthesis Wave');
      expect(finalWaveSystemPrompt).toContain('do not ask further open questions');
    });

    it('prevents consecutive self-replies and self-repetition when an agent was the sole speaker in the previous wave', async () => {
      const orchestrator = new GroupChatOrchestrator({ store, llm: mockLLM });

      // Wave 0: Only Maya speaks and asks the user which flight they prefer
      mockLLM.enqueueJSON({
        response: {
          maya: {
            effort: 'standard',
            thesis: 'Present Flight A (6:45 AM, $650) and Flight B (11:30 AM, $820) options',
          },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      mockLLM.enqueueStream({
        match: /You are Maya/,
        chunks: [
          'We have two flight options into KIX: Flight A (6:45 AM, $650) and Flight B (11:30 AM, $820). Please let me know which flight you prefer so we can finalize the travel plans!',
        ],
      });

      // Wave 1: Even if the LLM router mistakenly tries to make Maya speak again to "confirm preferred flight"
      // while Zoe chimes in about the 7:30 AM rule, the router MUST silence Maya's consecutive self-reply!
      mockLLM.enqueueJSON({
        response: {
          isResolved: false,
          maya: {
            effort: 'light',
            thesis: 'We need to confirm the preferred flight option to finalize travel plans.',
          },
          liam: { effort: 'silent', thesis: '' },
          zoe: {
            effort: 'standard',
            thesis: 'Veto 6:45 AM Flight A due to the group rule against waking before 7:30 AM',
          },
        },
      });

      mockLLM.enqueueStream({
        match: /You are Zoe/,
        chunks: [
          'We need to avoid Flight A (6:45 AM) since our group rule forbids waking before 7:30 AM—let us go with Flight B!',
        ],
      });

      // Wave 2: Even if the router tries to have Zoe repeat herself or Maya repeat her flight prompt, both are silenced and the turn resolves!
      mockLLM.enqueueJSON({
        response: {
          isResolved: false,
          maya: {
            effort: 'light',
            thesis: 'Confirm the preferred flight option between Flight A and Flight B',
          },
          liam: { effort: 'silent', thesis: '' },
          zoe: {
            effort: 'light',
            thesis: 'Avoid Flight A because of the 7:30 AM wakeup rule',
          },
        },
      });

      const res = await orchestrator.handleUserMessage(
        "What's our travel plans so far, when are our flights?",
      );

      // Maya spoke ONCE in Wave 0, Zoe spoke ONCE in Wave 1, and Wave 2 auto-resolved with 0 speakers
      expect(res.agentMessages).toHaveLength(2);
      expect(res.agentMessages.map((m) => m.agentId)).toEqual(['maya', 'zoe']);
    });

    it('guarantees router prompts contain zero private vault facts so Maya never learns Zoe 7:30 AM rule before Zoe speaks', async () => {
      const orchestrator = new GroupChatOrchestrator({ store, llm: mockLLM });

      mockLLM.enqueueJSON({
        response: {
          maya: {
            effort: 'standard',
            thesis: 'Present the flight options, departure times, and price trade-offs from your vault',
          },
          liam: { effort: 'silent', thesis: '' },
          zoe: {
            effort: 'light',
            thesis: 'Evaluate how the flight departure times fit the group morning wake-up rules',
          },
        },
      });

      mockLLM.enqueueStream({
        match: /You are Maya/,
        chunks: [
          'Flight A (ANA NH-105) departs at 6:45 AM for $650/person, while Flight B (JAL JL-701) departs at 11:30 AM for $820/person.',
        ],
      });

      mockLLM.enqueueStream({
        match: /You are Zoe/,
        chunks: [
          'Since our group has a strict rule against waking up before 7:30 AM, 6:45 AM Flight A is out—Flight B at 11:30 AM is the best choice!',
        ],
      });

      mockLLM.enqueueJSON({
        response: {
          isResolved: true,
          maya: { effort: 'silent', thesis: '' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      await orchestrator.handleUserMessage('what is the best flight option?');

      // Inspect the router prompt sent to completeJSON: it must NOT contain any private vault facts!
      const routerSystemPrompt = mockLLM.jsonCalls[0].messages[0].content;
      const forbiddenVaultLiteralsInRouter = [
        '7:30',
        '6:45',
        '$650',
        '$820',
        '$4,800',
        '6:00 PM',
        '2h 15m',
        'shellfish',
        'Shorai-an',
        'Granvia',
      ];
      for (const secretFact of forbiddenVaultLiteralsInRouter) {
        expect(routerSystemPrompt).not.toContain(secretFact);
      }

      // Inspect Maya's system prompt: it must NOT contain "7:30"
      const mayaSystemPrompt = mockLLM.streamCalls[0].messages[0].content;
      expect(mayaSystemPrompt).not.toContain('7:30');
    });
  });

  describe('5. Live OpenAI Multi-Agent Turn & A2A Synthesis Check', () => {
    it(
      'routes a live user turn with gpt-4o-mini, streams concurrent agents, and synthesizes private vault facts',
      async () => {
        const liveLLM = createLLMProvider();
        if (!liveLLM.config.apiKey) {
          return;
        }

        const liveRouter = new TurnRouter(liveLLM);
        // Verify silence gate on a single-domain question first
        const singleDomainUserRow = store.reserveMessage({
          role: 'user',
          content: '@Maya what is the exact total group budget in USD?',
        });
        const singlePlan = await liveRouter.routeWave({
          waveIndex: 0,
          transcript: [singleDomainUserRow],
          latestUserMessage: singleDomainUserRow.content,
        });
        expect(singlePlan.decisions.maya.effort).not.toBe('silent');
        expect(singlePlan.decisions.liam.effort).toBe('silent');
        expect(singlePlan.decisions.zoe.effort).toBe('silent');

        store.reset();

        // Now run a full multi-agent cross-domain synthesis turn through GroupChatOrchestrator
        const orchestrator = new GroupChatOrchestrator({
          store,
          llm: liveLLM,
          maxA2AWaves: 2,
        });

        const sseEvents: OrchestratorSSEEvent[] = [];
        orchestrator.onEvent((e) => sseEvents.push(e));

        const turnRes = await orchestrator.handleUserMessage(
          'We want to take Flight B (landing 3:45 PM) and check into Ryokan Gion on Day 1, then do the Nishiki Market Street Food Tour on Day 2. Can all three of you check if our timing, budget, and group allergies allow this?',
        );

        expect(turnRes.agentMessages.length).toBeGreaterThanOrEqual(2);
        const combinedText = turnRes.agentMessages
          .map((m) => `[${m.agentName}]: ${m.content}`)
          .join('\n');

        // Verify facts from private vaults were synthesized across the agents
        // Maya: $820 / $380 / 6:00 PM / Granvia
        expect(combinedText).toMatch(/820|380|6:00\s*PM|Granvia/i);
        // Liam: 2 hours 15 minutes / 2h 15m / Nishiki / shellfish
        expect(combinedText).toMatch(/2\s*hours?\s*15|2h\s*15m|Nishiki|11:00\s*AM|shellfish/i);
        // Zoe: Sam / shellfish allergy / Priya
        expect(combinedText).toMatch(/Sam|allergy|shellfish|Priya/i);

        // Verify reload consistency
        const reloaded = orchestrator.getPublicMessages();
        const seqs = reloaded.map((m) => m.seq);
        expect(seqs).toEqual([...seqs].sort((a, b) => a - b));
        expect(sseEvents.at(-1)?.type).toBe('turn.idle');

        // Regression check for "what is the best flight option?" on a fresh chat:
        // Maya's Wave 0 response MUST NOT know Zoe's private "7:30 AM" wake-up rule before Zoe speaks!
        store.reset();
        const bestFlightRes = await orchestrator.handleUserMessage(
          'what is the best flight option?',
        );
        const bestFlightMaya = bestFlightRes.agentMessages.find((m) => m.agentId === 'maya');
        const bestFlightZoe = bestFlightRes.agentMessages.find((m) => m.agentId === 'zoe');
        expect(bestFlightMaya).toBeDefined();
        expect(bestFlightZoe).toBeDefined();
        // Maya's first message cannot mention 7:30 AM (Zoe's exclusive vault fact)
        expect(bestFlightMaya!.content).not.toMatch(/7:30/);
        expect(bestFlightMaya!.content).toMatch(/650|820|Flight\s*A|Flight\s*B/i);
        // Zoe provides the 7:30 AM wake-up rule
        expect(bestFlightZoe!.content).toMatch(/7:30|wake|early/i);

        // Regression check for user-reported prompt: "What's our travel plans so far, when are our flights?"
        // Verify Maya NEVER speaks twice in a row or repeats herself across waves!
        store.reset();
        const flightCheckRes = await orchestrator.handleUserMessage(
          "What's our travel plans so far, when are our flights?",
        );
        const speakerSequence = flightCheckRes.agentMessages.map((m) => m.agentId);
        expect(speakerSequence).toContain('maya');
        for (let i = 1; i < speakerSequence.length; i++) {
          expect(speakerSequence[i]).not.toBe(speakerSequence[i - 1]);
        }
        const mayaCount = speakerSequence.filter((id) => id === 'maya').length;
        expect(mayaCount).toBe(1);

        // Regression check for multi-turn conversation followed by:
        // "what's our plan so far and what do we need to decide"
        // Seed prior agreed context (Flight B + Hotel Granvia + Day 2 Nishiki/Shorai-an)
        store.reserveMessage({
          role: 'user',
          content: 'Let us go with Flight B and Hotel Granvia, plus Nishiki Market lunch and Shorai-an dinner on Day 2.',
        });
        store.reserveMessage({
          role: 'agent',
          agentId: 'maya',
          content: 'Sounds great—Flight B ($820/person, lands 3:45 PM) and Hotel Granvia ($290/night, 24h check-in) are noted.',
          status: 'completed',
        });

        const recapRes = await orchestrator.handleUserMessage(
          "what's our plan so far and what do we need to decide",
        );
        expect(recapRes.agentMessages.length).toBeGreaterThanOrEqual(1);

        // Verify strict domain lanes: Liam and Zoe must NEVER re-list Flight B or Hotel Granvia when Maya covers flights/lodging
        const mayaRecap = recapRes.agentMessages.find((m) => m.agentId === 'maya');
        const liamRecap = recapRes.agentMessages.find((m) => m.agentId === 'liam');
        const zoeRecap = recapRes.agentMessages.find((m) => m.agentId === 'zoe');

        if (mayaRecap && liamRecap) {
          expect(liamRecap.content).not.toMatch(/Flight\s*B|JL-701|Hotel\s*Granvia/i);
        }
        if (mayaRecap && zoeRecap) {
          expect(zoeRecap.content).not.toMatch(/Hotel\s*Granvia/i);
        }
      },
      75000,
    );
  });
});
