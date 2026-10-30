import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { EmanateStore } from '../src/server/db.js';
import {
  AGENT_IDS,
  AGENT_PROFILES,
  buildAgentSystemPrompt,
  loadAgentPrivateContext,
} from '../src/server/agents.js';
import {
  createLLMProvider,
  HttpLLMProvider,
  resolveLLMConfigFromEnv,
  ScriptableMockLLMProvider,
} from '../src/server/llm.js';

describe('Part 1 Foundation: SQLite Store, Isolated Contexts & Swappable LLM Adapter', () => {
  let store: EmanateStore;

  beforeEach(() => {
    store = new EmanateStore(':memory:');
  });

  afterEach(() => {
    store.close();
  });

  describe('1. SQLite Messages Store — Monotonic seq & In-Place Streaming Updates', () => {
    it('reserves message slots with strictly increasing integer seq and updates deltas in place', () => {
      const userMsg = store.reserveMessage({
        id: 'msg-user-1',
        role: 'user',
        content: 'Should we take Flight B and stay at Ryokan Gion?',
      });
      const mayaMsg = store.reserveMessage({
        id: 'msg-maya-1',
        role: 'agent',
        agentId: 'maya',
      });
      const liamMsg = store.reserveMessage({
        id: 'msg-liam-1',
        role: 'agent',
        agentId: 'liam',
      });

      expect(userMsg.seq).toBe(1);
      expect(userMsg.status).toBe('completed');
      expect(mayaMsg.seq).toBe(2);
      expect(mayaMsg.status).toBe('streaming');
      expect(liamMsg.seq).toBe(3);
      expect(liamMsg.status).toBe('streaming');

      // Interleave deltas out of order (Liam first, then Maya, then Liam)
      store.appendMessageDelta('msg-liam-1', 'KIX to Gion takes ');
      store.appendMessageDelta('msg-maya-1', 'Flight B is $820 ');
      store.appendMessageDelta('msg-liam-1', '2 hours 15 minutes.');
      store.appendMessageDelta('msg-maya-1', 'and lands at 3:45 PM.');

      store.markMessageStatus('msg-liam-1', 'completed');
      store.markMessageStatus('msg-maya-1', 'completed');

      const allMessages = store.listMessages();
      expect(allMessages).toHaveLength(3);
      expect(allMessages.map((m) => m.seq)).toEqual([1, 2, 3]);
      expect(allMessages.map((m) => m.id)).toEqual(['msg-user-1', 'msg-maya-1', 'msg-liam-1']);

      expect(allMessages[1].content).toBe('Flight B is $820 and lands at 3:45 PM.');
      expect(allMessages[1].status).toBe('completed');
      expect(allMessages[2].content).toBe('KIX to Gion takes 2 hours 15 minutes.');
      expect(allMessages[2].status).toBe('completed');
    });

    it('supports full in-place content replacement while preserving seq and id', () => {
      const msg = store.reserveMessage({
        id: 'msg-maya-steer',
        role: 'agent',
        agentId: 'maya',
        content: 'Initial partial text',
      });
      expect(msg.seq).toBe(1);

      const updated = store.updateMessageContent(
        'msg-maya-steer',
        'Initial partial text — pivoted to JPY: ¥97,500.',
      );
      expect(updated.seq).toBe(1);
      expect(updated.id).toBe('msg-maya-steer');
      expect(updated.content).toBe('Initial partial text — pivoted to JPY: ¥97,500.');
    });
  });

  describe('2. Standing Rules — Scoping, Persistence & Conflict Superseding', () => {
    it('persists global and agent-scoped rules and filters accurately per agent', () => {
      store.addStandingRule({
        id: 'rule-global-1',
        scope: 'global',
        rule: 'Always format responses as concise bullet points.',
      });
      store.addStandingRule({
        id: 'rule-maya-1',
        scope: 'maya',
        rule: 'State all prices in Japanese Yen (JPY).',
      });
      store.addStandingRule({
        id: 'rule-liam-1',
        scope: 'liam',
        rule: 'Include walking durations in minutes.',
      });

      const mayaRules = store.listActiveStandingRules('maya');
      expect(mayaRules.map((r) => r.id)).toEqual(['rule-global-1', 'rule-maya-1']);

      const liamRules = store.listActiveStandingRules('liam');
      expect(liamRules.map((r) => r.id)).toEqual(['rule-global-1', 'rule-liam-1']);

      const zoeRules = store.listActiveStandingRules('zoe');
      expect(zoeRules.map((r) => r.id)).toEqual(['rule-global-1']);
    });

    it('supersedes conflicting rules in the same scope automatically and via explicit ID', () => {
      const firstCurrencyRule = store.addStandingRule({
        id: 'rule-maya-usd',
        scope: 'maya',
        rule: 'Always state all prices in USD dollars.',
      });
      expect(firstCurrencyRule.active).toBe(1);

      // Adding a conflicting currency rule in the 'maya' scope should automatically deactivate the old one
      const secondCurrencyRule = store.addStandingRule({
        id: 'rule-maya-jpy',
        scope: 'maya',
        rule: 'State all prices in Japanese Yen (JPY) instead.',
      });
      expect(secondCurrencyRule.active).toBe(1);

      const activeMayaRules = store.listActiveStandingRules('maya');
      expect(activeMayaRules).toHaveLength(1);
      expect(activeMayaRules[0].id).toBe('rule-maya-jpy');

      // Explicit supersede by ID also works for arbitrary rules
      store.supersedeStandingRule('rule-maya-jpy', {
        id: 'rule-maya-eur',
        scope: 'maya',
        rule: 'Quote totals in Euros.',
      });

      const afterExplicitSupersede = store.listActiveStandingRules('maya');
      expect(afterExplicitSupersede).toHaveLength(1);
      expect(afterExplicitSupersede[0].id).toBe('rule-maya-eur');

      const allRulesHistory = store.listAllStandingRules();
      expect(allRulesHistory).toHaveLength(3);
      expect(allRulesHistory.filter((r) => r.active === 1)).toHaveLength(1);
    });
  });

  describe('3. Strict Private Context Isolation (Maya, Liam, Zoe)', () => {
    it('defines all 3 agents with distinct roles and non-empty private markdown vaults', () => {
      expect(AGENT_IDS).toEqual(['maya', 'liam', 'zoe']);
      for (const id of AGENT_IDS) {
        expect(AGENT_PROFILES[id]).toBeDefined();
        const rawVault = loadAgentPrivateContext(id);
        expect(rawVault.length).toBeGreaterThan(200);
      }
    });

    it('guarantees strict vault & rule isolation in buildAgentSystemPrompt', () => {
      store.addStandingRule({
        id: 'r-global',
        scope: 'global',
        rule: 'Use bullet points.',
      });
      store.addStandingRule({
        id: 'r-maya',
        scope: 'maya',
        rule: 'Convert prices to JPY.',
      });
      store.addStandingRule({
        id: 'r-liam',
        scope: 'liam',
        rule: 'Highlight train line names.',
      });
      store.addStandingRule({
        id: 'r-zoe',
        scope: 'zoe',
        rule: 'Flag allergy risks first.',
      });

      const allActiveRules = store.listActiveStandingRules();

      const mayaPrompt = buildAgentSystemPrompt('maya', allActiveRules, {
        effort: 'standard',
        thesis: 'Compare Flight A and Flight B pricing and hotel cutoffs',
      });
      const liamPrompt = buildAgentSystemPrompt('liam', allActiveRules, {
        effort: 'deep',
        thesis: 'Explain KIX to Gion transit time and Nishiki Market hours',
      });
      const zoePrompt = buildAgentSystemPrompt('zoe', allActiveRules, {
        effort: 'light',
        thesis: 'Warn about Sam shellfish allergy and 7:30 AM wakeup rule',
      });

      // Maya's exclusive facts are in Maya's prompt ONLY
      const mayaExclusiveFacts = ['$4,800', '$650', '$820', 'Ryokan Gion', '$380', 'Hotel Granvia', '$290'];
      for (const fact of mayaExclusiveFacts) {
        expect(mayaPrompt).toContain(fact);
        expect(liamPrompt).not.toContain(fact);
        expect(zoePrompt).not.toContain(fact);
      }

      // Liam's exclusive facts are in Liam's prompt ONLY
      const liamExclusiveFacts = [
        '2 hours 15 minutes',
        'Nishiki Market Street Food Tour',
        'Arashiyama Bamboo Grove Tofu Kaiseki',
        '7:30 PM',
      ];
      for (const fact of liamExclusiveFacts) {
        expect(liamPrompt).toContain(fact);
        expect(mayaPrompt).not.toContain(fact);
        expect(zoePrompt).not.toContain(fact);
      }

      // Zoe's exclusive facts are in Zoe's prompt ONLY
      const zoeExclusiveFacts = [
        'Anaphylactic',
        'EpiPen',
        'Strict Vegetarian',
        'no one wakes up before 7:30 AM',
      ];
      for (const fact of zoeExclusiveFacts) {
        expect(zoePrompt).toContain(fact);
        expect(mayaPrompt).not.toContain(fact);
        expect(liamPrompt).not.toContain(fact);
      }

      // Scoped rules are isolated to their target agent, while global rule is in all 3
      expect(mayaPrompt).toContain('Use bullet points.');
      expect(mayaPrompt).toContain('Convert prices to JPY.');
      expect(mayaPrompt).not.toContain('Highlight train line names.');
      expect(mayaPrompt).not.toContain('Flag allergy risks first.');

      expect(liamPrompt).toContain('Use bullet points.');
      expect(liamPrompt).toContain('Highlight train line names.');
      expect(liamPrompt).not.toContain('Convert prices to JPY.');
      expect(liamPrompt).not.toContain('Flag allergy risks first.');

      expect(zoePrompt).toContain('Use bullet points.');
      expect(zoePrompt).toContain('Flag allergy risks first.');
      expect(zoePrompt).not.toContain('Convert prices to JPY.');
      expect(zoePrompt).not.toContain('Highlight train line names.');
    });
  });

  describe('4. Swappable LLM Adapter — Env Config, Streaming & AbortSignal', () => {
    it('swaps providers, models, and custom base URLs cleanly via environment variables', () => {
      const openaiCfg = resolveLLMConfigFromEnv({
        LLM_PROVIDER: 'openai',
        LLM_MODEL: 'gpt-4o-mini',
        LLM_API_KEY: 'sk-test-openai',
        LLM_BASE_URL: '',
      });
      expect(openaiCfg).toEqual({
        provider: 'openai',
        model: 'gpt-4o-mini',
        apiKey: 'sk-test-openai',
        baseUrl: undefined,
      });

      const anthropicProvider = createLLMProvider({
        LLM_PROVIDER: 'anthropic',
        LLM_MODEL: 'claude-3-5-haiku-latest',
        LLM_API_KEY: 'sk-ant-test',
      }) as HttpLLMProvider;
      expect(anthropicProvider.config.provider).toBe('anthropic');
      expect(anthropicProvider.getEndpointUrl(true)).toBe('https://api.anthropic.com/v1/messages');

      const geminiProvider = createLLMProvider({
        LLM_PROVIDER: 'gemini',
        LLM_MODEL: 'gemini-2.5-flash',
        LLM_API_KEY: 'ai-za-test',
      }) as HttpLLMProvider;
      expect(geminiProvider.config.provider).toBe('gemini');
      expect(geminiProvider.getEndpointUrl(true)).toContain(
        'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent',
      );

      const compatProvider = createLLMProvider({
        LLM_PROVIDER: 'openai-compatible',
        LLM_MODEL: 'deepseek-chat',
        LLM_API_KEY: 'sk-compat',
        LLM_BASE_URL: 'https://openrouter.ai/api/v1/',
      }) as HttpLLMProvider;
      expect(compatProvider.config.baseUrl).toBe('https://openrouter.ai/api/v1');
      expect(compatProvider.getEndpointUrl(false)).toBe(
        'https://openrouter.ai/api/v1/chat/completions',
      );
    });

    it('emits streaming tokens and structured JSON responses via ScriptableMockLLMProvider', async () => {
      const mockLLM = new ScriptableMockLLMProvider();
      mockLLM.enqueueStream({
        chunks: ['Flight B ', 'costs ', '$820.'],
        delayMs: 2,
      });
      mockLLM.enqueueJSON({
        response: { effort: 'standard', thesis: 'Compare flight prices' },
      });

      const emitted: string[] = [];
      for await (const token of mockLLM.streamChat({
        messages: [{ role: 'user', content: 'How much is Flight B?' }],
      })) {
        emitted.push(token);
      }
      expect(emitted).toEqual(['Flight B ', 'costs ', '$820.']);

      const jsonResult = await mockLLM.completeJSON<{ effort: string; thesis: string }>({
        messages: [{ role: 'user', content: 'Route turn' }],
      });
      expect(jsonResult).toEqual({ effort: 'standard', thesis: 'Compare flight prices' });
    });

    it('supports mid-stream cancellation via AbortSignal', async () => {
      const mockLLM = new ScriptableMockLLMProvider();
      mockLLM.enqueueStream({
        chunks: ['Chunk 1 ', 'Chunk 2 ', 'Chunk 3 ', 'Chunk 4'],
        delayMs: 15,
      });

      const abortController = new AbortController();
      const received: string[] = [];

      await expect(async () => {
        for await (const chunk of mockLLM.streamChat({
          messages: [{ role: 'user', content: 'Start long stream' }],
          signal: abortController.signal,
        })) {
          received.push(chunk);
          if (received.length === 2) {
            abortController.abort();
          }
        }
      }).rejects.toThrow();

      expect(received).toEqual(['Chunk 1 ', 'Chunk 2 ']);
    });

    it('streams SSE tokens and parses JSON over HTTP fetch for OpenAI-compatible endpoints', async () => {
      const ssePayload = [
        'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"Kyoto!"}}]}\n\n',
        'data: [DONE]\n\n',
      ];

      const fakeFetch: typeof fetch = async (_input, init) => {
        const bodyObj = JSON.parse(String(init?.body ?? '{}'));
        if (bodyObj.stream) {
          const stream = new ReadableStream<Uint8Array>({
            start(controller) {
              const textEncoder = new TextEncoder();
              for (const frame of ssePayload) {
                controller.enqueue(textEncoder.encode(frame));
              }
              controller.close();
            },
          });
          return new Response(stream, { status: 200 });
        }

        return new Response(
          JSON.stringify({
            choices: [{ message: { content: '```json\n{"isResolved": true}\n```' } }],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      };

      const provider = createLLMProvider(
        {
          LLM_PROVIDER: 'openai',
          LLM_MODEL: 'gpt-4o-mini',
          LLM_API_KEY: 'sk-unit-test',
        },
        fakeFetch,
      );

      const tokens: string[] = [];
      for await (const tok of provider.streamChat({
        messages: [{ role: 'user', content: 'Hi' }],
      })) {
        tokens.push(tok);
      }
      expect(tokens.join('')).toBe('Hello Kyoto!');

      const parsed = await provider.completeJSON<{ isResolved: boolean }>({
        messages: [{ role: 'user', content: 'Check resolution' }],
      });
      expect(parsed).toEqual({ isResolved: true });
    });
  });
});
