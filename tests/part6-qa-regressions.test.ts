import { describe, it, expect } from 'vitest';
import type { AgentId, MessageRow } from '../src/server/db.js';
import type { CompleteJSONOptions, LLMConfig, LLMProvider, StreamChatOptions } from '../src/server/llm.js';
import { TurnRouter } from '../src/server/router.js';
import { getScenario } from '../src/server/scenarios.js';

/** Not a ScriptableMockLLMProvider, so the router runs its live-model code paths. */
class FixedJSONProvider implements LLMProvider {
  readonly config: LLMConfig = { provider: 'openai', model: 'stub', apiKey: 'stub' };
  constructor(private readonly response: unknown) {}
  async *streamChat(_options: StreamChatOptions): AsyncIterable<string> {
    void _options;
  }
  async completeJSON<T>(_options: CompleteJSONOptions): Promise<T> {
    void _options;
    return this.response as T;
  }
}

let seq = 0;
function row(role: 'user' | 'agent', content: string, agentId: AgentId | null = null): MessageRow {
  seq += 1;
  return { seq, id: `m${seq}`, role, agent_id: agentId, content, status: 'completed', created_at: seq };
}

const silent = { effort: 'silent', thesis: '' };
const speak = (thesis: string, extra: Record<string, unknown> = {}) => ({ effort: 'standard', thesis, ...extra });

describe('Part 6: QA regressions', () => {
  describe('steering classification', () => {
    const activeAgents: AgentId[] = ['maya', 'liam'];

    it('treats "wrap up in one sentence" as a steer even when the model says stop', async () => {
      const router = new TurnRouter(
        new FixedJSONProvider({
          action: 'stop_all',
          targetAgents: activeAgents,
          directive: 'Wrap up in one short uppercase sentence.',
          removeRuleIds: [],
          standingRule: null,
        }),
      );
      const result = await router.classifySteering({
        userMessage: 'Actually, wrap up in one short uppercase sentence',
        activeAgents,
      });
      expect(result.action).toBe('steer');
      expect(result.targetAgents).toEqual(activeAgents);
      expect(result.directive).toMatch(/uppercase sentence/i);
    });

    it('still honours a real stop request the model classifies as stop', async () => {
      const router = new TurnRouter(
        new FixedJSONProvider({ action: 'stop_all', targetAgents: activeAgents, directive: '' }),
      );
      const result = await router.classifySteering({
        userMessage: 'ok everyone please stop writing now',
        activeAgents,
      });
      expect(result.action).toBe('stop_all');
    });
  });

  describe('follow-up rounds stay quiet unless a peer needs a reply', () => {
    const runtimeConfig = getScenario('nova-launch')!.runtimeConfig;

    it('lets only the first addressed agent reply when a peer vetoes both', async () => {
      const transcript = [
        row('user', 'Should we go with Cluster Beta and start the data migration right at Sunday handover?'),
        row('agent', 'Cluster Beta costs $12,800/month; the handover is Sunday 7:00 PM and go-live is Monday 9:00 AM.', 'maya'),
        row('agent', 'The Beta migration takes 14 hours end to end.', 'liam'),
        row(
          'agent',
          '@Maya, starting at the Sunday handover breaks our 60-minute handoff buffer. @Liam, the overnight checkpoints also violate our no-overnight rule.',
          'zoe',
        ),
      ];
      const router = new TurnRouter(
        new FixedJSONProvider({
          isResolved: false,
          maya: speak('Offer the alternative'),
          liam: speak('Respond to Zoe'),
          zoe: silent,
        }),
        runtimeConfig,
      );
      const plan = await router.routeWave({ waveIndex: 2, transcript, previousWaveSpeakers: ['zoe'] });
      expect(plan.speakers.map((s) => s.agentId)).toEqual(['maya']);
    });

    it('keeps an agent quiet when the peer already settled on a working option', async () => {
      const transcript = [
        row('user', 'Can we use VectorFlow for semantic search at launch, and what is the alternative?'),
        row('agent', 'VectorFlow stores data in a shared us-east-1 index. Self-hosted pgvector on Beta keeps data in our cluster.', 'liam'),
        row(
          'agent',
          "@Liam, VectorFlow violates St. Aurelia's EU residency clause. The self-hosted pgvector option works.",
          'zoe',
        ),
      ];
      const router = new TurnRouter(
        new FixedJSONProvider({ isResolved: false, maya: silent, liam: speak('Confirm pgvector'), zoe: silent }),
        runtimeConfig,
      );
      const plan = await router.routeWave({ waveIndex: 2, transcript, previousWaveSpeakers: ['zoe'] });
      expect(plan.speakers).toHaveLength(0);
    });

    it('rejects a co-speaker answer when the co-speaker never asked anything', async () => {
      const transcript = [
        row('user', 'Should we go with Cluster Beta and start the data migration right at Sunday handover?'),
        row('agent', 'Beta handover is Sunday 7:00 PM; go-live is Monday 9:00 AM.', 'maya'),
        row('agent', '@Maya, the 14-hour migration makes the early Saturday handover the safer option.', 'liam'),
        row('agent', '@Maya, starting Sunday evening leaves no room for our handoff buffer.', 'zoe'),
      ];
      const routeWith = (liamMessage: string) => {
        const t = transcript.map((m) => (m.agent_id === 'liam' ? { ...m, content: liamMessage } : m));
        const router = new TurnRouter(
          new FixedJSONProvider({
            isResolved: false,
            maya: silent,
            liam: silent,
            zoe: speak('Answer Liam', { answersCoSpeaker: 'liam' }),
          }),
          runtimeConfig,
        );
        return router.routeWave({ waveIndex: 2, transcript: t, previousWaveSpeakers: ['liam', 'zoe'] });
      };

      const unasked = await routeWith('@Maya, the 14-hour migration makes the early Saturday handover the safer option.');
      expect(unasked.speakers.map((s) => s.agentId)).not.toContain('zoe');

      const asked = await routeWith('@Zoe, does a Saturday 9:00 AM start break any team rule?');
      expect(asked.speakers.map((s) => s.agentId)).toContain('zoe');
    });
  });
});
