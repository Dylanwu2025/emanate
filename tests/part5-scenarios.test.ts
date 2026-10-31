import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { EmanateStore } from '../src/server/db.js';
import { ScriptableMockLLMProvider } from '../src/server/llm.js';
import { createAppServer, type AppServerInstance } from '../src/server/index.js';
import { AGENT_IDS, buildAgentSystemPrompt, getAgentVaultTopicSummary } from '../src/server/agents.js';
import { DEFAULT_SCENARIO_ID, getScenario, listScenarios } from '../src/server/scenarios.js';
import type { PublicScenario } from '../src/server/scenarios.js';

interface ScenariosResponse {
  activeScenarioId: string | null;
  scenarios: PublicScenario[];
}

async function switchScenario(baseUrl: string, scenarioId: string): Promise<Response> {
  return fetch(`${baseUrl}/api/scenario`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenarioId }),
  });
}

function enqueueSingleMayaTurn(mockLLM: ScriptableMockLLMProvider, reply: string): void {
  mockLLM.enqueueJSON({
    response: {
      maya: { effort: 'standard', thesis: 'Share the relevant quotes' },
      liam: { effort: 'silent', thesis: '' },
      zoe: { effort: 'silent', thesis: '' },
    },
  });
  mockLLM.enqueueStream({ match: /You are Maya/, chunks: [reply] });
  mockLLM.enqueueJSON({
    response: {
      isResolved: true,
      maya: { effort: 'silent', thesis: '' },
      liam: { effort: 'silent', thesis: '' },
      zoe: { effort: 'silent', thesis: '' },
    },
  });
}

describe('Part 5: Switchable Scenarios', () => {
  describe('scenario registry', () => {
    it('loads every scenario folder with a manifest and all three private vaults', () => {
      const ids = listScenarios().map((s) => s.id);
      expect(ids[0]).toBe(DEFAULT_SCENARIO_ID);
      expect(ids).toEqual(expect.arrayContaining(['kyoto-trip', 'nova-launch']));
    });

    it('builds agent prompts from the selected scenario only', () => {
      const nova = getScenario('nova-launch')!;
      const zoePrompt = buildAgentSystemPrompt('zoe', [], { runtimeConfig: nova.runtimeConfig });
      expect(zoePrompt).toContain('Security, Compliance & Team Health');
      expect(zoePrompt).toContain('St. Aurelia Health');
      expect(zoePrompt).not.toMatch(/Priya|Kyoto|shellfish/i);

      const kyoto = getScenario('kyoto-trip')!;
      const kyotoZoePrompt = buildAgentSystemPrompt('zoe', [], { runtimeConfig: kyoto.runtimeConfig });
      expect(kyotoZoePrompt).toContain('Priya');
      expect(kyotoZoePrompt).not.toContain('St. Aurelia');
    });

    it('keeps numbers out of the router-facing topic summary for every scenario', () => {
      for (const scenario of listScenarios()) {
        for (const agentId of AGENT_IDS) {
          const summary = getAgentVaultTopicSummary(agentId, scenario.runtimeConfig);
          expect(summary).not.toMatch(/\d{1,2}:\d{2}|\$\s*\d|\d+\s*(?:hours?|ms)\b/i);
        }
      }
    });
  });

  describe('scenario HTTP API', () => {
    let store: EmanateStore;
    let mockLLM: ScriptableMockLLMProvider;
    let app: AppServerInstance;
    let baseUrl: string;

    beforeEach(async () => {
      store = new EmanateStore(':memory:');
      mockLLM = new ScriptableMockLLMProvider();
      app = createAppServer({ store, llm: mockLLM });
      baseUrl = (await app.listen(0)).url;
    });

    afterEach(async () => {
      await app.close();
      store.close();
    });

    it('lists scenarios with display metadata and defaults to the trip', async () => {
      const res = await fetch(`${baseUrl}/api/scenarios`);
      expect(res.status).toBe(200);
      const data = (await res.json()) as ScenariosResponse;

      expect(data.activeScenarioId).toBe(DEFAULT_SCENARIO_ID);
      const nova = data.scenarios.find((s) => s.id === 'nova-launch')!;
      expect(nova.title).toBe('Project Nova EU Launch');
      expect(nova.starterPrompts.length).toBeGreaterThan(0);
      expect(nova.agents.map((a) => a.id)).toEqual(['maya', 'liam', 'zoe']);
      expect(nova.agents.find((a) => a.id === 'liam')!.shortRole).toBe('Migration & Performance');
      expect(JSON.stringify(data)).not.toMatch(/burnout|12,800|anaphylactic|EpiPen/i);
    });

    it('switches scenario, clears the chat, persists the choice, and routes with the new vaults', async () => {
      enqueueSingleMayaTurn(mockLLM, 'Flight B is $820.');
      await app.orchestrator.handleUserMessage('What does Flight B cost?');
      expect(app.orchestrator.getPublicMessages().length).toBeGreaterThan(0);

      const switchRes = await switchScenario(baseUrl, 'nova-launch');
      expect(switchRes.status).toBe(200);
      expect(app.orchestrator.getPublicMessages()).toHaveLength(0);
      expect(store.getSetting('activeScenarioId')).toBe('nova-launch');

      const listed = (await (await fetch(`${baseUrl}/api/scenarios`)).json()) as ScenariosResponse;
      expect(listed.activeScenarioId).toBe('nova-launch');

      mockLLM.streamCalls.length = 0;
      enqueueSingleMayaTurn(mockLLM, 'Cluster Beta is $12,800/month.');
      await app.orchestrator.handleUserMessage('What does Cluster Beta cost?');

      const mayaPrompt = mockLLM.streamCalls[0].messages[0].content;
      expect(mayaPrompt).toContain('Cloud Budget, Vendors & Contracts');
      expect(mayaPrompt).toContain('Cluster Beta');
      expect(mayaPrompt).not.toMatch(/Ryokan|Kyoto/);
    });

    it('restores the saved scenario when the server restarts on the same database', async () => {
      await switchScenario(baseUrl, 'nova-launch');
      const restarted = createAppServer({ store, llm: mockLLM });
      try {
        expect(restarted.orchestrator.runtimeConfig?.contextsDir).toContain('nova-launch');
      } finally {
        await restarted.close();
      }
    });

    it('rejects unknown scenarios without touching the chat', async () => {
      enqueueSingleMayaTurn(mockLLM, 'Flight B is $820.');
      await app.orchestrator.handleUserMessage('What does Flight B cost?');
      const before = app.orchestrator.getPublicMessages().length;

      const res = await switchScenario(baseUrl, 'does-not-exist');
      expect(res.status).toBe(404);
      expect(app.orchestrator.getPublicMessages()).toHaveLength(before);
      expect(store.getSetting('activeScenarioId')).toBeUndefined();
    });
  });
});
