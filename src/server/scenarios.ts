import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  AGENT_IDS,
  getAgentProfile,
  type AgentProfile,
  type CustomAgentRuntimeConfig,
} from './agents.js';
import type { AgentId } from './db.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SCENARIOS_DIR = join(__dirname, 'scenarios');
const MANIFEST_FILE = 'scenario.json';

export const DEFAULT_SCENARIO_ID = 'kyoto-trip';

type ScenarioAgentManifest = Partial<Pick<AgentProfile, 'roleTitle' | 'personality' | 'domainLaneRule'>> & {
  shortRole?: string;
};

interface ScenarioManifest {
  title: string;
  subtitle?: string;
  emblem?: string;
  welcomeTitle?: string;
  starterPrompts?: Array<{ label?: string; prompt: string }>;
  agents?: Partial<Record<AgentId, ScenarioAgentManifest>>;
}

export interface ScenarioDefinition {
  id: string;
  manifest: ScenarioManifest;
  runtimeConfig: CustomAgentRuntimeConfig;
}

export interface PublicScenario {
  id: string;
  title: string;
  subtitle: string;
  emblem: string;
  welcomeTitle: string;
  starterPrompts: Array<{ label: string; prompt: string }>;
  agents: Array<{ id: AgentId; name: string; roleTitle: string; shortRole: string }>;
}

function loadScenario(id: string, dir: string): ScenarioDefinition {
  const manifest = JSON.parse(readFileSync(join(dir, MANIFEST_FILE), 'utf-8')) as ScenarioManifest;
  if (!manifest.title) {
    throw new Error(`Scenario "${id}" is missing a title in ${MANIFEST_FILE}.`);
  }
  for (const agentId of AGENT_IDS) {
    const vaultFile = getAgentProfile(agentId).contextFile;
    if (!existsSync(join(dir, vaultFile))) {
      throw new Error(`Scenario "${id}" is missing the private vault ${vaultFile}.`);
    }
  }

  const profileOverrides: CustomAgentRuntimeConfig['profileOverrides'] = {};
  for (const agentId of AGENT_IDS) {
    const agentManifest = manifest.agents?.[agentId];
    if (!agentManifest) continue;
    const { roleTitle, personality, domainLaneRule } = agentManifest;
    const override: Partial<AgentProfile> = {};
    if (roleTitle) override.roleTitle = roleTitle;
    if (personality) override.personality = personality;
    if (domainLaneRule) override.domainLaneRule = domainLaneRule;
    if (Object.keys(override).length > 0) profileOverrides[agentId] = override;
  }

  return { id, manifest, runtimeConfig: { contextsDir: dir, profileOverrides } };
}

let cachedScenarios: Map<string, ScenarioDefinition> | null = null;

export function listScenarios(): ScenarioDefinition[] {
  if (!cachedScenarios) {
    cachedScenarios = new Map();
    const ids = readdirSync(SCENARIOS_DIR, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && existsSync(join(SCENARIOS_DIR, entry.name, MANIFEST_FILE)))
      .map((entry) => entry.name)
      .sort((a, b) => (a === DEFAULT_SCENARIO_ID ? -1 : b === DEFAULT_SCENARIO_ID ? 1 : a.localeCompare(b)));
    for (const id of ids) {
      cachedScenarios.set(id, loadScenario(id, join(SCENARIOS_DIR, id)));
    }
  }
  return Array.from(cachedScenarios.values());
}

export function getScenario(id: string): ScenarioDefinition | undefined {
  return listScenarios().find((s) => s.id === id);
}

export function toPublicScenario(scenario: ScenarioDefinition): PublicScenario {
  const { manifest, runtimeConfig } = scenario;
  return {
    id: scenario.id,
    title: manifest.title,
    subtitle: manifest.subtitle ?? '',
    emblem: manifest.emblem ?? manifest.title.slice(0, 2).toUpperCase(),
    welcomeTitle: manifest.welcomeTitle ?? manifest.title,
    starterPrompts: (manifest.starterPrompts ?? []).map((p) => ({
      label: p.label ?? p.prompt,
      prompt: p.prompt,
    })),
    agents: AGENT_IDS.map((agentId) => {
      const profile = getAgentProfile(agentId, runtimeConfig);
      return {
        id: agentId,
        name: profile.name,
        roleTitle: profile.roleTitle,
        shortRole: manifest.agents?.[agentId]?.shortRole ?? profile.roleTitle,
      };
    }),
  };
}
