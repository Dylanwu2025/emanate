import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { AgentId, StandingRuleRow } from './db.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONTEXTS_DIR = join(__dirname, 'contexts');

export type EffortLevel = 'silent' | 'light' | 'standard' | 'deep';

export interface AgentProfile {
  id: AgentId;
  name: string;
  roleTitle: string;
  personality: string;
  contextFile: string;
}

export const AGENT_IDS: readonly AgentId[] = ['maya', 'liam', 'zoe'] as const;

export const AGENT_PROFILES: Record<AgentId, AgentProfile> = {
  maya: {
    id: 'maya',
    name: 'Maya',
    roleTitle: 'Budget, Flights & Lodging',
    personality:
      'Direct, numbers-driven, and cost-conscious. You focus on concrete flight times, hotel check-in cutoffs, and keeping the 4-person group within budget.',
    contextFile: 'maya.md',
  },
  liam: {
    id: 'liam',
    name: 'Liam',
    roleTitle: 'Local Guide — Food, Sights & Transit',
    personality:
      'Warm, food-obsessed, and timing-focused. You know Kyoto and Osaka transit durations down to the minute, authentic culinary spots, and crowd windows.',
    contextFile: 'liam.md',
  },
  zoe: {
    id: 'zoe',
    name: 'Zoe',
    roleTitle: 'Group Dynamics, Dietary & Pacing',
    personality:
      'Empathetic, observant, and protective of the group’s health, allergies, sleep schedule, and energy levels.',
    contextFile: 'zoe.md',
  },
};

export interface BuildAgentPromptOptions {
  effort?: Exclude<EffortLevel, 'silent'>;
  thesis?: string;
  priorSpeakerTheses?: Array<{ agentId: AgentId; thesis: string }>;
  isFinalWave?: boolean;
  steeringContinuation?: {
    flushedContent: string;
    redirectInstruction: string;
  };
}

export function loadAgentPrivateContext(agentId: AgentId): string {
  const profile = AGENT_PROFILES[agentId];
  if (!profile) {
    throw new Error(`Unknown agentId: ${String(agentId)}`);
  }
  const filePath = join(CONTEXTS_DIR, profile.contextFile);
  return readFileSync(filePath, 'utf-8').trim();
}

export function filterApplicableStandingRules(
  agentId: AgentId,
  rules: StandingRuleRow[],
): StandingRuleRow[] {
  return rules.filter(
    (r) => r.active === 1 && (r.scope === 'global' || r.scope === agentId),
  );
}

const EFFORT_INSTRUCTIONS: Record<Exclude<EffortLevel, 'silent'>, string> = {
  light: 'Keep your response to 1 short, crisp sentence (maximum ~25 words).',
  standard: 'Keep your response concise: 2 to 4 clear sentences.',
  deep: 'Provide a thorough, well-structured breakdown with concrete numbers, timings, or trade-offs.',
};

export const EFFORT_MAX_TOKENS: Record<Exclude<EffortLevel, 'silent'>, number> = {
  light: 80,
  standard: 250,
  deep: 600,
};

export function buildAgentSystemPrompt(
  agentId: AgentId,
  activeRules: StandingRuleRow[] = [],
  opts: BuildAgentPromptOptions = {},
): string {
  const profile = AGENT_PROFILES[agentId];
  if (!profile) {
    throw new Error(`Unknown agentId: ${String(agentId)}`);
  }

  const privateVault = loadAgentPrivateContext(agentId);
  const applicableRules = filterApplicableStandingRules(agentId, activeRules);

  const peers = AGENT_IDS.filter((id) => id !== agentId)
    .map((id) => `${AGENT_PROFILES[id].name} (${AGENT_PROFILES[id].roleTitle})`)
    .join(' and ');

  const sections: string[] = [
    `You are ${profile.name}, the specialist for ${profile.roleTitle} in a live group chat with ${peers} and the User, planning a 5-day trip to Kyoto & Osaka for 4 friends (Sam, Priya, Alex, Jordan).`,
    `Personality & Voice: ${profile.personality}`,
    `## Your Private Knowledge Vault (Exclusive to ${profile.name})\n${privateVault}`,
    `## Collaboration & Non-Repetition Guidelines
- You ONLY have direct access to your own Private Knowledge Vault above. You do NOT have access to the private vaults of your peers unless they have already shared those facts in the conversation transcript.
- Address peers naturally by name (e.g., "@Liam" or "Liam,") when a decision depends on their domain or when a constraint in your vault interacts with theirs.
- Never repeat points already covered by another agent; add only your distinct domain facts and perspective.`,
  ];

  if (applicableRules.length > 0) {
    const formattedRules = applicableRules
      .map((r) => `- [${r.scope === 'global' ? 'Global Rule' : `Rule for ${profile.name}`}] ${r.rule}`)
      .join('\n');
    sections.push(
      `## Active Standing Instructions (Strictly Mandatory)\nYou MUST obey these standing rules in this and every response:\n${formattedRules}`,
    );
  }

  if (opts.thesis) {
    sections.push(`## Assigned Focus for This Turn\nCore thesis to cover: ${opts.thesis}`);
  }

  if (opts.priorSpeakerTheses && opts.priorSpeakerTheses.length > 0) {
    const priorLines = opts.priorSpeakerTheses
      .map((p) => `- ${AGENT_PROFILES[p.agentId].name} is already covering: "${p.thesis}"`)
      .join('\n');
    sections.push(
      `## Asymmetric Anti-Repetition Notice\nEarlier speakers in this wave are already covering:\n${priorLines}\nDo NOT repeat their points; contribute only your unique angle and private facts.`,
    );
  }

  if (opts.effort) {
    sections.push(`## Response Length & Effort Calibration (${opts.effort})\n${EFFORT_INSTRUCTIONS[opts.effort]}`);
  }

  if (opts.isFinalWave) {
    sections.push(
      `## Final Synthesis Wave\nWrap up the current exchange concisely with a clear conclusion and do not ask further open questions to other agents.`,
    );
  }

  if (opts.steeringContinuation) {
    sections.push(
      `## Mid-Stream Steering Continuation
You were in the middle of writing a response and have already emitted the following text to the chat bubble:
<already_spoken>${opts.steeringContinuation.flushedContent}</already_spoken>

The user just interjected with this live redirect instruction: "${opts.steeringContinuation.redirectInstruction}"
Continue seamlessly from the exact end of <already_spoken> to honor the redirect within the same message. Output ONLY the continuation suffix—do not repeat <already_spoken> or mention that you are being redirected.`,
    );
  }

  return sections.join('\n\n');
}
