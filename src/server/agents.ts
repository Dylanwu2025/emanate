import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { AgentId, StandingRuleRow } from './db.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEFAULT_CONTEXTS_DIR = join(__dirname, 'contexts');

export type EffortLevel = 'silent' | 'light' | 'standard' | 'deep';

export interface AgentProfile {
  id: AgentId;
  name: string;
  roleTitle: string;
  personality: string;
  domainLaneRule: string;
  contextFile: string;
}

export const AGENT_IDS: readonly AgentId[] = ['maya', 'liam', 'zoe'] as const;

export const AGENT_PROFILES: Record<AgentId, AgentProfile> = {
  maya: {
    id: 'maya',
    name: 'Maya',
    roleTitle: 'Budget, Flights & Lodging',
    personality:
      'Direct, numbers-driven, and cost-conscious. You focus on concrete options, pricing, cutoffs, and keeping the group within budget.',
    domainLaneRule:
      'Focus exclusively on the topics, options, pricing, and policies in your own Private Knowledge Vault. Do not list, summarize, or repeat topics that belong to your peers’ domains unless they have already shared a constraint that directly changes your recommendation.',
    contextFile: 'maya.md',
  },
  liam: {
    id: 'liam',
    name: 'Liam',
    roleTitle: 'Local Guide — Food, Sights & Transit',
    personality:
      'Warm, detail-oriented, and timing-focused. You focus on the concrete logistics, timings, and experiences in your private vault.',
    domainLaneRule:
      'Focus exclusively on the topics, timings, and experiences in your own Private Knowledge Vault (including always noting any relevant warnings or constraints documented in your vault for an option you mention). Do not list, summarize, or repeat options or prices from your peers’ domains.',
    contextFile: 'liam.md',
  },
  zoe: {
    id: 'zoe',
    name: 'Zoe',
    roleTitle: 'Group Dynamics, Dietary & Pacing',
    personality:
      'Empathetic, observant, and protective of the group’s constraints, health, and well-being.',
    domainLaneRule:
      'Focus exclusively on the group constraints, restrictions, and rules in your own Private Knowledge Vault. Do not repeat full recaps of your peers’ domains—only state how proposed options or schedules interact with the constraints in your vault.',
    contextFile: 'zoe.md',
  },
};

export interface CustomAgentRuntimeConfig {
  contextsDir?: string;
  vaultOverrides?: Partial<Record<AgentId, string>>;
  profileOverrides?: Partial<Record<AgentId, Partial<AgentProfile>>>;
}

export interface BuildAgentPromptOptions {
  effort?: Exclude<EffortLevel, 'silent'>;
  thesis?: string;
  peerTheses?: string[];
  priorSpeakerTheses?: Array<{ agentId: AgentId; thesis: string }>;
  isFinalWave?: boolean;
  runtimeConfig?: CustomAgentRuntimeConfig;
  continuation?: {
    flushedContent: string;
    steeringDirective: string;
  };
  steeringContinuation?: {
    flushedContent: string;
    redirectInstruction: string;
  };
}

export function getAgentProfile(
  agentId: AgentId,
  runtimeConfig?: CustomAgentRuntimeConfig,
): AgentProfile {
  const base = AGENT_PROFILES[agentId];
  if (!base) {
    throw new Error(`Unknown agentId: ${String(agentId)}`);
  }
  const override = runtimeConfig?.profileOverrides?.[agentId];
  return override ? { ...base, ...override } : base;
}

export function loadAgentPrivateContext(
  agentId: AgentId,
  runtimeConfig?: CustomAgentRuntimeConfig,
): string {
  const profile = getAgentProfile(agentId, runtimeConfig);
  const overrideContent = runtimeConfig?.vaultOverrides?.[agentId];
  if (typeof overrideContent === 'string') {
    return overrideContent.trim();
  }
  const baseDir = runtimeConfig?.contextsDir ?? DEFAULT_CONTEXTS_DIR;
  const filePath = join(baseDir, profile.contextFile);
  return readFileSync(filePath, 'utf-8').trim();
}

/**
 * Extracts high-level markdown section headings (`#` / `##` / `###`) from an agent's vault
 * without exposing any underlying bullet facts, numbers, or private details.
 * This allows the router to dynamically adapt if the user swaps out the markdown files for a new scenario!
 */
export function getAgentVaultTopicSummary(
  agentId: AgentId,
  runtimeConfig?: CustomAgentRuntimeConfig,
): string {
  try {
    const rawVault = loadAgentPrivateContext(agentId, runtimeConfig);
    const headings = rawVault
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => /^#{2,3}\s+/.test(line))
      .map((line) =>
        line
          .replace(/^#{2,3}\s+/, '')
          .replace(/\(.*?\)/g, '')
          .trim(),
      )
      .filter(Boolean);
    if (headings.length > 0) {
      return headings.join('; ');
    }
  } catch {
    // Fallback to roleTitle if vault file cannot be read
  }
  return getAgentProfile(agentId, runtimeConfig).roleTitle;
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
  light:
    'Keep your response to 1 short, crisp sentence (maximum ~25 words). Always finish with a complete sentence.',
  standard:
    'Keep your response concise: 2 to 4 clear sentences. Always finish with a complete sentence.',
  deep:
    'Provide a thorough, well-structured breakdown with concrete numbers, timings, or trade-offs. Always finish your final sentence completely without cutting off mid-thought.',
};

export const EFFORT_MAX_TOKENS: Record<Exclude<EffortLevel, 'silent'>, number> = {
  light: 120,
  standard: 320,
  deep: 800,
};

export function buildAgentSystemPrompt(
  agentId: AgentId,
  activeRules: StandingRuleRow[] = [],
  opts: BuildAgentPromptOptions = {},
): string {
  const profile = getAgentProfile(agentId, opts.runtimeConfig);
  const privateVault = loadAgentPrivateContext(agentId, opts.runtimeConfig);
  const applicableRules = filterApplicableStandingRules(agentId, activeRules);

  const peerProfiles = AGENT_IDS.filter((id) => id !== agentId).map((id) =>
    getAgentProfile(id, opts.runtimeConfig),
  );
  const peers = peerProfiles.map((p) => `${p.name} (${p.roleTitle})`).join(' and ');
  const peerDomainSummary = peerProfiles
    .map((p) => `${p.name}'s domain (${p.roleTitle})`)
    .join(' or ');

  const sections: string[] = [
    `You are ${profile.name}, the specialist for ${profile.roleTitle} in a live multi-agent group chat with ${peers} and the User.`,
    `Personality & Voice: ${profile.personality}`,
    `## Your Private Knowledge Vault (Exclusive Insider Notes for ${profile.name})\n${privateVault}`,
    `## How to Combine Your Private Vault with Your General Knowledge (CRITICAL)
- **1. Authoritative Private Insider Notes**: Your Private Knowledge Vault above contains non-public, group-specific facts, negotiated hold quotes, personal timings, or private constraints that only you have. Whenever a question touches something covered in your vault, treat your vault's numbers, names, and rules as authoritative and never contradict them.
- **2. Freely Use General Knowledge for Everything Else**: When the user (or a peer) asks a question in your domain that is NOT explicitly written in your private vault (for example: typical menu/meal prices at a restaurant, additional neighborhood recommendations, cultural tips, or reasonable estimates), **use your built-in general world knowledge to give a helpful, concrete answer or estimate**! NEVER say "I don't have that information, please check directly with the venue/restaurant."
- **3. Strict Domain Lane**: ${profile.domainLaneRule} Do NOT list, summarize, or repeat options or prices from ${peerDomainSummary}—even if they appear in the shared chat history. Do not repeat options or prices from your peers’ domains.
- **4. Ignore Out-of-Domain Parts of a Prompt**: Even if the user's message asks about multiple topics across different agents' domains, ONLY answer the part that belongs to ${profile.name} (${profile.roleTitle}) and leave the other parts for your peers!
- **5. ZERO Role Disclaimers**: NEVER start or include sentences like "I can provide details on ... only", "I'm focused on ..., so...", or "I can't provide ..., but...". Speak like a natural, knowledgeable human teammate in a group chat.
- **6. Addressing Peers with @Name**: When a decision depends on another agent's domain, address that peer directly using "@Maya", "@Liam", or "@Zoe" so they can chime in.`,
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
    sections.push(
      `## Assigned Focus for This Turn\nCore thesis to cover (stay strictly within this point and your domain): ${opts.thesis}`,
    );
  }

  const seenTheses = new Set<string>(opts.peerTheses ?? []);
  if (opts.priorSpeakerTheses && opts.priorSpeakerTheses.length > 0) {
    for (const p of opts.priorSpeakerTheses) {
      const peerName = getAgentProfile(p.agentId, opts.runtimeConfig).name;
      seenTheses.add(`${peerName} is covering: ${p.thesis}`);
    }
  }
  const peerThesisLines = Array.from(seenTheses).map((line) => `- ${line}`);
  if (peerThesisLines.length > 0) {
    sections.push(
      `## Asymmetric Anti-Repetition Notice (STRICT)
Earlier speakers in this wave are already covering:
${peerThesisLines.join('\n')}
- When another agent is already speaking in this wave, DO NOT write an intro recap, DO NOT restate options or facts that another agent is covering, and DO NOT repeat the same action items.
- Jump straight into your single unique domain point in 1 to 3 concise sentences.`,
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

  const activeContinuation = opts.continuation
    ? {
        flushedContent: opts.continuation.flushedContent,
        redirectInstruction: opts.continuation.steeringDirective,
      }
    : opts.steeringContinuation;

  if (activeContinuation) {
    sections.push(
      `## Mid-Stream Steering Continuation (CRITICAL — STRICT FORWARD PIVOT)
You were in the middle of writing a response and have ALREADY emitted the following prefix to the user's screen:
<already_spoken>${activeContinuation.flushedContent}</already_spoken>

The user just interjected with this live redirect instruction: "${activeContinuation.redirectInstruction}"
Continue seamlessly from the exact end of <already_spoken> to honor "${activeContinuation.redirectInstruction}" within the same message bubble:
1. STRICTLY OBEY THE USER'S EXACT REDIRECT ("${activeContinuation.redirectInstruction}"):
   - If the user asked for a specific currency or unit conversion, use ONLY that exact requested currency or unit in your continuation!
   - If the user asked for a length or format change (e.g. "1 short sentence", "uppercase", "bullet points") without changing currency, keep your normal vault units and apply ONLY the requested length/format change!
2. NEVER re-generate or re-list the full bullet list from the beginning! Any item already described in <already_spoken> is ALREADY visible on screen—do NOT output duplicate bullets for items already listed in <already_spoken>.
3. Pivot forward from the exact point where <already_spoken> stopped, staying strictly within ${profile.name}'s domain (${profile.roleTitle}). Output ONLY the continuation suffix that follows <already_spoken>.`,
    );
  }

  return sections.join('\n\n');
}
