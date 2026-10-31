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
 * Extracts high-level markdown section headings (`#` / `##` / `###`) and bolded option/entity names
 * (`**...**`) from an agent's vault while stripping any numbers, prices, times, or medical/allergy
 * secrets so zero private constraints leak into the router prompt!
 * This allows the router to dynamically know which named items live in which agent's vault
 * across ANY scenario (Kyoto trip, B2B cloud architecture, product launch, etc.).
 */
export function getAgentVaultTopicSummary(
  agentId: AgentId,
  runtimeConfig?: CustomAgentRuntimeConfig,
): string {
  try {
    const rawVault = loadAgentPrivateContext(agentId, runtimeConfig);
    const topics: string[] = [];

    for (const rawLine of rawVault.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (/^#{2,3}\s+/.test(line)) {
        const heading = line
          .replace(/^#{2,3}\s+/, '')
          .replace(/\(.*?\)/g, '')
          .replace(/\b\d[\d:.,]*\s*(?:AM|PM|USD|JPY|EUR|hours?|hrs?|mins?|minutes?)?\b/gi, '')
          .trim();
        if (heading) topics.push(heading);
        continue;
      }

      const itemMatch = line.match(/^(?:\d+[.)]|[-*•])\s+\*\*([^*:]+)\*\*\s*(?::|$)/);
      if (itemMatch) {
        const itemLabel = itemMatch[1]
          .replace(/\(.*?\)/g, '')
          .replace(/—.*$/, '')
          .replace(/\b\d[\d:.,]*\s*(?:AM|PM|USD|JPY|EUR|hours?|hrs?|mins?|minutes?)?\b/gi, '')
          .trim();
        if (
          itemLabel &&
          itemLabel.length >= 4 &&
          !/^(?:departure|arrival|price|notes|nightly\s+rate|check-?in|operating\s+window|food\s+profile|dietary\s+profile|hours|reserved|locked|negotiated|strict|total|currency|unbooked|insider|severe|allergy|motion\s+sickness|no\s+waking|midday)\b/i.test(
            itemLabel,
          )
        ) {
          topics.push(itemLabel);
        }
      }
    }

    if (topics.length > 0) {
      return Array.from(new Set(topics)).join('; ');
    }
  } catch {
    // Fallback to roleTitle if vault file cannot be read
  }
  return getAgentProfile(agentId, runtimeConfig).roleTitle;
}

/**
 * Extracts significant searchable keywords from an agent's roleTitle, section headings,
 * and bolded item/entity titles (without ever injecting those entity names into another agent's
 * prompt or the shared LLM router prompt) so Wave 0 can deterministically match user prompts
 * that name specific vault entities across ANY scenario!
 */
export function getAgentDomainIndexKeywords(
  agentId: AgentId,
  runtimeConfig?: CustomAgentRuntimeConfig,
): Set<string> {
  const profile = getAgentProfile(agentId, runtimeConfig);
  const topics: string[] = [profile.roleTitle];

  try {
    const rawVault = loadAgentPrivateContext(agentId, runtimeConfig);
    for (const rawLine of rawVault.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (/^#{2,3}\s+/.test(line)) {
        topics.push(line.replace(/^#{2,3}\s+/, '').replace(/\(.*?\)/g, ''));
        continue;
      }
      const itemMatch = line.match(/^(?:\d+[.)]|[-*•])\s+\*\*([^*:]+)\*\*\s*(?::|$)/);
      if (itemMatch) {
        topics.push(itemMatch[1].replace(/\(.*?\)/g, '').replace(/—.*$/, ''));
      }
    }
  } catch {
    // Fallback to roleTitle
  }

  const genericIgnore = new Set([
    'the', 'and', 'for', 'with', 'from', 'into', 'options', 'option', 'private',
    'group', 'notes', 'insider', 'hold', 'holds', 'rules', 'rule', 'total',
    'departure', 'arrival', 'price', 'nightly', 'rate', 'hours', 'strict',
  ]);

  const tokens = topics
    .join(' ')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length >= 4 && !genericIgnore.has(w));
  return new Set(tokens);
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
- **1. Authoritative Private Insider Notes & Complete Caveats**: Your Private Knowledge Vault above contains non-public, group-specific facts, negotiated hold quotes, personal timings, or private constraints that only you have. Whenever you present or compare an option from your vault, **ALWAYS include its critical operational caveats, warnings, or physical/dietary profiles** documented in your vault (not just its name or hours!) so your peers can immediately spot conflicts with their own private constraints.
- **2. General Knowledge is Strictly Scoped to YOUR Own Domain Only**: You may ONLY use general world knowledge for topics strictly inside your own role/domain (${profile.roleTitle})—for example, estimating typical prices/costs if you are the budget specialist, or suggesting local spots if you are the local guide. **NEVER use general knowledge to estimate, guess, or calculate a measurement that belongs to a peer's domain!** Specifically:
  * If you are ${profile.name} (${profile.roleTitle}), NEVER guess or estimate ${peerDomainSummary} (for example, a budget/pricing specialist must NEVER estimate airport customs or transit durations such as "assuming 1 hour to reach the venue"; a transit/technical guide must NEVER repeat lodging/contract check-in policies or prices without stating their own exact vault durations/metrics first; a group/compliance specialist must NEVER echo venue recommendations or check-in feasibility when no sleep, health, dietary, or compliance rule is triggered).
- **3. Strict Domain Lane & Never Guess Peers' Private Cutoffs/Timings**: ${profile.domainLaneRule} Do NOT list, summarize, or repeat options or prices from ${peerDomainSummary}—even if they appear in the shared chat history. Do not repeat options or prices from your peers’ domains. If answering the user's question requires a measurement from a peer's domain (like transit duration or latency), state ONLY your domain's fact and ask that peer by "@Name" to provide their measurement!
- **4. ZERO "I Agree" Echo Bubbles & ZERO Role Disclaimers**: NEVER send a message that merely agrees with another agent (e.g. "I agree with Maya; opting for that option would be the safer choice..."). Only speak when contributing a distinct fact or constraint from your own domain. NEVER start with "I can provide details on ... only", "I'm focused on ..., so...", or "I can't provide ..., but...".
- **5. Addressing Peers with @Name**: When a decision depends on another agent's domain that hasn't been shared in the chat yet, address that peer directly using "@Maya", "@Liam", or "@Zoe" so they can chime in.`,
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
      `## Mid-Stream Steering Continuation (CRITICAL — SURGICAL IN-PLACE RETROFIT & COMPLETION)
You were in the middle of writing a response and have emitted the following partial draft so far:
<already_spoken>${activeContinuation.flushedContent}</already_spoken>

The user just interjected with this live redirect instruction: "${activeContinuation.redirectInstruction}"
Your output will replace your message bubble in-place, so you MUST output the COMPLETE, updated message from the beginning by surgically editing <already_spoken> and then finishing the rest of your response:
1. PRESERVE UNAFFECTED WORDING VERBATIM: Keep the exact opening wording, sentence structure, and unaffected parts of <already_spoken>, changing ONLY the specific values, units, currency, format, or length required by "${activeContinuation.redirectInstruction}".
   - Example (Currency/Unit Change): If <already_spoken> was "Looking at the flight options, Flight A is $650 and Flight B is $820 " and the user says "convert all prices to Japanese Yen instead", output "Looking at the flight options, Flight A is ¥97,500 and Flight B is ¥123,000..." (preserving "Looking at the flight options, Flight A is ... and Flight B is ..." and only replacing the USD amounts with the requested currency, then completing the rest of your answer in that currency).
   - Example (Partial Length/Scope Change): If <already_spoken> had two sentences on food and started a paragraph on transit, and the user says "keep your transit part to one short sentence", keep the two food sentences verbatim and condense only the transit part into one short sentence.
2. STRICTLY OBEY THE USER'S EXACT REDIRECT ("${activeContinuation.redirectInstruction}"):
   - Change ONLY what the redirect asks for. Do not rewrite unrelated sentences, and do not mention or acknowledge that you were redirected.
3. Stay strictly within ${profile.name}'s domain (${profile.roleTitle})—do not answer parts of the prompt that belong to your peers' domains.`,
    );
  }

  return sections.join('\n\n');
}
