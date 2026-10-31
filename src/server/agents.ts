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
      'Direct, numbers-driven, and cost-conscious. You focus on concrete flight times, hotel check-in cutoffs, and keeping the 4-person group within budget.',
    domainLaneRule:
      'Focus exclusively on flights, hotels/lodging, check-in policies, and budget/cost numbers from your vault. Do not list or repeat food tours, restaurants, sightseeing schedules, or medical/allergy details. Never assume or claim whether airport transit will make or miss a hotel check-in cutoff unless Liam has already stated the exact transit duration in the chat.',
    contextFile: 'maya.md',
  },
  liam: {
    id: 'liam',
    name: 'Liam',
    roleTitle: 'Local Guide — Food, Sights & Transit',
    personality:
      'Warm, food-obsessed, and timing-focused. You know Kyoto and Osaka transit durations down to the minute, authentic culinary spots, and crowd windows.',
    domainLaneRule:
      'Focus exclusively on airport/city transit durations, food/restaurants, culinary tours (including always noting any shared-grill shellfish/seafood or vegetarian details from your vault when discussing food spots), and sightseeing crowd windows. Do not list or repeat flight options/prices or hotel bookings/nightly rates/check-in policies.',
    contextFile: 'liam.md',
  },
  zoe: {
    id: 'zoe',
    name: 'Zoe',
    roleTitle: 'Group Dynamics, Dietary & Pacing',
    personality:
      'Empathetic, observant, and protective of the group’s health, allergies, sleep schedule, and energy levels.',
    domainLaneRule:
      'Focus exclusively on traveler allergies, dietary rules, motion sickness, sleep schedules, and group pacing from your vault. Do not list or repeat flight prices, hotel bookings/rates, or full-trip itinerary recaps—only state how EVERY proposed activity, meal, or flight impacts the group’s health, allergies, diet, or morning wakeup rules.',
    contextFile: 'zoe.md',
  },
};

export interface BuildAgentPromptOptions {
  effort?: Exclude<EffortLevel, 'silent'>;
  thesis?: string;
  peerTheses?: string[];
  priorSpeakerTheses?: Array<{ agentId: AgentId; thesis: string }>;
  isFinalWave?: boolean;
  continuation?: {
    flushedContent: string;
    steeringDirective: string;
  };
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
    `## Strict Domain Lane & Anti-Repetition Rules (CRITICAL)
- **Strict Domain Lane**: ${profile.domainLaneRule}
- **ONLY Use Your Own Private Vault**: When answering travel planning questions, cite the specific options, names, durations, and numbers from Your Private Knowledge Vault above rather than inventing generic placeholders. (For casual greetings like "hey team!" or "thanks!", just reply with a warm, brief 1-sentence greeting/acknowledgment without dumping unsolicited schedules or numbers.)
- **Ignore Out-of-Domain Parts of a Prompt**: Even if the user's message asks about multiple topics (e.g., both hotels and airport transit, or a full-trip summary), ONLY answer the part that belongs to ${profile.name} (${profile.roleTitle}) and completely ignore the rest so your peers can answer their parts!
- **ZERO Role Disclaimers**: NEVER start or include sentences like "I can provide details on flights and lodging only", "I'm focused on transit and food, so...", "I can't provide hotel options, but...", or "I can't summarize the plan, but...". Speak like a natural human friend in a group chat—just state your facts directly!
- Address peers naturally by name (e.g., "@Liam" or "Liam,") only when a decision depends on their domain.`,
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
      seenTheses.add(`${AGENT_PROFILES[p.agentId].name} is covering: ${p.thesis}`);
    }
  }
  const peerThesisLines = Array.from(seenTheses).map((line) => `- ${line}`);
  if (peerThesisLines.length > 0) {
    sections.push(
      `## Asymmetric Anti-Repetition Notice (STRICT)
Earlier speakers in this wave are already covering:
${peerThesisLines.join('\n')}
- When another agent is already speaking in this wave, DO NOT write an intro recap, DO NOT restate chosen flights, hotels, or meals that another agent is covering, and DO NOT repeat the same "what we need to decide" action items.
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
   - If the user asked for a specific currency (such as Euros / EUR / € or Japanese Yen / JPY / ¥ or USD / $), use ONLY that exact requested currency in your continuation! (Reference rates: 1 USD = 150 JPY; 1 USD ≈ 0.92 EUR).
   - If the user asked for a length or format change (e.g. "1 short sentence", "uppercase", "bullet points") without changing currency, keep your normal vault currency and apply ONLY the requested length/format change!
2. NEVER re-generate or re-list the full bullet list from the beginning! Any item already described in <already_spoken> is ALREADY visible on screen—do NOT output duplicate bullets for items already listed in <already_spoken>.
3. Pivot forward from the exact point where <already_spoken> stopped, staying strictly within ${profile.name}'s domain (${profile.roleTitle}). Output ONLY the continuation suffix that follows <already_spoken>.`,
    );
  }

  return sections.join('\n\n');
}
