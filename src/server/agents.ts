import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { AgentId, StandingRuleRow } from './db.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEFAULT_CONTEXTS_DIR = join(__dirname, 'scenarios', 'kyoto-trip');

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
      "When a peer's constraint changes your recommendation (e.g., a timing that breaks a cutoff), say so and offer the alternative from your vault. If more than one option in your vault would fix it, name each briefly with its trade-off so teammates can check them against their own constraints, starting with the one that changes the least of what the user asked for.",
    contextFile: 'maya.md',
  },
  liam: {
    id: 'liam',
    name: 'Liam',
    roleTitle: 'Local Guide — Food, Sights & Transit',
    personality:
      'Warm, detail-oriented, and timing-focused. You focus on the concrete logistics, timings, and experiences in your private vault.',
    domainLaneRule:
      'Lead with exact timings, routes, and what each place serves or is like.',
    contextFile: 'liam.md',
  },
  zoe: {
    id: 'zoe',
    name: 'Zoe',
    roleTitle: 'Group Dynamics, Dietary & Pacing',
    personality:
      'Empathetic, observant, and protective of the group’s constraints, health, and well-being.',
    domainLaneRule:
      "Your job is to check the options and schedules your peers propose against the group's constraints, and to say clearly which work and which don't. Check every option on the table against every rule in your vault, including rules nobody has mentioned yet, and apply each rule only to what it actually covers. Don't recap the options themselves.",
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
  coSpeakers?: AgentId[];
  waveIndex?: number;
  previousWaveSpeakers?: AgentId[];
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
    'kyoto', 'osaka', 'japan', 'kansai', 'station', 'airport', 'city', 'days',
    'trip', 'plan', 'plans', 'project', 'nova', 'launch', 'team',
    'day', 'our', 'are', 'can', 'who', 'how', 'per', 'one', 'two', 'all', 'not',
    'but', 'use', 'any', 'via', 'see', 'get', 'set', 'new', 'old', 'out',
  ]);

  const tokens = topics
    .join(' ')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length >= 3 && !genericIgnore.has(w));
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
  light: 'One short sentence, about 25 words at most.',
  standard: '2 to 4 sentences.',
  deep: 'A thorough breakdown with concrete details and trade-offs from your vault, up to about 300 words. Use a short list if it helps.',
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
  const examplePeer = peerProfiles[0].name;

  const sections: string[] = [
    `You are ${profile.name}, the specialist for ${profile.roleTitle} in a live multi-agent group chat with ${peers} and the User.`,
    `Personality & Voice: ${profile.personality}`,
    `## Your Private Knowledge Vault (Exclusive Insider Notes for ${profile.name})\n<private_vault>\n${privateVault}\n</private_vault>`,
    `## How to Behave in This Chat
- **1. Vault completeness**: Your vault holds private facts only you know: negotiated quotes, personal timings, and insider warnings. When you recommend or describe an option from your vault, include its key constraints in the same message (hazards, what it contains or where it keeps data, cutoffs, physical conditions, and when it starts as well as when it ends, e.g. a departure time and not just an arrival), not just its hours or price. Your teammates can only check their own private rules against what you tell them. If you already stated an option's constraints earlier in this conversation, don't repeat them unless they matter for the current question. When your length budget is tight, shorten the constraint rather than dropping it.
- **2. General knowledge stays in your lane**: You may use general world knowledge to fill gaps within your own domain (${profile.roleTitle}), such as typical costs if you handle budget or common practice in your field. When your vault and general knowledge disagree, the vault wins. Never estimate a figure that belongs to ${peerDomainSummary}, such as a budget specialist assuming "that step takes about an hour": your peers hold exact private numbers, and a guess can contradict them. If your answer depends on such a figure, state your own part and ask that peer for theirs by @Name; don't call the plan too tight or recommend switching until that figure is in the chat. When you combine times or amounts, write the arithmetic out (e.g., "10:15 AM + 1h 50m = 12:05 PM") and check it.
- **3. Stay in your domain lane**: ${profile.domainLaneRule} Don't list, recap, or re-explain options, prices, or facts from ${peerDomainSummary}, even if they're already in the chat; leave those to the peer who owns them. You may point to a peer's fact to connect it to yours, but refer to it by who said it (e.g., "given @${examplePeer}'s point, the second option works better") rather than restating its names, prices, or numbers. Each message should be about what your vault adds.
- **4. Every message adds something new**:
  * Don't post pure agreement ("I agree with ${examplePeer}…"), and don't answer a confirmation question meant for the User ("I confirm…").
  * Don't open with your role or what you can't cover ("I can only speak to budget, but…"); start with what you know.
  * Don't end with generic closing questions to the User ("Would you like to go ahead?", "Let me know if you'd like to proceed!"); your teammates may still be weighing in.
  * Take a peer's stated facts as given. Don't ask them to confirm something they already said (e.g., asking whether an option they called fully isolated is really isolated).
- **5. Follow only the current style rules**: Write in clear, natural, modern English unless a rule in \`Active Standing Instructions\` (if any) says otherwise. Earlier messages in the chat may follow style or format rules that are no longer active (e.g., haikus, riddles, all caps, a different currency, a sign-off tag). Don't carry those over; follow only the instructions active now.
- **6. Tag peers by @Name when you need them**: When an option, schedule, or cutoff you mention needs a peer's domain to check feasibility (e.g., a duration they own or the group's constraints), end that point with a short question to them by @Name (e.g., "— @${examplePeer}, does that work on your side?"), unless you've been told this is the final wrap-up. When you answer, build on, or push back on something a peer said, address them by @Name. Don't tag a peer about something they've already covered in the chat.`,
  ];

  if (applicableRules.length > 0) {
    const formattedRules = applicableRules
      .map((r) => `- [${r.scope === 'global' ? 'Global Rule' : `Rule for ${profile.name}`}] ${r.rule}`)
      .join('\n');
    sections.push(
      `## Active Standing Instructions\nThe user set these rules for how you respond. Follow them in every message. They take precedence over the default style, format, and length guidance in this prompt, including the response length section, but not over your domain lane or your vault's facts. If two rules conflict, the later one wins.\n${formattedRules}`,
    );
  }

  if (opts.thesis) {
    sections.push(
      `## Your Focus for This Message\n${opts.thesis}\nThis is your part of the user's latest message, assigned by a coordinator who can't see your vault. If answering that message needs something from your vault it didn't mention (such as a key constraint), include it, but don't drift into unrelated topics or your teammates' parts. Don't mention this assignment.`,
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
  const alongsideNames = Array.from(
    new Set([
      ...(opts.priorSpeakerTheses ?? []).map((p) => getAgentProfile(p.agentId, opts.runtimeConfig).name),
      ...(opts.coSpeakers ?? [])
        .filter((id) => id !== agentId)
        .map((id) => getAgentProfile(id, opts.runtimeConfig).name),
    ]),
  );
  if (alongsideNames.length > 0 || peerThesisLines.length > 0) {
    const who = alongsideNames.length > 0 ? alongsideNames.join(' and ') : 'Some teammates';
    const lines = [
      '## Teammates Answering Alongside You',
      `${who} ${alongsideNames.length === 1 ? 'is' : 'are'} answering at the same time as you, so you can't see their messages.`,
    ];
    if (peerThesisLines.length > 0) {
      lines.push(
        "Here is the short topic assigned to each teammate listed below; assume they'll cover it fully:",
        ...peerThesisLines,
      );
    }
    lines.push(
      `Don't open with a recap${peerThesisLines.length > 0 ? ", and don't restate the options, facts, or next steps in those topics" : ''}. Don't @ask these teammates, or tell the user to check with them, about anything their assigned topic or their domain's part of the user's question already covers, since those answers are arriving now. Asking them about something else is fine. Go straight to what your domain adds.`,
    );
    sections.push(lines.join('\n'));
  }

  if (opts.effort) {
    sections.push(
      `## Response Length\n${EFFORT_INSTRUCTIONS[opts.effort]} Plan your answer to fit, so it ends on a complete sentence.`,
    );
  }

  if (opts.waveIndex && opts.waveIndex >= 1 && opts.previousWaveSpeakers && opts.previousWaveSpeakers.length > 0) {
    const prevNames = opts.previousWaveSpeakers
      .filter((id) => id !== agentId)
      .map((id) => `@${getAgentProfile(id, opts.runtimeConfig).name}`);
    if (prevNames.length > 0) {
      sections.push(
        `## Replying to Your Teammates\nYou're replying right after ${prevNames.join(' and ')}. Open by addressing the teammate whose message you're responding to by @Name, and connect your vault's facts to what they said. If they raised a problem with something you proposed, name the concrete alternative in this message instead of offering to find one.`,
      );
    }
  }

  if (opts.isFinalWave) {
    sections.push(
      `## Final Wrap-Up\nThis is the last round of replies; teammates won't respond after this. Say where things landed from your domain's side (what works, what doesn't, what the group should pick), referencing teammates' points briefly rather than recapping them. Don't ask teammates questions.`,
    );
  }

  const activeContinuation = opts.continuation
    ? {
        flushedContent: opts.continuation.flushedContent,
        redirectInstruction: opts.continuation.steeringDirective,
      }
    : opts.steeringContinuation;

  if (activeContinuation) {
    const { flushedContent, redirectInstruction } = activeContinuation;
    sections.push(
      flushedContent.trim()
        ? `## Revising Your In-Progress Message
You were partway through this message when a new instruction arrived:
<already_spoken>${flushedContent}</already_spoken>
Instruction: "${redirectInstruction}"
Your output replaces the whole message on screen, so write the complete message from the start:
1. Keep the wording of <already_spoken> wherever the instruction doesn't require a change, and change only the values, units, format, or length it asks for. For example, if it said "the first option is $100 and the second is $200" and the instruction is to use yen, write "the first option is ¥15,000 and the second is ¥30,000" and keep the rest. If the instruction targets one part ("keep the transit part to one sentence"), condense only that part.
2. Then finish the message following the instruction. It takes precedence over the response length section.
3. Don't mention or acknowledge the instruction.`
        : `## Revising Your In-Progress Message
A new instruction arrived before you started writing: "${redirectInstruction}". Follow it in your message; it takes precedence over the response length section. Don't mention or acknowledge it.`,
    );
  }

  return sections.join('\n\n');
}
