import {
  AGENT_IDS,
  AGENT_PROFILES,
  EFFORT_MAX_TOKENS,
  type EffortLevel,
} from './agents.js';
import type { AgentId, MessageRow, RuleScope, StandingRuleRow } from './db.js';
import type { LLMProvider } from './llm.js';

export interface AgentRouteDecision {
  effort: EffortLevel;
  thesis: string;
}

export interface ExtractedStandingRule {
  scope: RuleScope;
  rule: string;
}

export interface RawRouterLLMResponse {
  maya?: Partial<AgentRouteDecision>;
  liam?: Partial<AgentRouteDecision>;
  zoe?: Partial<AgentRouteDecision>;
  agents?: Partial<Record<AgentId, Partial<AgentRouteDecision>>>;
  isResolved?: boolean;
  standingRule?: ExtractedStandingRule | null;
}

export interface SpeakerPlan {
  agentId: AgentId;
  effort: Exclude<EffortLevel, 'silent'>;
  maxTokens: number;
  thesis: string;
  peerTheses: string[];
  priorSpeakerTheses: Array<{ agentId: AgentId; thesis: string }>;
}

export interface WaveRoutingPlan {
  waveIndex: number;
  isResolved: boolean;
  decisions: Record<AgentId, AgentRouteDecision>;
  speakers: SpeakerPlan[];
  standingRule: ExtractedStandingRule | null;
}

export interface RouteWaveInput {
  waveIndex: number;
  transcript: MessageRow[];
  activeRules?: StandingRuleRow[];
  latestUserMessage?: string;
  signal?: AbortSignal;
}

const VALID_EFFORTS = new Set<EffortLevel>(['silent', 'light', 'standard', 'deep']);
const VALID_SCOPES = new Set<RuleScope>(['global', 'maya', 'liam', 'zoe']);

const THESIS_STOPWORDS = new Set([
  'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'about', 'what',
  'when', 'where', 'which', 'how', 'much', 'many', 'are', 'was', 'were', 'will',
  'would', 'should', 'could', 'can', 'may', 'might', 'flight', 'flights', 'hotel',
  'kyoto', 'osaka', 'trip', 'group', 'share', 'explain', 'state', 'provide',
  'discuss', 'mention', 'note', 'cover', 'answer', 'question', 'details', 'info',
]);

function normalizeThesisTokens(thesis: string): Set<string> {
  const words = thesis
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 2 && !THESIS_STOPWORDS.has(w));
  return new Set(words);
}

export function areThesesRedundant(earlierThesis: string, laterThesis: string): boolean {
  const normEarlier = earlierThesis.trim().toLowerCase();
  const normLater = laterThesis.trim().toLowerCase();
  if (!normLater) return true;
  if (normEarlier === normLater) return true;
  if (normEarlier.includes(normLater) || normLater.includes(normEarlier)) {
    return true;
  }

  const earlierTokens = normalizeThesisTokens(normEarlier);
  const laterTokens = normalizeThesisTokens(normLater);
  if (earlierTokens.size === 0 || laterTokens.size === 0) {
    return false;
  }

  let overlap = 0;
  for (const token of laterTokens) {
    if (earlierTokens.has(token)) {
      overlap++;
    }
  }

  const ratioLater = overlap / laterTokens.size;
  const ratioEarlier = overlap / earlierTokens.size;
  return overlap >= 3 && (ratioLater >= 0.75 || ratioEarlier >= 0.75);
}

export function extractMentionedAgents(text: string): AgentId[] {
  if (!text) return [];
  const matches: Array<{ agentId: AgentId; index: number }> = [];

  for (const agentId of AGENT_IDS) {
    const name = AGENT_PROFILES[agentId].name;
    const regex = new RegExp(`(?:^|[^a-zA-Z0-9])@?${name}\\b`, 'i');
    const match = regex.exec(text);
    if (match) {
      matches.push({ agentId, index: match.index });
    }
  }

  matches.sort((a, b) => a.index - b.index);
  return matches.map((m) => m.agentId);
}

export function computeDeterministicSpeakerOrder(referenceText = ''): AgentId[] {
  const mentioned = extractMentionedAgents(referenceText);
  const ordered: AgentId[] = [];
  for (const id of mentioned) {
    if (!ordered.includes(id)) {
      ordered.push(id);
    }
  }
  for (const id of AGENT_IDS) {
    if (!ordered.includes(id)) {
      ordered.push(id);
    }
  }
  return ordered;
}

export function applyAsymmetricDedup(
  rawDecisions: Record<AgentId, AgentRouteDecision>,
  speakerOrder: AgentId[],
): {
  decisions: Record<AgentId, AgentRouteDecision>;
  speakers: SpeakerPlan[];
} {
  const decisions: Record<AgentId, AgentRouteDecision> = {
    maya: { ...rawDecisions.maya },
    liam: { ...rawDecisions.liam },
    zoe: { ...rawDecisions.zoe },
  };

  const speakers: SpeakerPlan[] = [];
  const acceptedPrior: Array<{ agentId: AgentId; thesis: string }> = [];

  for (const agentId of speakerOrder) {
    const decision = decisions[agentId];
    if (!decision || decision.effort === 'silent') {
      decisions[agentId] = { effort: 'silent', thesis: '' };
      continue;
    }

    const trimmedThesis = decision.thesis.trim();
    const isRedundant = acceptedPrior.some((prior) =>
      areThesesRedundant(prior.thesis, trimmedThesis),
    );

    if (isRedundant) {
      decisions[agentId] = { effort: 'silent', thesis: '' };
      continue;
    }

    const peerTheses = acceptedPrior.map(
      (prior) => `${AGENT_PROFILES[prior.agentId].name} is covering: ${prior.thesis}`,
    );
    const priorSpeakerTheses = acceptedPrior.map((prior) => ({ ...prior }));

    speakers.push({
      agentId,
      effort: decision.effort,
      maxTokens: EFFORT_MAX_TOKENS[decision.effort],
      thesis: trimmedThesis,
      peerTheses,
      priorSpeakerTheses,
    });

    acceptedPrior.push({ agentId, thesis: trimmedThesis });
  }

  return { decisions, speakers };
}

function normalizeAgentDecision(raw?: Partial<AgentRouteDecision>): AgentRouteDecision {
  const effort = raw?.effort && VALID_EFFORTS.has(raw.effort) ? raw.effort : 'silent';
  const thesis = typeof raw?.thesis === 'string' ? raw.thesis.trim() : '';
  if (effort === 'silent') {
    return { effort: 'silent', thesis: '' };
  }
  return {
    effort,
    thesis: thesis || 'Provide relevant domain facts from private vault.',
  };
}

function normalizeStandingRule(raw?: ExtractedStandingRule | null): ExtractedStandingRule | null {
  if (!raw || typeof raw !== 'object') return null;
  const scope = raw.scope;
  const rule = typeof raw.rule === 'string' ? raw.rule.trim() : '';
  if (!scope || !VALID_SCOPES.has(scope) || !rule) {
    return null;
  }
  return { scope, rule };
}

function formatTranscriptForRouter(transcript: MessageRow[]): string {
  if (transcript.length === 0) return '(No messages yet)';
  return transcript
    .map((m) => {
      const speaker =
        m.role === 'user'
          ? 'User'
          : m.agent_id
            ? AGENT_PROFILES[m.agent_id].name
            : 'Agent';
      return `[seq=${m.seq}] ${speaker}: ${m.content}`;
    })
    .join('\n');
}

export class TurnRouter {
  constructor(private readonly llm: LLMProvider) {}

  async routeWave(input: RouteWaveInput): Promise<WaveRoutingPlan> {
    const { waveIndex, transcript, activeRules = [], latestUserMessage, signal } = input;
    const isFollowUpWave = waveIndex > 0;

    const rulesSummary =
      activeRules.length > 0
        ? activeRules.map((r) => `- [${r.scope}] ${r.rule}`).join('\n')
        : '(None)';

    const systemPrompt = isFollowUpWave
      ? `You are the A2A Follow-Up Wave Router for a 3-agent Kyoto & Osaka trip planning group chat:
- maya (Maya — Budget, Flights & Lodging: knows the $4,800 budget, Flight A 6:45 AM $650 vs Flight B 11:30 AM $820, Ryokan Gion $380/night with strict 6:00 PM check-in cutoff vs Hotel Granvia $290/night 24h check-in, Cross Hotel Osaka $260/night).
- liam (Liam — Local Guide for Food, Sights & Transit: knows KIX to Gion takes 2h 15m, KIX to Kyoto Station takes 1h 35m, Nishiki Market 11 AM–2 PM has heavy shared-grill shellfish, Arashiyama Tofu Kaiseki is 100% vegetarian & shellfish-free until 7:30 PM, sight crowd windows).
- zoe (Zoe — Group Dynamics, Dietary & Pacing: knows Sam has a severe shellfish allergy and boat/bus motion sickness, Priya is strict vegetarian, and the group has a strict rule against waking before 7:30 AM which rules out 6:45 AM flights).

Analyze the latest wave of agent responses in the transcript:
1. Determine if an agent asked a peer a question (e.g., "@Liam...") OR revealed a constraint/option that requires another agent's private vault facts to verify or resolve (e.g., Flight B landing at 3:45 PM + 2h 15m transit = 6:00 PM sharp cutoff risk, or proposing Flight A 6:45 AM which violates Zoe's 7:30 AM wakeup rule, or Nishiki Market conflicting with Sam's shellfish allergy).
2. If the exchange has already synthesized the needed facts and answered the user's question without unresolved peer questions or unflagged constraints, set "isResolved": true and set all agents to "silent".
3. Otherwise set "isResolved": false and assign ONLY the specific agent(s) who have new private facts or answers to contribute an effort level ("light" | "standard" | "deep") and a distinct 1-line "thesis". Agents with nothing new to add MUST be "silent".

Return strict JSON:
{
  "isResolved": boolean,
  "maya": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "liam": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "zoe": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string }
}`
      : `You are the Pre-Response Effort & Non-Repetition Router for a 3-agent Kyoto & Osaka trip planning group chat:
- maya (Maya — Budget, Flights & Lodging: private vault has group budget $4,800, Flight A 6:45 AM $650 vs Flight B 11:30 AM $820, Ryokan Gion $380/night strict 6:00 PM cutoff vs Hotel Granvia $290/night 24h check-in, Cross Hotel Osaka $260/night).
- liam (Liam — Local Guide for Food, Sights & Transit: private vault has KIX to Gion 2h 15m transit time, KIX to Kyoto Station 1h 35m, Nishiki Market 11 AM–2 PM shared-grill shellfish, Arashiyama Tofu Kaiseki vegetarian/shellfish-free until 7:30 PM, crowd windows).
- zoe (Zoe — Group Dynamics, Dietary & Pacing: private vault has Sam's severe shellfish allergy & bus/boat motion sickness, Priya's strict vegetarianism, and the group's no-waking-before-7:30-AM rule).

Rules for routing Wave 0:
1. Silence Gate: If an agent's domain is not relevant to the user's message, set their effort to "silent" and thesis to "".
2. Effort Calibration:
   - "light": Simple factual check or quick confirmation (1 short sentence).
   - "standard": Moderate multi-fact answer (2–4 sentences).
   - "deep": Complex trade-off or full itinerary/budget breakdown.
3. Distinct Theses: Each speaking agent MUST have a distinct 1-line "thesis" focused strictly on their own private domain so agents never repeat each other.
4. Standing Rule Extraction: If the user message establishes a persistent instruction for how agents should respond (e.g. "Always answer in bullet points", "Maya, quote all prices in JPY from now on"), return it in "standingRule": { "scope": "global" | "maya" | "liam" | "zoe", "rule": "..." }, otherwise null.

Return strict JSON:
{
  "maya": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "liam": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "zoe": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "standingRule": { "scope": "global" | "maya" | "liam" | "zoe", "rule": string } | null
}`;

    const userPrompt = `Active Standing Rules:\n${rulesSummary}\n\nConversation Transcript:\n${formatTranscriptForRouter(transcript)}\n\nRoute Wave ${waveIndex} now.`;

    const raw = await this.llm.completeJSON<RawRouterLLMResponse>({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      maxTokens: 350,
      temperature: 0.1,
      signal,
    });

    const rawDecisions: Record<AgentId, AgentRouteDecision> = {
      maya: normalizeAgentDecision(raw.maya ?? raw.agents?.maya),
      liam: normalizeAgentDecision(raw.liam ?? raw.agents?.liam),
      zoe: normalizeAgentDecision(raw.zoe ?? raw.agents?.zoe),
    };

    const referenceText =
      latestUserMessage ??
      (transcript.length > 0 ? transcript[transcript.length - 1].content : '');

    const speakerOrder = computeDeterministicSpeakerOrder(referenceText);
    const { decisions, speakers } = applyAsymmetricDedup(rawDecisions, speakerOrder);

    const isResolved = isFollowUpWave
      ? Boolean(raw.isResolved) || speakers.length === 0
      : speakers.length === 0;

    if (isFollowUpWave && raw.isResolved) {
      for (const id of AGENT_IDS) {
        decisions[id] = { effort: 'silent', thesis: '' };
      }
      speakers.length = 0;
    }

    return {
      waveIndex,
      isResolved,
      decisions,
      speakers,
      standingRule: isFollowUpWave ? null : normalizeStandingRule(raw.standingRule),
    };
  }
}
