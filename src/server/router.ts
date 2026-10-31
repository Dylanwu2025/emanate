import {
  AGENT_IDS,
  AGENT_PROFILES,
  EFFORT_MAX_TOKENS,
  type EffortLevel,
} from './agents.js';
import type { AgentId, MessageRow, RuleScope, StandingRuleRow } from './db.js';
import { ScriptableMockLLMProvider, type LLMProvider } from './llm.js';

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
  clearRuleScope?: RuleScope | null;
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
  clearRuleScope?: RuleScope | null;
}

export interface RouteWaveInput {
  waveIndex: number;
  transcript: MessageRow[];
  activeRules?: StandingRuleRow[];
  latestUserMessage?: string;
  previousWaveSpeakers?: AgentId[];
  turnThesesByAgent?: Partial<Record<AgentId, string[]>>;
  signal?: AbortSignal;
}

export type SteeringAction = 'steer' | 'stop_all' | 'new_turn';

export interface SteeringClassification {
  action: SteeringAction;
  targetAgents: AgentId[];
  directive: string;
  standingRule: ExtractedStandingRule | null;
  clearRuleScope: RuleScope | null;
}

export interface RawSteeringLLMResponse {
  action?: SteeringAction;
  targetAgents?: AgentId[];
  directive?: string;
  standingRule?: ExtractedStandingRule | null;
  clearRuleScope?: RuleScope | null;
}

export interface ClassifySteeringInput {
  userMessage: string;
  activeAgents: AgentId[];
  transcript?: MessageRow[];
  activeRules?: StandingRuleRow[];
  signal?: AbortSignal;
}

const VALID_EFFORTS = new Set<EffortLevel>(['silent', 'light', 'standard', 'deep']);
const VALID_SCOPES = new Set<RuleScope>(['global', 'maya', 'liam', 'zoe']);
const VALID_STEERING_ACTIONS = new Set<SteeringAction>(['steer', 'stop_all', 'new_turn']);

const THESIS_STOPWORDS = new Set([
  'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'about', 'what',
  'when', 'where', 'which', 'how', 'much', 'many', 'are', 'was', 'were', 'will',
  'would', 'should', 'could', 'can', 'may', 'might', 'flight', 'flights', 'hotel',
  'kyoto', 'osaka', 'trip', 'group', 'share', 'explain', 'state', 'provide',
  'discuss', 'mention', 'note', 'cover', 'answer', 'question', 'details', 'info',
  'option', 'options', 'confirm', 'choose', 'prefer', 'preferred', 'finalize',
  'travel', 'plans', 'plan', 'between', 'recommend', 'stay', 'summarize', 'summary',
  'recap', 'current', 'status', 'itinerary', 'decide', 'decisions', 'decision',
  'pending', 'need', 'needed', 'far', 'overall', 'update', 'outline', 'review',
  'agreed', 'chosen', 'next', 'steps',
]);

const GENERAL_RECAP_RE =
  /\b(summarize|summary|recap|plan\s+so\s+far|plans\s+so\s+far|current\s+(?:travel\s+|trip\s+)?(?:plan|itinerary|status)|what\s+(?:we\s+)?need\s+to\s+decide|pending\s+decisions|open\s+decisions|decisions\s+needed)\b/i;

export function isGeneralRecapQuery(text: string): boolean {
  return GENERAL_RECAP_RE.test(text);
}

const CASUAL_SOCIAL_TURN_RE =
  /^(?:hey|hi|hello|yo|good\s+(?:morning|afternoon|evening)|thanks|thank\s+you|thx|ty|got\s+it|sounds\s+good|awesome|great|cool|ok|okay|nice|perfect)(?:\s+(?:everyone|team|all|guys|folks|maya|liam|zoe))?[!.,\s]*$/i;

export function isCasualSocialMessage(text: string): boolean {
  return CASUAL_SOCIAL_TURN_RE.test(text.trim());
}

const DOMAIN_SAFE_THESIS_FALLBACKS: Record<AgentId, string> = {
  maya: 'Share ONLY the flight options, hotel rates/check-in cutoffs, and budget numbers from your private vault.',
  liam: 'Share ONLY the exact airport/city transit durations and the named food experiences from your private vault.',
  zoe: 'Share ONLY how the options affect traveler allergies, motion sickness, vegetarian diet, and the group morning wake-up rule from your private vault.',
};

const CROSS_DOMAIN_FORBIDDEN_IN_THESIS: Record<AgentId, RegExp> = {
  maya: /\b(transit\s+time|airport\s+transit|haruka|customs|food|dinner|lunch|restaurant|market|sightseeing|allergy|shellfish|vegetarian|wake-?up|sleep|7:30)\b/i,
  liam: /\b(flight\s+[ab]|flight\s+options?|hotel\s+options?|lodging\s+options?|ryokan|granvia|cross\s+hotel|budget|price|cost|check-?in|allergy|shellfish|vegetarian|wake-?up|sleep)\b/i,
  zoe: /\b(flight\s+price|hotel\s+price|nightly\s+rate|transit\s+time|airport\s+transit|summarize\s+the\s+plan)\b/i,
};

const REAL_LLM_DISALLOWED_THESIS_RE: Record<AgentId, RegExp> = {
  maya: /\b(transit\s+time|airport\s+transit|haruka|customs|food|dinner|lunch|restaurant|market|sightseeing|allergy|shellfish|vegetarian|wake-?up|sleep|7:30)\b/i,
  liam: /\b(flight\s+[ab]|flight\s+options?|hotel\s+options?|lodging\s+options?|ryokan|granvia|cross\s+hotel|budget|price|cost|check-?in|allergy|shellfish|vegetarian|wake-?up|sleep|izakaya|ramen|sushi)\b/i,
  zoe: /\b(flight\s+price|hotel\s+price|nightly\s+rate|transit\s+time|airport\s+transit|summarize\s+the\s+plan)\b/i,
};

// Detects when the LLM router invents specific fake prices, hours, or clock times inside a thesis
const HALLUCINATED_NUMBERS_IN_THESIS_RE =
  /(?:[$¥€£]\s*\d|\b\d[\d,]*\s*(?:usd|jpy|eur|yen|dollars?|euros?)\b|\b\d{1,2}:\d{2}\s*(?:am|pm)?\b|\b\d+\s*(?:am|pm)\b|\b\d+\s*(?:hours?|hrs?|minutes?|mins?)\b)/i;

export function sanitizeThesisForAgentDomain(
  agentId: AgentId,
  thesis: string,
  isMock = true,
): string | null {
  const trimmed = thesis.trim();
  if (!trimmed) return DOMAIN_SAFE_THESIS_FALLBACKS[agentId];
  const pattern = isMock
    ? CROSS_DOMAIN_FORBIDDEN_IN_THESIS[agentId]
    : REAL_LLM_DISALLOWED_THESIS_RE[agentId];
  if (pattern.test(trimmed)) {
    return null;
  }
  if (!isMock && HALLUCINATED_NUMBERS_IN_THESIS_RE.test(trimmed)) {
    return DOMAIN_SAFE_THESIS_FALLBACKS[agentId];
  }
  return trimmed;
}

const SPECIFIC_CONSTRAINT_CUE_RE =
  /\b(allergy|allergies|dietary|vegetarian|wake|sleep|pacing|motion\s+sickness|cutoff|transit\s+time|crowd\s+window|conflict|unsafe|veto)\b/i;

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

  if (
    GENERAL_RECAP_RE.test(normEarlier) &&
    GENERAL_RECAP_RE.test(normLater) &&
    !SPECIFIC_CONSTRAINT_CUE_RE.test(normLater)
  ) {
    return true;
  }

  const earlierTokens = normalizeThesisTokens(normEarlier);
  const laterTokens = normalizeThesisTokens(normLater);
  if (earlierTokens.size === 0 && laterTokens.size === 0) {
    return true;
  }
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
  return (overlap >= 3 && (ratioLater >= 0.7 || ratioEarlier >= 0.7)) || ratioLater >= 0.85;
}

export function isSelfRepetitiveForAgent(
  proposedThesis: string,
  priorThesesForAgent: string[],
  priorMessagesForAgent: string[],
): boolean {
  const trimmed = proposedThesis.trim();
  if (!trimmed) return true;

  if (
    /^(?:ask|prompt|remind|request|wait\s+for)\s+(?:the\s+)?(?:user|group|everyone)\s+to\s+(?:choose|confirm|select|decide|pick|let\s+us\s+know)/i.test(
      trimmed,
    ) ||
    /\b(?:confirm\s+(?:the\s+)?preferred\s+flight|choose\s+between\s+flight\s+a\s+(?:and|or)\s+flight\s+b|finalize\s+travel\s+plans)\b/i.test(
      trimmed,
    )
  ) {
    return true;
  }

  for (const oldThesis of priorThesesForAgent) {
    if (areThesesRedundant(oldThesis, trimmed)) {
      return true;
    }
  }

  const proposedTokens = normalizeThesisTokens(trimmed);
  for (const oldMsg of priorMessagesForAgent) {
    if (areThesesRedundant(oldMsg, trimmed)) {
      return true;
    }
    if (proposedTokens.size > 0) {
      const msgTokens = normalizeThesisTokens(oldMsg);
      let coveredCount = 0;
      for (const tok of proposedTokens) {
        if (msgTokens.has(tok)) {
          coveredCount++;
        }
      }
      if (coveredCount / proposedTokens.size >= 0.8) {
        return true;
      }
    }
  }

  return false;
}

const ROLE_ALIAS_PATTERNS: Array<{ agentId: AgentId; pattern: RegExp }> = [
  {
    agentId: 'maya',
    pattern: /\b(budget\s+(?:person|agent|expert|planner|specialist)|flight\s+(?:person|agent|expert|planner)|hotel\s+(?:person|agent|expert|planner)|lodging\s+(?:person|agent|expert|planner))\b/i,
  },
  {
    agentId: 'liam',
    pattern: /\b(local\s+guide|food\s+(?:person|agent|guide|expert)|transit\s+(?:person|agent|guide|expert)|sights\s+(?:person|agent|guide))\b/i,
  },
  {
    agentId: 'zoe',
    pattern: /\b(allergy\s+(?:person|agent|expert)|dietary\s+(?:person|agent|expert)|pacing\s+(?:person|agent|expert)|group\s+dynamics\s+(?:person|agent))\b/i,
  },
];

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

  for (const alias of ROLE_ALIAS_PATTERNS) {
    if (matches.some((m) => m.agentId === alias.agentId)) continue;
    const match = alias.pattern.exec(text);
    if (match) {
      matches.push({ agentId: alias.agentId, index: match.index });
    }
  }

  matches.sort((a, b) => a.index - b.index);
  return matches.map((m) => m.agentId);
}

export function isExplicitStopCommand(text: string): boolean {
  const cleaned = text.trim().toLowerCase().replace(/[.!?]+$/g, '').trim();
  if (!cleaned) return false;
  const exactStopPhrases = new Set([
    '/stop',
    'stop',
    'stop all',
    'stop everyone',
    'everyone stop',
    'halt',
    'halt all',
    'cancel',
    'cancel all',
    'quiet',
    'be quiet',
    'shut up',
    'stop talking',
    'stop writing',
    'stop streaming',
    'stop generating',
    'pause',
    'enough',
  ]);
  if (exactStopPhrases.has(cleaned)) {
    return true;
  }
  return /^(?:please\s+)?(?:everyone\s+)?(?:stop|halt|cancel)(?:\s+(?:talking|writing|streaming|generating|everything|all|now|immediately))?$/i.test(
    cleaned,
  );
}

export function detectRuleClearScope(text: string, mentionedAgents: AgentId[]): RuleScope | null {
  const lower = text.trim().toLowerCase();
  if (isExplicitStopCommand(lower)) return null;

  const isClearIntent =
    /\b(?:forget|clear|remove|drop|cancel|reset|stop\s+following)\b.*\b(?:standing\s+rules?|rules?|instructions?|format|prefix|haikus?|jpy|yen|euros?|bullet\s+points?|uppercase)\b/i.test(
      lower,
    ) ||
    /\b(?:stop\s+(?:using|answering\s+in|writing\s+in|starting\s+(?:every\s+|all\s+)?(?:replies|responses|messages)\s+with|including)|no\s+longer\s+need\s+to|no\s+more\s+(?:haikus?|bullets?|bullet\s+points?|prefixes|tags|jpy|yen|euros?|uppercase)|don'?t\s+use\s+(?:haikus?|bullets?|prefixes)|answer\s+in\s+\d*\s*normal\s+sentences?|without\s+any\s+prefix|go\s+back\s+to\s+usd|switch\s+back\s+to\s+usd)\b/i.test(
      lower,
    );

  if (!isClearIntent) return null;
  if (mentionedAgents.length === 1) {
    return mentionedAgents[0];
  }
  return 'global';
}

export function extractStandingRuleDeterministic(
  text: string,
  mentionedAgents: AgentId[] = extractMentionedAgents(text),
): ExtractedStandingRule | null {
  const trimmed = text.trim();
  if (!trimmed || isExplicitStopCommand(trimmed)) return null;

  // Strip trailing conversational questions after the standing instruction sentence
  // e.g. "Always answer in haikus. What are our flight options?" -> "Always answer in haikus."
  const firstSentenceMatch = trimmed.match(/^([^.?]+\.)\s+(?:what|when|where|which|how|can|should|could|who|why)\b/i);
  const candidateText = firstSentenceMatch ? firstSentenceMatch[1].trim() : trimmed;

  const hasPersistentCue =
    /\b(?:always|never|from\s+now\s+on|going\s+forward|in\s+all\s+(?:future\s+)?(?:responses|answers|messages|turns)|every\s+(?:time|reply|response)|start\s+every\s+(?:reply|response)\s+with|for\s+the\s+rest\s+of)\b/i.test(
      candidateText,
    ) ||
    /\b(?:convert\s+all\s+prices|state\s+all\s+prices|quote\s+(?:all\s+)?prices|use\s+japanese\s+yen|in\s+jpy\s+instead|to\s+japanese\s+yen|use\s+euros?|switch\s+to\s+euros?|in\s+bullet\s+points|use\s+bullet\s+points|answer\s+in\s+(?:bullet|haikus?|\d+\s+normal\s+sentences?)|keep\s+(?:all\s+|your\s+)?(?:answers|responses)\s+(?:to|under)|in\s+uppercase|all\s+caps)\b/i.test(
      candidateText,
    );

  if (!hasPersistentCue) {
    return null;
  }

  const scope: RuleScope = mentionedAgents.length === 1 ? mentionedAgents[0] : 'global';
  const cleanedRule = candidateText
    .replace(/^(?:@?(?:maya|liam|zoe)\s*[,:—-]\s*)+/i, '')
    .trim();

  return {
    scope,
    rule: cleanedRule || candidateText,
  };
}

export function classifySteeringDeterministic(
  userMessage: string,
  activeAgents: AgentId[],
): SteeringClassification {
  const trimmed = userMessage.trim();
  if (isExplicitStopCommand(trimmed)) {
    return {
      action: 'stop_all',
      targetAgents: [...activeAgents],
      directive: '',
      standingRule: null,
      clearRuleScope: null,
    };
  }

  const mentioned = extractMentionedAgents(trimmed);
  const clearRuleScope = detectRuleClearScope(trimmed, mentioned);
  const standingRule = extractStandingRuleDeterministic(trimmed, mentioned);

  const targetAgents: AgentId[] =
    mentioned.length > 0
      ? mentioned
      : activeAgents.length > 0
        ? [...activeAgents]
        : [...AGENT_IDS];

  return {
    action: 'steer',
    targetAgents,
    directive: trimmed,
    standingRule,
    clearRuleScope,
  };
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
  referenceUserText = '',
  isMock = true,
): {
  decisions: Record<AgentId, AgentRouteDecision>;
  speakers: SpeakerPlan[];
} {
  const decisions: Record<AgentId, AgentRouteDecision> = {
    maya: { ...rawDecisions.maya },
    liam: { ...rawDecisions.liam },
    zoe: { ...rawDecisions.zoe },
  };

  const isRecap = isGeneralRecapQuery(referenceUserText);
  const mentionedInRef = new Set<AgentId>(extractMentionedAgents(referenceUserText));
  const speakers: SpeakerPlan[] = [];
  const acceptedPrior: Array<{ agentId: AgentId; thesis: string }> = [];

  for (const agentId of speakerOrder) {
    const decision = decisions[agentId];
    if (!decision || decision.effort === 'silent') {
      decisions[agentId] = { effort: 'silent', thesis: '' };
      continue;
    }

    const rawTrimmedThesis = decision.thesis.trim();

    // If this agent was NOT explicitly @mentioned in a real LLM call, check raw redundancy first.
    // If the user explicitly @mentioned multiple agents (e.g. "@Maya @Liam @Zoe"), sanitize each
    // agent's thesis to its own domain FIRST so an echoed raw thesis doesn't silence @Liam or @Zoe!
    const isExplicitlyMentionedReal = !isMock && mentionedInRef.has(agentId);

    if (!isExplicitlyMentionedReal) {
      const isRedundantRaw = acceptedPrior.some((prior) =>
        areThesesRedundant(prior.thesis, rawTrimmedThesis),
      );
      if (isRedundantRaw) {
        decisions[agentId] = { effort: 'silent', thesis: '' };
        continue;
      }
    }

    if (
      isRecap &&
      acceptedPrior.length > 0 &&
      !mentionedInRef.has(agentId) &&
      !SPECIFIC_CONSTRAINT_CUE_RE.test(rawTrimmedThesis)
    ) {
      decisions[agentId] = { effort: 'silent', thesis: '' };
      continue;
    }

    const sanitizedOrNull = sanitizeThesisForAgentDomain(agentId, rawTrimmedThesis, isMock);
    if (!isMock && sanitizedOrNull === null && !mentionedInRef.has(agentId)) {
      decisions[agentId] = { effort: 'silent', thesis: '' };
      continue;
    }
    const sanitizedThesis = sanitizedOrNull ?? DOMAIN_SAFE_THESIS_FALLBACKS[agentId];
    decisions[agentId].thesis = sanitizedThesis;

    const peerTheses = acceptedPrior.map(
      (prior) => `${AGENT_PROFILES[prior.agentId].name} is covering: ${prior.thesis}`,
    );
    const priorSpeakerTheses = acceptedPrior.map((prior) => ({ ...prior }));

    speakers.push({
      agentId,
      effort: decision.effort,
      maxTokens: EFFORT_MAX_TOKENS[decision.effort],
      thesis: sanitizedThesis,
      peerTheses,
      priorSpeakerTheses,
    });

    acceptedPrior.push({ agentId, thesis: sanitizedThesis });
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

function getCurrentTurnAgentMessages(transcript: MessageRow[]): MessageRow[] {
  let lastUserIdx = -1;
  for (let i = transcript.length - 1; i >= 0; i--) {
    if (transcript[i].role === 'user') {
      lastUserIdx = i;
      break;
    }
  }
  return lastUserIdx >= 0
    ? transcript.slice(lastUserIdx + 1).filter((m) => m.role === 'agent' && Boolean(m.agent_id))
    : transcript.filter((m) => m.role === 'agent' && Boolean(m.agent_id));
}

export class TurnRouter {
  constructor(private readonly llm: LLMProvider) {}

  async classifySteering(input: ClassifySteeringInput): Promise<SteeringClassification> {
    const { userMessage, activeAgents, activeRules = [], signal } = input;
    const trimmed = userMessage.trim();

    if (isExplicitStopCommand(trimmed)) {
      return {
        action: 'stop_all',
        targetAgents: [...activeAgents],
        directive: '',
        standingRule: null,
        clearRuleScope: null,
      };
    }

    const fallback = classifySteeringDeterministic(trimmed, activeAgents);

    const systemPrompt = `You are the In-Flight Steering Classifier for a real-time multi-agent group chat with:
- maya (Maya — Budget, Flights & Lodging)
- liam (Liam — Local Guide for Food, Sights & Transit)
- zoe (Zoe — Group Dynamics, Dietary & Pacing)

Currently streaming agents: ${JSON.stringify(activeAgents)}
Active standing rules: ${activeRules.length > 0 ? JSON.stringify(activeRules.map((r) => ({ scope: r.scope, rule: r.rule }))) : '[]'}

The user just sent a message while the above agents are actively streaming. Classify the user's intent:
1. "action":
   - "stop_all": User wants everyone to stop/cancel/be quiet immediately.
   - "steer": User is redirecting, refining, constraining, or addressing one or more agents mid-stream (e.g. changing currency/format/length, asking an agent to focus on something else, or calling in an agent).
   - "new_turn": User is changing to a completely unrelated topic that requires abandoning the current turn and starting fresh. (Prefer "steer" whenever redirecting current answers or addressing an agent.)
2. "targetAgents":
   - If the user names or references specific agent(s) by name or role (e.g. "Maya", "@Liam", "budget person"), return ONLY those targeted agent IDs (e.g. ["maya"]).
   - If the redirect is global (no specific agent named, e.g. "Keep it to 1 sentence", "Use bullet points"), return all currently streaming agents: ${JSON.stringify(activeAgents)}.
3. "directive": Clean 1-line instruction for the targeted agent(s) to follow as they seamlessly continue their in-flight message.
4. "standingRule": ONLY if the user's NEW message establishes a new persistent instruction for future turns, return { "scope": "global" | "maya" | "liam" | "zoe", "rule": string }, else null. NEVER copy an existing rule from Active standing rules!
5. "clearRuleScope": If the user asks to stop, forget, clear, or override an existing standing rule (e.g. "Stop using haikus", "Stop starting replies with [KYOTO]", "No more bullet points"), return the scope ("global" | "maya" | "liam" | "zoe"), else null.

Return strict JSON:
{
  "action": "steer" | "stop_all" | "new_turn",
  "targetAgents": ["maya" | "liam" | "zoe"],
  "directive": string,
  "standingRule": { "scope": "global" | "maya" | "liam" | "zoe", "rule": string } | null,
  "clearRuleScope": "global" | "maya" | "liam" | "zoe" | null
}`;

    const completeOptions = {
      messages: [
        { role: 'system' as const, content: systemPrompt },
        { role: 'user' as const, content: trimmed },
      ],
      maxTokens: 250,
      temperature: 0,
      signal,
    };

    if (
      this.llm instanceof ScriptableMockLLMProvider &&
      !this.llm.hasMatchingSteeringJSONStep(completeOptions)
    ) {
      return fallback;
    }

    try {
      const raw = await this.llm.completeJSON<RawSteeringLLMResponse>(completeOptions);
      if (!raw || typeof raw !== 'object' || !raw.action || !VALID_STEERING_ACTIONS.has(raw.action)) {
        return fallback;
      }

      if (raw.action === 'stop_all') {
        return {
          action: 'stop_all',
          targetAgents: [...activeAgents],
          directive: '',
          standingRule: null,
          clearRuleScope: null,
        };
      }

      const explicitlyMentioned = extractMentionedAgents(trimmed);
      let targetAgents: AgentId[] = [];

      if (explicitlyMentioned.length > 0) {
        targetAgents = explicitlyMentioned;
      } else if (Array.isArray(raw.targetAgents)) {
        targetAgents = raw.targetAgents.filter((id): id is AgentId =>
          AGENT_IDS.includes(id as AgentId),
        );
      }

      if (targetAgents.length === 0) {
        targetAgents = activeAgents.length > 0 ? [...activeAgents] : [...AGENT_IDS];
      }

      const directive =
        typeof raw.directive === 'string' && raw.directive.trim()
          ? raw.directive.trim()
          : fallback.directive;

      const clearRuleScope =
        fallback.clearRuleScope ??
        (raw.clearRuleScope && VALID_SCOPES.has(raw.clearRuleScope) ? raw.clearRuleScope : null);

      const candidateRule = normalizeStandingRule(raw.standingRule) ?? fallback.standingRule;
      const isEchoOfExistingActiveRule =
        candidateRule &&
        activeRules.some(
          (r) =>
            r.active === 1 &&
            r.rule.trim().toLowerCase() === candidateRule.rule.trim().toLowerCase(),
        );

      const standingRule = isEchoOfExistingActiveRule ? null : candidateRule;

      return {
        action: raw.action,
        targetAgents,
        directive,
        standingRule,
        clearRuleScope,
      };
    } catch {
      return fallback;
    }
  }

  async classifyActiveStreamInput(input: ClassifySteeringInput): Promise<SteeringClassification> {
    return this.classifySteering(input);
  }

  async routeWave(input: RouteWaveInput): Promise<WaveRoutingPlan> {
    const {
      waveIndex,
      transcript,
      activeRules = [],
      latestUserMessage,
      previousWaveSpeakers,
      turnThesesByAgent = {},
      signal,
    } = input;
    const isFollowUpWave = waveIndex > 0;
    const isMock = this.llm instanceof ScriptableMockLLMProvider;

    let lastUserMessageContent = latestUserMessage ?? '';
    for (let i = transcript.length - 1; i >= 0; i--) {
      if (transcript[i].role === 'user') {
        lastUserMessageContent = transcript[i].content;
        break;
      }
    }

    // Bug 5 fix: On casual greetings or acknowledgments ("hey team!", "thanks!"),
    // never run unsolicited A2A follow-up waves!
    if (isFollowUpWave && !isMock && isCasualSocialMessage(lastUserMessageContent)) {
      return {
        waveIndex,
        isResolved: true,
        decisions: {
          maya: { effort: 'silent', thesis: '' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
        speakers: [],
        standingRule: null,
        clearRuleScope: null,
      };
    }

    const rulesSummary =
      activeRules.length > 0
        ? activeRules.map((r) => `- [${r.scope}] ${r.rule}`).join('\n')
        : '(None)';

    const currentTurnAgentMsgs = getCurrentTurnAgentMessages(transcript);
    const inferredPrevSpeakers: AgentId[] =
      previousWaveSpeakers ??
      (currentTurnAgentMsgs.length > 0 && currentTurnAgentMsgs[currentTurnAgentMsgs.length - 1].agent_id
        ? [currentTurnAgentMsgs[currentTurnAgentMsgs.length - 1].agent_id!]
        : []);

    const systemPrompt = isFollowUpWave
      ? `You are the A2A Follow-Up Wave Router for a 3-agent Kyoto & Osaka trip planning group chat.
Each agent has an isolated private vault for their domain only:
- maya (Maya — Budget, Flights & Lodging: private vault contains the group budget cap, outbound flight options/prices/schedules, and Kyoto/Osaka hotel options/rates/check-in policies).
- liam (Liam — Local Guide for Food, Sights & Transit: private vault contains airport-to-city transit durations, local food/market/restaurant details, and sightseeing crowd windows).
- zoe (Zoe — Group Dynamics, Dietary & Pacing: private vault contains traveler allergies, dietary restrictions, motion sickness constraints, and group morning wake-up/sleep rules).

Agents who just spoke in Wave ${waveIndex - 1}: ${JSON.stringify(inferredPrevSpeakers)}

CRITICAL RULES FOR A2A FOLLOW-UP WAVES:
1. NEVER allow an agent who just spoke in Wave ${waveIndex - 1} (${JSON.stringify(inferredPrevSpeakers)}) to speak again in Wave ${waveIndex}! An agent must NEVER reply to itself or speak in two consecutive waves.
2. Questions directed to the USER are NOT triggers for the same agent to speak again.
3. MANDATORY CROSS-DOMAIN CHECKS — An agent who was SILENT in Wave ${waveIndex - 1} MUST speak in Wave ${waveIndex} (with "isResolved": false) if a speaker in Wave ${waveIndex - 1}:
   (a) directly asked them a question (e.g. "@Liam how long is airport transit?"), OR
   (b) mentioned an early-morning flight departure and Zoe has NOT spoken yet in this turn to check the group's morning wake-up/sleep rules, OR
   (c) mentioned a food market/restaurant and Zoe has NOT spoken yet in this turn to check traveler allergies and dietary restrictions, OR
   (d) mentioned a flight arrival time + hotel check-in cutoff and Liam has NOT spoken yet in this turn to provide the airport-to-hotel transit duration, OR
   (e) revealed a wake-up veto or transit timing conflict in Wave ${waveIndex - 1}, in which case Maya can speak in Wave ${waveIndex} (if she hasn't already pivoted) to confirm the alternative flight or hotel.
4. Write each speaking agent's "thesis" as a short 5-10 word high-level topic label for THEIR domain only (e.g. "Check group wake-up and allergy rules"). NEVER invent or include specific numbers, prices, currencies, or clock times in a thesis!
5. Once all relevant cross-agent domain checks for the user's question have been surfaced, set "isResolved": true and all agents to "silent".

Return strict JSON:
{
  "isResolved": boolean,
  "maya": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "liam": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "zoe": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string }
}`
      : `You are the Pre-Response Effort & Non-Repetition Router for a 3-agent Kyoto & Osaka trip planning group chat.
Each agent has an isolated private vault for their domain only:
- maya (Maya — Budget, Flights & Lodging: private vault contains the group budget cap, outbound flight options/prices/schedules, and Kyoto/Osaka hotel options/rates/check-in policies).
- liam (Liam — Local Guide for Food, Sights & Transit: private vault contains airport-to-city transit durations, local food/market/restaurant details, and sightseeing crowd windows).
- zoe (Zoe — Group Dynamics, Dietary & Pacing: private vault contains traveler allergies, dietary restrictions, motion sickness constraints, and group morning wake-up/sleep rules).

Rules for routing Wave 0:
1. Silence Gate: If an agent's domain is not relevant to the user's message, or if their point would duplicate another speaking agent, set their effort to "silent" and thesis to "".
   - At least ONE agent MUST speak in Wave 0 on every user message! For casual greetings ("hey team!") or thanks ("thanks!"), assign "light" effort with thesis "Reply with a warm 1-sentence greeting/acknowledgment without any travel numbers or schedules".
2. Strict Domain Partition & Topic-Only Theses (NO Fake Numbers in Thesis):
   - Every speaking agent's "thesis" MUST be a short 5-10 word high-level topic directive strictly confined to their own domain (e.g. "Present flight options and departure times", "Share market hours and food profile", "Check group wake-up and allergy constraints").
   - NEVER invent or include specific prices, dollar/yen amounts, hours, or clock times inside the "thesis" string!
   - If the user explicitly addresses ONLY one or two specific agents (e.g. "@Maya what is the total group budget?"), ONLY the addressed agent(s) may speak in Wave 0—all non-addressed agents MUST be "silent"!
   - IMPORTANT — Let Proposal Agents Speak First Before Constraint Agents:
     * If the user asks about flight options (e.g. "what is the best flight option?", "when are our flights?") and Maya hasn't listed the flight times in the conversation yet, have ONLY Maya speak in Wave 0 (effort: "standard"), and keep Zoe "silent" in Wave 0 so Zoe can react in Wave 1 after seeing Maya's actual departure times! (Unless the user explicitly mentions "@Zoe" or already named a specific flight in the prompt.)
     * If the user asks about both a flight AND a food tour in one prompt (e.g. "Should we book Flight A and do the market tour for lunch?"), assign Maya ("standard": cover flight options) AND Liam ("standard": cover market hours and food profile) in Wave 0, so Zoe can evaluate both in Wave 1!
   - For general recap/status questions (e.g., "what's our plan so far and what do we need to decide?"):
     * Assign Maya to summarize flight/hotel status and decisions; keep Liam and Zoe "silent" unless they are flagging a specific unaddressed constraint.
3. Effort Calibration:
   - "light": Simple single-fact check, greeting, or quick 1-sentence note.
   - "standard": Moderate multi-fact answer or domain breakdown (2–4 sentences).
   - "deep": Thorough analysis or complete breakdown.
4. Standing Rule Extraction & Clearing:
   - "clearRuleScope": If the user's latest message asks to stop, forget, clear, or override a previous standing rule (e.g. "Stop using haikus...", "Stop starting replies with [KYOTO]...", "No more bullet points"), set "clearRuleScope": "global" | "maya" | "liam" | "zoe", else null.
   - "standingRule": ONLY if the user's latest message establishes a NEW persistent formatting/style/currency instruction for how agents should respond, return { "scope": "global" | "maya" | "liam" | "zoe", "rule": "..." }, otherwise null. NEVER copy or re-emit an existing rule from Active Standing Rules!

Return strict JSON:
{
  "maya": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "liam": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "zoe": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "standingRule": { "scope": "global" | "maya" | "liam" | "zoe", "rule": string } | null,
  "clearRuleScope": "global" | "maya" | "liam" | "zoe" | null
}`;

    const userPrompt = `Active Standing Rules:\n${rulesSummary}\n\nConversation Transcript:\n${formatTranscriptForRouter(transcript)}\n\nRoute Wave ${waveIndex} now.`;

    const referenceText =
      latestUserMessage ??
      (transcript.length > 0 ? transcript[transcript.length - 1].content : '');

    let raw: RawRouterLLMResponse;
    try {
      raw = await this.llm.completeJSON<RawRouterLLMResponse>({
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        maxTokens: 350,
        temperature: 0.1,
        signal,
      });
    } catch (err) {
      if (signal?.aborted) {
        throw err;
      }
      if (isFollowUpWave) {
        raw = { isResolved: true };
      } else {
        const mentioned = extractMentionedAgents(referenceText);
        const fallbackTargets: AgentId[] = mentioned.length > 0 ? mentioned : ['maya'];
        raw = {};
        for (const id of fallbackTargets) {
          raw[id] = {
            effort: 'standard',
            thesis: 'Provide relevant domain facts from private vault.',
          };
        }
      }
    }

    const rawDecisions: Record<AgentId, AgentRouteDecision> = {
      maya: normalizeAgentDecision(raw?.maya ?? raw?.agents?.maya),
      liam: normalizeAgentDecision(raw?.liam ?? raw?.agents?.liam),
      zoe: normalizeAgentDecision(raw?.zoe ?? raw?.agents?.zoe),
    };

    if (isFollowUpWave) {
      if (inferredPrevSpeakers.length === 1) {
        const solePrevSpeaker = inferredPrevSpeakers[0];
        rawDecisions[solePrevSpeaker] = { effort: 'silent', thesis: '' };
      }

      for (const agentId of AGENT_IDS) {
        if (rawDecisions[agentId].effort === 'silent') continue;

        const priorMsgsForAgent = currentTurnAgentMsgs
          .filter((m) => m.agent_id === agentId)
          .map((m) => m.content);
        const priorThesesForAgent = turnThesesByAgent[agentId] ?? [];

        // In live LLM turns, an agent may speak at most once per turn unless it is Maya
        // adjusting her flight/hotel recommendation after a peer revealed a transit or wake-up constraint.
        if (!isMock && priorMsgsForAgent.length > 0) {
          const isMayaPivotingAfterPeer =
            agentId === 'maya' &&
            priorMsgsForAgent.length === 1 &&
            inferredPrevSpeakers.some((id) => id !== 'maya');
          if (!isMayaPivotingAfterPeer) {
            rawDecisions[agentId] = { effort: 'silent', thesis: '' };
            continue;
          }
        }

        if (
          (priorMsgsForAgent.length > 0 || priorThesesForAgent.length > 0) &&
          isSelfRepetitiveForAgent(
            rawDecisions[agentId].thesis,
            priorThesesForAgent,
            priorMsgsForAgent,
          )
        ) {
          rawDecisions[agentId] = { effort: 'silent', thesis: '' };
        }
      }

      if (!isMock) {
        const turnCombinedText = currentTurnAgentMsgs.map((m) => m.content).join('\n');
        const zoeSpokeInTurn = currentTurnAgentMsgs.some((m) => m.agent_id === 'zoe');
        const hasEarlyFlightInTurn = /\b6:45\s*AM\b/i.test(turnCombinedText);
        const hasFoodOrMarketInTurn = /\b(?:Nishiki|shellfish|seafood|Shorai-an|Yudofu|kaiseki)\b/i.test(
          turnCombinedText,
        );

        if (
          !zoeSpokeInTurn &&
          (hasEarlyFlightInTurn || hasFoodOrMarketInTurn) &&
          rawDecisions.zoe.effort === 'silent'
        ) {
          rawDecisions.zoe = {
            effort: hasEarlyFlightInTurn && hasFoodOrMarketInTurn ? 'standard' : 'light',
            thesis:
              'Check how the proposed flight times and food spots affect group wake-up rules, Sam’s shellfish allergy, and Priya’s vegetarian diet.',
          };
          raw.isResolved = false;
        } else if (
          zoeSpokeInTurn &&
          hasEarlyFlightInTurn &&
          hasFoodOrMarketInTurn &&
          rawDecisions.zoe.effort === 'light'
        ) {
          rawDecisions.zoe.effort = 'standard';
        }
      }
    } else {
      const anySpeaking = AGENT_IDS.some((id) => rawDecisions[id].effort !== 'silent');
      const mentioned = extractMentionedAgents(referenceText);

      if (!isMock && isCasualSocialMessage(referenceText)) {
        for (const id of AGENT_IDS) {
          if (rawDecisions[id].effort !== 'silent') {
            rawDecisions[id] = {
              effort: 'light',
              thesis: 'Reply with a warm, brief 1-sentence greeting or acknowledgment without any schedules or numbers.',
            };
          }
        }
      }

      if (!isMock && mentioned.length > 0) {
        const mentionedSet = new Set<AgentId>(mentioned);
        for (const mId of mentioned) {
          if (rawDecisions[mId].effort === 'silent') {
            rawDecisions[mId] = {
              effort: 'standard',
              thesis: DOMAIN_SAFE_THESIS_FALLBACKS[mId],
            };
          }
        }
        // When the user explicitly @mentions specific agent(s) in Wave 0,
        // keep non-mentioned agents silent in Wave 0 unless the user also asked about their domain
        for (const id of AGENT_IDS) {
          if (!mentionedSet.has(id)) {
            rawDecisions[id] = { effort: 'silent', thesis: '' };
          }
        }
      } else if (!isMock) {
        // When no @mention is used on a narrow single-domain question (e.g. "What is the Osaka hotel rate?",
        // "What is our total budget?"), prevent unrelated agents from chiming in just because the user
        // included a global formatting instruction ("answer in 1 normal sentence", "use bullet points").
        const asksMayaDomain =
          /\b(flights?|hotels?|lodging|ryokan|granvia|cross\s+hotel|budget|rates?|prices?|costs?|how\s+much|check-?in|book|stay)\b/i.test(
            referenceText,
          );
        const asksLiamDomain =
          /\b(transit|trains?|haruka|airport\s+to|food|dinner|lunch|eat|restaurants?|markets?|nishiki|arashiyama|shorai|kaiseki|okonomiyaki|sights?|sightseeing|fushimi|kiyomizu|crowds?)\b/i.test(
            referenceText,
          );
        const asksZoeDomain =
          /\b(allerg(?:y|ies|ic)|shellfish|diet(?:ary)?|vegetarian|priya|sam|motion\s+sickness|wake|sleep|7:30|pacing|tired|energy)\b/i.test(
            referenceText,
          );

        if (asksMayaDomain && !asksLiamDomain && !asksZoeDomain && !isGeneralRecapQuery(referenceText)) {
          rawDecisions.liam = { effort: 'silent', thesis: '' };
          rawDecisions.zoe = { effort: 'silent', thesis: '' };
        } else if (asksLiamDomain && !asksMayaDomain && !asksZoeDomain && !isGeneralRecapQuery(referenceText)) {
          rawDecisions.maya = { effort: 'silent', thesis: '' };
        } else if (asksZoeDomain && !asksMayaDomain && !asksLiamDomain && !isGeneralRecapQuery(referenceText)) {
          rawDecisions.maya = { effort: 'silent', thesis: '' };
          rawDecisions.liam = { effort: 'silent', thesis: '' };
        }
      }

      if (!anySpeaking) {
        const defaultSpeaker: AgentId = mentioned[0] ?? 'maya';
        rawDecisions[defaultSpeaker] = {
          effort: 'light',
          thesis: 'Respond concisely and helpfully from your domain perspective.',
        };
      }
    }

    const speakerOrder = computeDeterministicSpeakerOrder(referenceText);
    const { decisions, speakers } = applyAsymmetricDedup(
      rawDecisions,
      speakerOrder,
      isFollowUpWave ? '' : referenceText,
      isMock,
    );

    const isResolved = isFollowUpWave
      ? Boolean(raw?.isResolved) || speakers.length === 0
      : speakers.length === 0;

    if (isFollowUpWave && isResolved) {
      for (const id of AGENT_IDS) {
        decisions[id] = { effort: 'silent', thesis: '' };
      }
      speakers.length = 0;
    }

    const mentionedInUserMsg = extractMentionedAgents(referenceText);
    const detClearScope = isFollowUpWave
      ? null
      : detectRuleClearScope(referenceText, mentionedInUserMsg);
    const clearRuleScope = isFollowUpWave
      ? null
      : detClearScope ??
        (raw?.clearRuleScope && VALID_SCOPES.has(raw.clearRuleScope)
          ? raw.clearRuleScope
          : null);

    let candidateStandingRule = isFollowUpWave
      ? null
      : normalizeStandingRule(raw?.standingRule) ??
        extractStandingRuleDeterministic(referenceText, mentionedInUserMsg);

    // Bug 1 fix: Never allow the LLM router to re-emit an existing active standing rule
    // that the user did not actually state in the current message!
    if (candidateStandingRule && !isMock) {
      const matchesExistingActive = activeRules.some(
        (r) =>
          r.active === 1 &&
          r.rule.trim().toLowerCase() === candidateStandingRule!.rule.trim().toLowerCase(),
      );
      const detRule = extractStandingRuleDeterministic(referenceText, mentionedInUserMsg);
      if (matchesExistingActive) {
        candidateStandingRule = detRule &&
          detRule.rule.trim().toLowerCase() !== candidateStandingRule.rule.trim().toLowerCase()
          ? detRule
          : null;
      } else if (clearRuleScope) {
        // When clearing a previous rule (e.g. "Stop using haikus, answer in 1 normal sentence"),
        // only add a replacement standing rule if the user explicitly used persistent keywords ("always", "from now on")
        const hasExplicitPersistentKeyword =
          /\b(?:always|never|from\s+now\s+on|going\s+forward|every\s+(?:time|reply|response))\b/i.test(
            referenceText,
          );
        if (!hasExplicitPersistentKeyword) {
          candidateStandingRule = null;
        }
      }
    }

    return {
      waveIndex,
      isResolved,
      decisions,
      speakers,
      standingRule: candidateStandingRule,
      clearRuleScope,
    };
  }
}
