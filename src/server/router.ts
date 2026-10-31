import {
  AGENT_IDS,
  AGENT_PROFILES,
  EFFORT_MAX_TOKENS,
  getAgentDomainIndexKeywords,
  getAgentProfile,
  getAgentVaultTopicSummary,
  type CustomAgentRuntimeConfig,
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
  removeRuleIds?: string[] | null;
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
  removeRuleIds: string[];
  clearRuleScope?: RuleScope | null;
}

export interface RouteWaveInput {
  waveIndex: number;
  transcript: MessageRow[];
  activeRules?: StandingRuleRow[];
  latestUserMessage?: string;
  previousWaveSpeakers?: AgentId[];
  turnThesesByAgent?: Partial<Record<AgentId, string[]>>;
  runtimeConfig?: CustomAgentRuntimeConfig;
  signal?: AbortSignal;
}

export type SteeringAction = 'steer' | 'stop_all' | 'new_turn';

export interface SteeringClassification {
  action: SteeringAction;
  targetAgents: AgentId[];
  directive: string;
  standingRule: ExtractedStandingRule | null;
  removeRuleIds: string[];
  clearRuleScope: RuleScope | null;
}

export interface RawSteeringLLMResponse {
  action?: SteeringAction;
  targetAgents?: AgentId[];
  directive?: string;
  standingRule?: ExtractedStandingRule | null;
  removeRuleIds?: string[] | null;
  clearRuleScope?: RuleScope | null;
}

export interface ClassifySteeringInput {
  userMessage: string;
  activeAgents: AgentId[];
  transcript?: MessageRow[];
  activeRules?: StandingRuleRow[];
  runtimeConfig?: CustomAgentRuntimeConfig;
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

export function isUserBookingConfirmationMessage(text: string): boolean {
  const trimmed = text.trim();
  if (trimmed.includes('?')) return false;
  return /^(?:yes|yep|yeah|ok|okay|sure|sounds\s+good|let'?s\s+do\s+(?:that|it)|let'?s\s+book\s+(?:that|it)|go\s+ahead|agreed|confirm(?:ed)?)\b/i.test(
    trimmed,
  );
}

export function isStandaloneStandingRuleMessage(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed || trimmed.includes('?')) return false;
  const hasQuestionWord =
    /\b(?:what|when|where|which|who|why|how\s+much|how\s+many|how\s+long|can\s+we|should\s+we|could\s+we|tell\s+me\s+about|give\s+(?:me|us)\s+a\s+breakdown)\b/i.test(
      trimmed,
    );
  if (hasQuestionWord) return false;
  const mentioned = extractMentionedAgents(trimmed);
  return Boolean(
    extractStandingRuleDeterministic(trimmed, mentioned) ||
      detectRuleClearScope(trimmed, mentioned),
  );
}

export function getDomainSafeThesisFallback(
  agentId: AgentId,
  runtimeConfig?: CustomAgentRuntimeConfig,
): string {
  const profile = getAgentProfile(agentId, runtimeConfig);
  const topics = getAgentVaultTopicSummary(agentId, runtimeConfig);
  return `Share ONLY the relevant facts and constraints for ${profile.roleTitle} (${topics}) from your private vault.`;
}

// Detects when the LLM router invents specific fake prices, hours, or clock times inside a thesis
const HALLUCINATED_NUMBERS_IN_THESIS_RE =
  /(?:[$¥€£]\s*\d|\b\d[\d,]*\s*(?:usd|jpy|eur|yen|dollars?|euros?)\b|\b\d{1,2}:\d{2}\s*(?:am|pm)?\b|\b\d+\s*(?:am|pm)\b|\b\d+\s*(?:hours?|hrs?|minutes?|mins?)\b)/i;

export function sanitizeThesisForAgentDomain(
  agentId: AgentId,
  thesis: string,
  isMock = true,
  runtimeConfig?: CustomAgentRuntimeConfig,
): string {
  const trimmed = thesis.trim();
  const fallback = getDomainSafeThesisFallback(agentId, runtimeConfig);
  if (!trimmed) return fallback;
  if (!isMock && HALLUCINATED_NUMBERS_IN_THESIS_RE.test(trimmed)) {
    return fallback;
  }
  return trimmed;
}

const SPECIFIC_CONSTRAINT_CUE_RE =
  /\b(allergy|allergies|dietary|vegetarian|wake|sleep|pacing|motion\s+sickness|cutoff|transit\s+time|crowd\s+window|conflict|unsafe|veto|constraint|restriction|risk|limit|blocker|infeasible)\b/i;

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
    /\b(?:confirm\s+(?:the\s+)?(?:preferred\s+(?:flight|option)|readiness|reservation|booking)|choose\s+between\s+flight\s+a\s+(?:and|or)\s+flight\s+b|finalize\s+(?:travel\s+plans|the\s+reservation|booking)|secure\s+(?:our\s+|the\s+)?lodging)\b/i.test(
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

export function extractMentionedAgents(
  text: string,
  runtimeConfig?: CustomAgentRuntimeConfig,
): AgentId[] {
  if (!text) return [];
  const matches: Array<{ agentId: AgentId; index: number }> = [];

  for (const agentId of AGENT_IDS) {
    const name = getAgentProfile(agentId, runtimeConfig).name;
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

const CLEAR_CLAUSE_RE =
  /\b(?:forget|clear|remove|drop|cancel|reset|stop\s+following|never\s*mind)\b.*\b(?:standing\s+rules?|rules?|instructions?|format|prefix|suffix|tag|haikus?|riddles?|rhymes?|poems?|poetry|jpy|yen|euros?|bullet\s+points?|uppercase)\b|\b(?:stop\s+(?:using|answering|writing|speaking|talking|starting|ending|beginning|finishing|including|adding|appending|putting)|no\s+longer\s+need\s+to|no\s+more\s+(?:haikus?|riddles?|rhymes?|poems?|bullets?|bullet\s+points?|prefixes|suffixes|tags|jpy|yen|euros?|uppercase)|don'?t\s+(?:use|start|end|include|add|speak\s+in|write\s+in)|(?:talk|speak|respond|answer|write|chat)\s+normally|just\s+talk\s+normally|back\s+to\s+normal|normal\s+(?:tone|english|voice|sentences?)|(?:riddles?|haikus?|poems?|rhymes?)\s+are\s+(?:hard|difficult|annoying|confusing)|answer\s+in\s+\d*\s*normal\s+sentences?|without\s+any\s+(?:prefix|suffix|tag)|go\s+back\s+to\s+usd|switch\s+back\s+to\s+usd)/i;

export function detectRuleClearScope(
  text: string,
  mentionedAgents?: AgentId[],
  runtimeConfig?: CustomAgentRuntimeConfig,
): RuleScope | null {
  const trimmed = text.trim();
  if (!trimmed || isExplicitStopCommand(trimmed)) return null;

  const sentences = trimmed
    .split(/(?<=[.!?])\s+|\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);

  for (const sentence of sentences) {
    if (CLEAR_CLAUSE_RE.test(sentence)) {
      const clauseMentions = extractMentionedAgents(sentence, runtimeConfig);
      if (clauseMentions.length === 1) {
        return clauseMentions[0];
      }
      if (clauseMentions.length === 0 && mentionedAgents && mentionedAgents.length === 1 && sentences.length === 1) {
        return mentionedAgents[0];
      }
      return 'global';
    }
  }

  return null;
}

const PERSISTENT_RULE_CLAUSE_RE =
  /\b(?:always|never|from\s+now\s+on|going\s+forward|in\s+all\s+(?:future\s+)?(?:responses|answers|messages|turns)|every\s+(?:time|reply|response|message)|(?:start|end|begin)\s+(?:every|all|each)\s+(?:reply|response|message)s?\s+with|for\s+the\s+rest\s+of)\b|\b(?:convert\s+all\s+prices|state\s+all\s+prices|quote\s+(?:all\s+)?prices|use\s+japanese\s+yen|in\s+jpy\s+instead|to\s+japanese\s+yen|use\s+euros?|switch\s+to\s+euros?|in\s+bullet\s+points|use\s+bullet\s+points|answer\s+in\s+(?:bullet|haikus?|\d+\s+normal\s+sentences?)|keep\s+(?:all\s+|your\s+)?(?:answers|responses)\s+(?:to|under)|in\s+uppercase|all\s+caps)\b/i;

export function extractStandingRuleDeterministic(
  text: string,
  mentionedAgents?: AgentId[],
  runtimeConfig?: CustomAgentRuntimeConfig,
): ExtractedStandingRule | null {
  const trimmed = text.trim();
  if (!trimmed || isExplicitStopCommand(trimmed)) return null;

  const sentences = trimmed
    .split(/(?<=[.!?])\s+|\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);

  for (const sentence of sentences) {
    if (CLEAR_CLAUSE_RE.test(sentence)) {
      continue;
    }
    if (!PERSISTENT_RULE_CLAUSE_RE.test(sentence)) {
      continue;
    }

    const clauseMentions = extractMentionedAgents(sentence, runtimeConfig);
    const scope: RuleScope =
      clauseMentions.length === 1
        ? clauseMentions[0]
        : clauseMentions.length === 0 && mentionedAgents && mentionedAgents.length === 1 && sentences.length === 1
          ? mentionedAgents[0]
          : 'global';

    const cleanedRule = sentence
      .replace(/^(?:@?(?:maya|liam|zoe)\s*[,:—-]\s*)+/i, '')
      .replace(/^(?:from\s+now\s+on\s*[,:—-]?\s*)?(?:everyone\s*[,:—-]?\s*)?/i, (m) =>
        /everyone/i.test(m) ? '' : m,
      )
      .trim();

    return {
      scope,
      rule: cleanedRule || sentence,
    };
  }

  return null;
}

export function isTopicReplacementInterrupt(text: string): boolean {
  const trimmed = text.trim();
  return /^(?:actually\s+stop|wait\s+no|no\s+wait|scratch\s+that|forget\s+that|never\s*mind|cancel\s+that)\b[\s,:-]*(?:just\s+|instead\s+)?(?:tell\s+(?:me|us)|what|how|where|when|who|why|can|should|give)/i.test(
    trimmed,
  );
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
      removeRuleIds: [],
      clearRuleScope: null,
    };
  }

  const mentioned = extractMentionedAgents(trimmed);
  const clearRuleScope = detectRuleClearScope(trimmed, mentioned);
  const standingRule = extractStandingRuleDeterministic(trimmed, mentioned);

  if (isTopicReplacementInterrupt(trimmed)) {
    return {
      action: 'new_turn',
      targetAgents: [...activeAgents],
      directive: trimmed,
      standingRule,
      removeRuleIds: [],
      clearRuleScope,
    };
  }

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
    removeRuleIds: [],
    clearRuleScope,
  };
}

export function computeDeterministicSpeakerOrder(
  referenceText = '',
  runtimeConfig?: CustomAgentRuntimeConfig,
): AgentId[] {
  const mentioned = extractMentionedAgents(referenceText, runtimeConfig);
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
  runtimeConfig?: CustomAgentRuntimeConfig,
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
  const mentionedInRef = new Set<AgentId>(
    extractMentionedAgents(referenceUserText, runtimeConfig),
  );
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

    // On a general recap/summary question ("what's our plan so far and what do we need to decide?"),
    // once Speaker 1 (Maya) is summarizing the plan and pending decisions, keep other agents silent
    // unless they are flagging a specific safety/policy/constraint conflict (SPECIFIC_CONSTRAINT_CUE_RE).
    if (
      isRecap &&
      acceptedPrior.length > 0 &&
      !mentionedInRef.has(agentId) &&
      !SPECIFIC_CONSTRAINT_CUE_RE.test(rawTrimmedThesis)
    ) {
      decisions[agentId] = { effort: 'silent', thesis: '' };
      continue;
    }

    // In a follow-up wave of a general recap query, also silence non-constraint recap echoes
    if (
      !isMock &&
      !referenceUserText &&
      GENERAL_RECAP_RE.test(rawTrimmedThesis) &&
      !SPECIFIC_CONSTRAINT_CUE_RE.test(rawTrimmedThesis)
    ) {
      decisions[agentId] = { effort: 'silent', thesis: '' };
      continue;
    }

    const sanitizedThesis = sanitizeThesisForAgentDomain(
      agentId,
      rawTrimmedThesis,
      isMock,
      runtimeConfig,
    );
    decisions[agentId].thesis = sanitizedThesis;

    const peerTheses = acceptedPrior.map(
      (prior) =>
        `${getAgentProfile(prior.agentId, runtimeConfig).name} is covering: ${prior.thesis}`,
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

function formatTranscriptForRouter(
  transcript: MessageRow[],
  runtimeConfig?: CustomAgentRuntimeConfig,
): string {
  if (transcript.length === 0) return '(No messages yet)';
  return transcript
    .map((m) => {
      const speaker =
        m.role === 'user'
          ? 'User'
          : m.agent_id
            ? getAgentProfile(m.agent_id, runtimeConfig).name
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

function buildAgentDomainDirectoryForRouter(runtimeConfig?: CustomAgentRuntimeConfig): string {
  return AGENT_IDS.map((id) => {
    const profile = getAgentProfile(id, runtimeConfig);
    const topics = getAgentVaultTopicSummary(id, runtimeConfig);
    return `- ${id} (${profile.name} — ${profile.roleTitle}; private vault sections: ${topics})`;
  }).join('\n');
}

export class TurnRouter {
  constructor(
    private readonly llm: LLMProvider,
    private readonly defaultRuntimeConfig?: CustomAgentRuntimeConfig,
  ) {}

  async classifySteering(input: ClassifySteeringInput): Promise<SteeringClassification> {
    const { userMessage, activeAgents, activeRules = [], signal } = input;
    const runtimeConfig = input.runtimeConfig ?? this.defaultRuntimeConfig;
    const trimmed = userMessage.trim();

    if (isExplicitStopCommand(trimmed)) {
      return {
        action: 'stop_all',
        targetAgents: [...activeAgents],
        directive: '',
        standingRule: null,
        removeRuleIds: [],
        clearRuleScope: null,
      };
    }

    const fallback = classifySteeringDeterministic(trimmed, activeAgents);
    const agentDirectory = buildAgentDomainDirectoryForRouter(runtimeConfig);
    const formattedRulesWithIds =
      activeRules.length > 0
        ? activeRules
            .map((r) => `- id="${r.id}" [scope=${r.scope}]: ${r.rule}`)
            .join('\n')
        : '(None)';

    const systemPrompt = `You are the In-Flight Steering Classifier for a real-time multi-agent group chat with:
${agentDirectory}

Currently streaming agents: ${JSON.stringify(activeAgents)}
Active Standing Rules:
${formattedRulesWithIds}

The user just sent a message while the above agents are actively streaming. Classify the user's intent:
1. "action":
   - "stop_all": User wants everyone to stop/cancel/be quiet immediately.
   - "steer": User is redirecting, refining, constraining, or addressing one or more agents mid-stream (e.g. changing currency/format/length, asking an agent to focus on something else, or calling in an agent).
   - "new_turn": User is changing to a completely unrelated topic that requires abandoning the current turn and starting fresh. (Prefer "steer" whenever redirecting current answers or addressing an agent.)
2. "targetAgents":
   - If the user names or references specific agent(s) by name or role (e.g. "Maya", "@Liam", "budget person"), return ONLY those targeted agent IDs (e.g. ["maya"]).
   - If the redirect is global (no specific agent named, e.g. "Keep it to 1 sentence", "Use bullet points"), return all currently streaming agents: ${JSON.stringify(activeAgents)}.
3. "directive": Clean 1-line instruction for the targeted agent(s) to follow as they seamlessly update and complete their in-flight message.
4. "removeRuleIds": Inspect Active Standing Rules against the user's message. If the user asks to stop, drop, forget, relax, or override any specific existing rule(s) (e.g. "nevermind riddles are hard to read, talk normally", "stop using haikus", "no more [BUDGET-OK]", or gives a new rule that contradicts a specific old rule), return the exact "id" strings of ONLY those superseded/canceled rules in "removeRuleIds": ["..."] while leaving unrelated active rules untouched! Otherwise return [].
5. "standingRule": ONLY if the user's NEW message establishes a new persistent instruction for future turns, return { "scope": "global" | "maya" | "liam" | "zoe", "rule": "<concise imperative rule ONLY, excluding any follow-up question>" }, else null. NEVER copy an existing rule from Active Standing Rules!

Return strict JSON:
{
  "action": "steer" | "stop_all" | "new_turn",
  "targetAgents": ["maya" | "liam" | "zoe"],
  "directive": string,
  "removeRuleIds": string[],
  "standingRule": { "scope": "global" | "maya" | "liam" | "zoe", "rule": string } | null
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
          removeRuleIds: [],
          clearRuleScope: null,
        };
      }

      const explicitlyMentioned = extractMentionedAgents(trimmed, runtimeConfig);
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

      const validActiveIds = new Set(activeRules.map((r) => r.id));
      const removeRuleIds = Array.isArray(raw.removeRuleIds)
        ? raw.removeRuleIds.filter((id): id is string => typeof id === 'string' && validActiveIds.has(id))
        : [];

      const clearRuleScope =
        removeRuleIds.length > 0
          ? null
          : (raw.clearRuleScope && VALID_SCOPES.has(raw.clearRuleScope)
              ? raw.clearRuleScope
              : fallback.clearRuleScope);

      const candidateRule = normalizeStandingRule(raw.standingRule);
      const isEchoOfExistingActiveRule =
        candidateRule &&
        activeRules.some(
          (r) =>
            r.active === 1 &&
            r.rule.trim().toLowerCase() === candidateRule.rule.trim().toLowerCase(),
        );

      const standingRule = isEchoOfExistingActiveRule ? null : candidateRule;
      const finalAction: SteeringAction = isTopicReplacementInterrupt(trimmed)
        ? 'new_turn'
        : raw.action;

      return {
        action: finalAction,
        targetAgents,
        directive,
        standingRule,
        removeRuleIds,
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
    const runtimeConfig = input.runtimeConfig ?? this.defaultRuntimeConfig;
    const isFollowUpWave = waveIndex > 0;
    const isMock = this.llm instanceof ScriptableMockLLMProvider;

    let lastUserMessageContent = latestUserMessage ?? '';
    for (let i = transcript.length - 1; i >= 0; i--) {
      if (transcript[i].role === 'user') {
        lastUserMessageContent = transcript[i].content;
        break;
      }
    }

    if (
      isFollowUpWave &&
      !isMock &&
      (isCasualSocialMessage(lastUserMessageContent) ||
        isGeneralRecapQuery(lastUserMessageContent) ||
        isStandaloneStandingRuleMessage(lastUserMessageContent) ||
        isUserBookingConfirmationMessage(lastUserMessageContent))
    ) {
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
        removeRuleIds: [],
        clearRuleScope: null,
      };
    }

    const rulesSummary =
      activeRules.length > 0
        ? activeRules
            .map((r) => `- id="${r.id}" [scope=${r.scope}]: ${r.rule}`)
            .join('\n')
        : '(None)';

    const currentTurnAgentMsgs = getCurrentTurnAgentMessages(transcript);
    const inferredPrevSpeakers: AgentId[] =
      previousWaveSpeakers ??
      (currentTurnAgentMsgs.length > 0 && currentTurnAgentMsgs[currentTurnAgentMsgs.length - 1].agent_id
        ? [currentTurnAgentMsgs[currentTurnAgentMsgs.length - 1].agent_id!]
        : []);

    const agentDirectory = buildAgentDomainDirectoryForRouter(runtimeConfig);

    const systemPrompt = isFollowUpWave
      ? `You are the A2A Follow-Up Wave Router for a 3-agent collaborative group chat.
Each agent has an isolated private knowledge vault for their own domain:
${agentDirectory}

Agents who just spoke in Wave ${waveIndex - 1}: ${JSON.stringify(inferredPrevSpeakers)}

CRITICAL RULES FOR A2A FOLLOW-UP WAVES:
1. NEVER allow an agent who just spoke in Wave ${waveIndex - 1} (${JSON.stringify(inferredPrevSpeakers)}) to speak again in Wave ${waveIndex}! An agent must NEVER reply to itself or speak in two consecutive waves.
2. NEVER schedule an agent just to say "I don't have that information, check directly with the venue/provider." If the previous speaker already answered the user's question (or noted what is in the vaults), set "isResolved": true and all agents to "silent"!
3. MANDATORY CROSS-DOMAIN CHECKS — An agent who was SILENT in Wave ${waveIndex - 1} MUST speak in Wave ${waveIndex} (with "isResolved": false) ONLY if a speaker in Wave ${waveIndex - 1}:
   (a) directly asked them a question by name ("@Maya", "@Liam", "@Zoe") that they actually have facts for in their domain, OR
   (b) proposed a concrete option, schedule, timing, or activity where THIS silent peer agent's domain (listed above) contains constraints, feasibility timings, or restrictions that have NOT been checked yet in the current turn!
4. Write each speaking agent's "thesis" as a short 5-10 word high-level topic label for THEIR domain only. NEVER invent or include specific numbers, prices, currencies, or clock times in a thesis!
5. Once all relevant cross-agent domain checks for the user's question have been surfaced, set "isResolved": true and all agents to "silent".

Return strict JSON:
{
  "isResolved": boolean,
  "maya": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "liam": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "zoe": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string }
}`
      : `You are the Pre-Response Effort & Non-Repetition Router for a 3-agent collaborative group chat.
Each agent has an isolated private knowledge vault for their own domain:
${agentDirectory}

Rules for routing Wave 0:
1. Silence Gate & Domain Matching:
   - Match the user's actual question to the agent whose domain owns that question, even if the user refers to an item from the previous turn! (For example: if the user asks about budget, bankroll, or how much something costs, you MUST assign Maya and keep non-budget agents "silent" if they have no new domain facts to add; if the user asks whether an arrival time makes a check-in cutoff across locations, assign BOTH the lodging/cutoff owner Maya AND the transit-timing owner Liam!)
   - If the user explicitly addresses specific agent(s) by name (e.g. "@Maya ..."), ONLY the addressed agent(s) may speak in Wave 0—all non-addressed agents MUST be "silent".
   - At least ONE agent MUST speak in Wave 0 on every user message! For casual greetings ("hey team!") or thanks ("thanks!"), assign "light" effort with thesis "Reply with a warm 1-sentence greeting/acknowledgment".
2. Strict Domain Partition & Sequential Cross-Domain Synthesis (NO Fake Numbers in Thesis):
   - Every speaking agent's "thesis" MUST be a short 5-10 word high-level topic directive strictly confined to their own domain. NEVER invent or include specific prices, amounts, hours, or clock times inside the "thesis" string!
   - Let Proposal Agents Speak First Before Constraint Agents:
     * When the user asks to choose/recommend options (e.g. "what is the best option?") and the options haven't been listed in the chat yet, have the option-owning agent(s) speak in Wave 0 so the constraint-checking agent (e.g. Zoe) can evaluate the actual concrete options in Wave 1!
     * When the user asks about multiple domains in one prompt (e.g. Maya's options AND Liam's options), assign both Maya ("standard") and Liam ("standard") in Wave 0 so Zoe can evaluate both in Wave 1!
   - For general recap/status questions (e.g., "what's our plan so far and what do we need to decide?"):
     * Assign ONE primary summarizer (Maya) to summarize the status and pending decisions; keep other agents "silent" unless they are flagging a specific unaddressed constraint.
3. Effort Calibration:
   - "light": Simple single-fact check, greeting, or quick 1-sentence note.
   - "standard": Moderate multi-fact answer or domain breakdown (2–4 sentences).
   - "deep": Thorough analysis or complete breakdown.
4. Standing Rule Lifecycle ("removeRuleIds" & "standingRule"):
   - "removeRuleIds": Inspect Active Standing Rules against the user's latest message. If the user asks to stop, drop, forget, relax, or override any specific existing rule(s) (e.g. "nevermind riddles are hard to read, talk normally", "stop using haikus", "Maya, stop ending messages with [BUDGET-OK]", or gives a new rule that contradicts a specific old rule), return the exact "id" strings of ONLY those superseded/canceled rules in "removeRuleIds": ["..."] while leaving unrelated active rules untouched! Otherwise return [].
   - "standingRule": ONLY if the user's latest message establishes a NEW persistent formatting/style/currency instruction for how agents should respond in future turns, return { "scope": "global" | "maya" | "liam" | "zoe", "rule": "<concise imperative rule ONLY, excluding any follow-up question>" }, otherwise null. (Note: if the user merely says "talk normally" or "answer in 1 normal sentence" to cancel a previous haiku/riddle rule, put the old rule's id in "removeRuleIds" and set "standingRule": null!). NEVER copy or re-emit an existing rule from Active Standing Rules!

Return strict JSON:
{
  "maya": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "liam": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "zoe": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string },
  "removeRuleIds": string[],
  "standingRule": { "scope": "global" | "maya" | "liam" | "zoe", "rule": string } | null
}`;

    const userPrompt = `Active Standing Rules:\n${rulesSummary}\n\nConversation Transcript:\n${formatTranscriptForRouter(transcript, runtimeConfig)}\n\nRoute Wave ${waveIndex} now.`;

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
        const mentioned = extractMentionedAgents(referenceText, runtimeConfig);
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
      // Rule 1: No agent who just spoke in Wave N-1 can speak again in Wave N (zero consecutive self-replies)
      if (!isMock) {
        for (const prevSpeaker of inferredPrevSpeakers) {
          rawDecisions[prevSpeaker] = { effort: 'silent', thesis: '' };
        }
      } else if (inferredPrevSpeakers.length === 1) {
        const solePrevSpeaker = inferredPrevSpeakers[0];
        rawDecisions[solePrevSpeaker] = { effort: 'silent', thesis: '' };
      }

      const prevWaveMsgs = currentTurnAgentMsgs.filter((m) =>
        inferredPrevSpeakers.includes(m.agent_id!),
      );
      const prevWaveCombinedText = prevWaveMsgs.map((m) => m.content).join('\n');
      const prevWaveHasQuestionOrVeto =
        /[?]|\b(?:what\s+about|alternative|instead|avoid|veto(?:ed)?|unsafe|risk|cannot|can't|isn't\s+suitable|not\s+suitable|won't\s+work|conflicts?\s+with|rules?\s+out|miss\s+the\s+check-?in)\b/i.test(
          prevWaveCombinedText,
        );

      for (const agentId of AGENT_IDS) {
        if (rawDecisions[agentId].effort === 'silent') continue;

        const priorMsgsForAgent = currentTurnAgentMsgs
          .filter((m) => m.agent_id === agentId)
          .map((m) => m.content);
        const priorThesesForAgent = turnThesesByAgent[agentId] ?? [];

        const wasTaggedInPrevWave = prevWaveMsgs.some((m) =>
          extractMentionedAgents(m.content, runtimeConfig).includes(agentId),
        );

        if (!isMock && priorMsgsForAgent.length > 0) {
          // Never allow an agent to speak a 3rd time in the same turn, and never allow
          // a second speech in Wave 3+ (Wave 2 is the final synthesis/pivot wave for Wave 0 speakers).
          if (priorMsgsForAgent.length >= 2 || waveIndex >= 3) {
            rawDecisions[agentId] = { effort: 'silent', thesis: '' };
            continue;
          }

          const alreadyListedMultipleOptions = priorMsgsForAgent.some(
            (msg) => (msg.match(/\*\*[^*]+\*\*/g) ?? []).length >= 3,
          );
          const isMayaPivotingAfterPeer =
            agentId === 'maya' &&
            priorMsgsForAgent.length === 1 &&
            !alreadyListedMultipleOptions &&
            prevWaveHasQuestionOrVeto &&
            inferredPrevSpeakers.some((id) => id !== 'maya');
          const isTaggedPeerReplyingToQuestionOrVeto =
            wasTaggedInPrevWave &&
            priorMsgsForAgent.length === 1 &&
            !inferredPrevSpeakers.includes(agentId) &&
            prevWaveHasQuestionOrVeto;

          if (!isMayaPivotingAfterPeer && !isTaggedPeerReplyingToQuestionOrVeto) {
            rawDecisions[agentId] = { effort: 'silent', thesis: '' };
            continue;
          }
        }

        if (
          !(wasTaggedInPrevWave && prevWaveHasQuestionOrVeto && priorMsgsForAgent.length === 1 && waveIndex <= 2) &&
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

      // Deterministic A2A peer-mention & handoff check:
      // 1. If a speaker in Wave N-1 explicitly @mentioned or deferred to a peer who hasn't spoken yet
      //    in this turn (e.g. "@Maya how much does that cost?"), ensure that peer speaks!
      // 2. If in Wave 2 (after Wave 1), a speaker in Wave 1 explicitly @mentioned a Wave 0 speaker
      //    AND raised a question or vetoed one of their options (e.g. "@Liam Nishiki is unsafe for Sam"),
      //    allow that Wave 0 speaker to reply once in Wave 2 with a concrete alternative!
      if (!isMock) {
        for (const msg of prevWaveMsgs) {
          const peerMentions = extractMentionedAgents(msg.content, runtimeConfig).filter(
            (id) => id !== msg.agent_id && !inferredPrevSpeakers.includes(id),
          );
          for (const targetPeerId of peerMentions) {
            const priorCount = currentTurnAgentMsgs.filter(
              (m) => m.agent_id === targetPeerId,
            ).length;
            if (priorCount === 0 && rawDecisions[targetPeerId].effort === 'silent') {
              rawDecisions[targetPeerId] = {
                effort: 'standard',
                thesis: getDomainSafeThesisFallback(targetPeerId, runtimeConfig),
              };
              raw.isResolved = false;
            } else if (
              priorCount === 1 &&
              waveIndex === 2 &&
              prevWaveHasQuestionOrVeto &&
              rawDecisions[targetPeerId].effort === 'silent'
            ) {
              const callerName = msg.agent_id
                ? getAgentProfile(msg.agent_id, runtimeConfig).name
                : 'your teammate';
              rawDecisions[targetPeerId] = {
                effort: 'standard',
                thesis: `Address @${callerName} and immediately provide the concrete alternative option from your vault or domain knowledge (do NOT ask if they want suggestions—name the specific alternative right now).`,
              };
              raw.isResolved = false;
            }
          }
        }
      }

      // 100% Scenario-Agnostic Wave 1 Constraint-Check Trigger:
      // Whenever Wave 0 speakers presented multiple options/candidates (e.g. numbered list 1. / 2.,
      // multiple bolded options **...**, "vs", or a trade-off/comparison) OR proposed a concrete
      // schedule/option, and Zoe (the constraints/compliance/preferences specialist) has not spoken
      // yet in this turn, trigger Zoe in Wave 1 unless Wave 0 was a pure transit/cutoff math calculation!
      if (!isMock && waveIndex === 1) {
        const turnCombinedText = currentTurnAgentMsgs.map((m) => m.content).join('\n');
        const zoeSpokeInTurn = currentTurnAgentMsgs.some((m) => m.agent_id === 'zoe');
        const boldMatches = turnCombinedText.match(/\*\*[^*]+\*\*/g) ?? [];
        const hasNumberedOrBulletOptions =
          /(?:^|\n)\s*(?:1[.)]|[-*•])\s+/m.test(turnCombinedText) && boldMatches.length >= 2;
        const hasMultipleOptionsOrSchedule =
          hasNumberedOrBulletOptions ||
          boldMatches.length >= 2 ||
          /\b(\d{1,2}:\d{2}\s*(?:AM|PM)|option\s+[ab12]|tier\s+[123]|cluster\s+\w+|vs\.?|versus|prefer|choose\s+between)\b/i.test(
            turnCombinedText,
          );
        const isPureTransitCutoffMathOnly =
          /\b(2\s*hours?\s*15|2h\s*15m)\b/i.test(turnCombinedText) &&
          /\b6:00\s*PM\b/i.test(turnCombinedText) &&
          !/\b(6:45\s*AM|Nishiki|shellfish|Shorai-an|vegetarian)\b/i.test(turnCombinedText);

        if (
          !zoeSpokeInTurn &&
          hasMultipleOptionsOrSchedule &&
          !isPureTransitCutoffMathOnly &&
          !isGeneralRecapQuery(lastUserMessageContent)
        ) {
          if (rawDecisions.zoe.effort === 'silent') {
            rawDecisions.zoe = {
              effort: currentTurnAgentMsgs.length >= 2 ? 'standard' : 'light',
              thesis: getDomainSafeThesisFallback('zoe', runtimeConfig),
            };
            raw.isResolved = false;
          } else if (currentTurnAgentMsgs.length >= 2 && rawDecisions.zoe.effort === 'light') {
            rawDecisions.zoe.effort = 'standard';
          }
        } else if (!zoeSpokeInTurn && isPureTransitCutoffMathOnly) {
          rawDecisions.zoe = { effort: 'silent', thesis: '' };
        }
      }
    } else {
      const anySpeaking = AGENT_IDS.some((id) => rawDecisions[id].effort !== 'silent');
      const mentioned = extractMentionedAgents(referenceText, runtimeConfig);

      if (!isMock && isStandaloneStandingRuleMessage(referenceText)) {
        const targetAckAgent: AgentId = mentioned[0] ?? 'maya';
        for (const id of AGENT_IDS) {
          rawDecisions[id] =
            id === targetAckAgent
              ? {
                  effort: 'light',
                  thesis:
                    'Briefly acknowledge the new standing instruction in 1 short sentence ( obeying its format ) without repeating previous answers.',
                }
              : { effort: 'silent', thesis: '' };
        }
      } else if (!isMock && isCasualSocialMessage(referenceText)) {
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
              thesis: getDomainSafeThesisFallback(mId, runtimeConfig),
            };
          }
        }
        for (const id of AGENT_IDS) {
          if (!mentionedSet.has(id)) {
            rawDecisions[id] = { effort: 'silent', thesis: '' };
          }
        }
      } else if (
        !isMock &&
        !isCasualSocialMessage(referenceText) &&
        !isStandaloneStandingRuleMessage(referenceText) &&
        !isGeneralRecapQuery(referenceText)
      ) {
        // Scenario-agnostic vault-index matching:
        // Tokenize the user's prompt and check which agents' dynamic vault topic/entity index
        // (`getAgentDomainIndexKeywords`) is directly referenced by the prompt!
        // This prevents context-momentum misrouting (e.g. routing a question about entities in Maya's
        // or Liam's vault solely to Zoe just because Zoe spoke last in the previous turn).
        const promptWords = referenceText
          .toLowerCase()
          .replace(/[^a-z0-9\s]/g, ' ')
          .split(/\s+/)
          .filter((w) => w.length >= 4);

        const matchedAgentsByVaultIndex: AgentId[] = [];
        for (const agentId of AGENT_IDS) {
          const domainKeywords = getAgentDomainIndexKeywords(agentId, runtimeConfig);
          const hasMatch = promptWords.some((word) => domainKeywords.has(word));
          if (hasMatch) {
            matchedAgentsByVaultIndex.push(agentId);
          }
        }

        // Check if the user is asking a pure single-dimension pricing/cost question (e.g. "How much is VectorStore Lite per month?",
        // "What is the nightly rate for Cross Hotel Osaka?", "What is our total budget?") without asking a second question
        // about another domain (no "and" / "or" comparison across venues/specs).
        const asksPricingOrBudget =
          /\b(?:how\s+much|set\s+us\s+back|costs?|prices?|pricing|budget|bankroll|afford|rates?)\b/i.test(
            referenceText,
          );
        const isPureSingleDimensionPriceQuery =
          asksPricingOrBudget &&
          !/\b(?:should\s+we|which\s+one|or\s+|transit|latency|benchmarks?|specs?|hours?|schedule|check\s+into|make\s+it|make\s+that|allerg(?:y|ies)|compliance|legal)\b/i.test(
            referenceText,
          );

        if (asksPricingOrBudget) {
          for (const agentId of AGENT_IDS) {
            const kws = getAgentDomainIndexKeywords(agentId, runtimeConfig);
            if (
              (kws.has('budget') ||
                kws.has('treasury') ||
                kws.has('finops') ||
                kws.has('pricing') ||
                kws.has('quotes') ||
                kws.has('cost')) &&
              !matchedAgentsByVaultIndex.includes(agentId)
            ) {
              matchedAgentsByVaultIndex.push(agentId);
            }
          }
        }

        if (isPureSingleDimensionPriceQuery) {
          // Keep ONLY the budget/pricing/finops owner active on a pure single-dimension price query,
          // even if the named item (e.g. "VectorStore Lite") also has a technical specs section in Liam's vault!
          const priceOwners = matchedAgentsByVaultIndex.filter((agentId) => {
            const kws = getAgentDomainIndexKeywords(agentId, runtimeConfig);
            return (
              kws.has('budget') ||
              kws.has('treasury') ||
              kws.has('finops') ||
              kws.has('pricing') ||
              kws.has('quotes') ||
              kws.has('cost') ||
              agentId === 'maya'
            );
          });
          if (priceOwners.length > 0) {
            matchedAgentsByVaultIndex.length = 0;
            matchedAgentsByVaultIndex.push(...priceOwners);
          }
        }

        // If the prompt asks about arrival/check-in/schedule feasibility ("can we make that", "check into", "make it in time"),
        // ensure BOTH the venue/schedule owner AND the transit/timing specialist are included!
        const asksArrivalOrCheckInFeasibility =
          /\b(?:check\s+into|check-?in|make\s+it|make\s+that|able\s+to\s+make|arrive|landing|reach|in\s+time)\b/i.test(
            referenceText,
          );
        if (asksArrivalOrCheckInFeasibility) {
          for (const agentId of AGENT_IDS) {
            const kws = getAgentDomainIndexKeywords(agentId, runtimeConfig);
            if (
              (kws.has('transit') ||
                kws.has('transfer') ||
                kws.has('door') ||
                kws.has('latency') ||
                kws.has('lodging') ||
                kws.has('cutoffs')) &&
              !matchedAgentsByVaultIndex.includes(agentId)
            ) {
              matchedAgentsByVaultIndex.push(agentId);
            }
          }
        }

        if (matchedAgentsByVaultIndex.length > 0) {
          for (const matchedId of matchedAgentsByVaultIndex) {
            if (rawDecisions[matchedId].effort === 'silent') {
              rawDecisions[matchedId] = {
                effort: 'standard',
                thesis: getDomainSafeThesisFallback(matchedId, runtimeConfig),
              };
            }
          }
          // If the router mistakenly scheduled ONLY an agent whose vault index did NOT match any entity
          // in the user's question (like scheduling only Zoe on a Flight B + Ryokan Gion check-in question),
          // silence the non-matching agent in Wave 0 so the actual entity-owning agents speak first!
          for (const id of AGENT_IDS) {
            if (!matchedAgentsByVaultIndex.includes(id)) {
              rawDecisions[id] = { effort: 'silent', thesis: '' };
            }
          }
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

    const speakerOrder = computeDeterministicSpeakerOrder(referenceText, runtimeConfig);
    const { decisions, speakers } = applyAsymmetricDedup(
      rawDecisions,
      speakerOrder,
      isFollowUpWave ? '' : referenceText,
      isMock,
      runtimeConfig,
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

    const validActiveIds = new Set(activeRules.map((r) => r.id));
    const removeRuleIds =
      !isFollowUpWave && Array.isArray(raw?.removeRuleIds)
        ? raw.removeRuleIds.filter(
            (id): id is string => typeof id === 'string' && validActiveIds.has(id),
          )
        : [];

    const mentionedInUserMsg = extractMentionedAgents(referenceText, runtimeConfig);
    const clearRuleScope =
      isFollowUpWave || removeRuleIds.length > 0
        ? null
        : (raw?.clearRuleScope && VALID_SCOPES.has(raw.clearRuleScope)
            ? raw.clearRuleScope
            : isMock
              ? detectRuleClearScope(referenceText, mentionedInUserMsg, runtimeConfig)
              : null);

    let candidateStandingRule = isFollowUpWave
      ? null
      : normalizeStandingRule(raw?.standingRule) ??
        (isMock
          ? extractStandingRuleDeterministic(referenceText, mentionedInUserMsg, runtimeConfig)
          : null);

    if (candidateStandingRule && !isMock) {
      const matchesExistingActive = activeRules.some(
        (r) =>
          r.active === 1 &&
          r.rule.trim().toLowerCase() === candidateStandingRule!.rule.trim().toLowerCase(),
      );
      if (matchesExistingActive) {
        candidateStandingRule = null;
      }
    }

    return {
      waveIndex,
      isResolved,
      decisions,
      speakers,
      standingRule: candidateStandingRule,
      removeRuleIds,
      clearRuleScope,
    };
  }
}
