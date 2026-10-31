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
  /** A same-wave speaker whose finished answer this agent must see before it starts streaming. */
  dependsOn?: AgentId | null;
  /** Follow-up waves: the previous-wave co-speaker whose question to this agent it is answering. */
  answersCoSpeaker?: AgentId | null;
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
  dependsOn?: AgentId;
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
  /** Seq of the user message that opened this turn; mid-stream steering messages don't start a new turn. */
  turnUserMessageSeq?: number;
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

const FOLLOW_UP_VERIFICATION_CUE_RE =
  /\b(?:i\s+am\s+gathering|i'm\s+gathering|so\s+i\s+gather|so\s+we\s+are\s+good|so\s+we're\s+good|should\s+still\s+be\s+good|we\s+should\s+be\s+fine|is\s+that\s+right|correct\?|right\?|just\s+to\s+confirm|to\s+make\s+sure|wait\s+so|so\s+basically)\b/i;

export function isFollowUpVerificationOnDiscussedTopics(
  userText: string,
  transcript: MessageRow[],
  runtimeConfig?: CustomAgentRuntimeConfig,
): boolean {
  const trimmed = userText.trim();
  const priorAgentMsgs = transcript.filter((m) => m.role === 'agent' && m.content.length > 0);
  if (priorAgentMsgs.length === 0) return false;

  if (FOLLOW_UP_VERIFICATION_CUE_RE.test(trimmed)) {
    return true;
  }

  // Also check if every vault-indexed entity mentioned in the user's prompt has ALREADY been
  // discussed in prior agent messages AND the user is asking if it's "good" / "fine" / "okay" / "safe"
  const asksIfGoodOrSafe =
    /\b(?:still\s+be\s+good|be\s+fine|be\s+okay|work\s+for\s+us|good\s+to\s+go)\b/i.test(trimmed);
  if (!asksIfGoodOrSafe) return false;

  const priorCombined = priorAgentMsgs.map((m) => m.content.toLowerCase()).join('\n');
  const promptWords = trimmed
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length >= 4);

  const allVaultKeywords = new Set<string>();
  for (const id of AGENT_IDS) {
    for (const kw of getAgentDomainIndexKeywords(id, runtimeConfig)) {
      allVaultKeywords.add(kw);
    }
  }

  const mentionedVaultWords = promptWords.filter((w) => allVaultKeywords.has(w));
  return (
    mentionedVaultWords.length > 0 &&
    mentionedVaultWords.every((w) => priorCombined.includes(w))
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

export function getConstraintCheckThesis(
  agentId: AgentId,
  runtimeConfig?: CustomAgentRuntimeConfig,
): string {
  const topics = getAgentVaultTopicSummary(agentId, runtimeConfig);
  return `Check the options and schedules your teammates just proposed against the constraints in your private vault (${topics}). Say which ones work and which don't, naming the constraint each time. Only flag a conflict a constraint actually covers; never stretch a rule to fit an option it doesn't apply to.`;
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

const VETO_CUE_RE =
  /\b(?:avoid|veto(?:ed)?|unsafe|not\s+suitable|isn't\s+suitable|cannot|can't|won't\s+work|violates?|forfeits?|rules?\s+out|conflicts?\s+with|breaks?|different\s+\w+\s+option|alternative)\b/i;

const RESOLUTION_CUE_RE =
  /\b(?:works|agree|you're\s+right|exactly|go\s+with|opt\s+for|stick\s+with|recommend|(?:is|sounds\s+like|seems\s+like|looks\s+like)\s+(?:\w+ly\s+|indeed\s+)?(?:a\s+|the\s+)?(?:better|safer|good|right|best)\s+(?:choice|option|fit|bet|call|plan))\b/i;

const ALTERNATIVE_OFFER_RE = /\b(?:instead|alternative|recommend|consider(?:ing)?|switch(?:ing)?\s+to)\b/i;

function hasAlreadyOfferedAlternatives(messages: string[]): boolean {
  return messages.some(
    (msg) =>
      /\b1[.)]\s+[\s\S]*\b2[.)]\s+/.test(msg) ||
      (msg.match(/(?:^|\n)\s*[-*•]\s+/g) ?? []).length >= 2 ||
      ALTERNATIVE_OFFER_RE.test(msg),
  );
}

function asksAgentDirectly(content: string, agentId: AgentId, runtimeConfig?: CustomAgentRuntimeConfig): boolean {
  return content
    .split(/(?<=[.!?])\s+|\r?\n/)
    .some((sentence) => sentence.includes('?') && extractMentionedAgents(sentence, runtimeConfig).includes(agentId));
}

/**
 * A peer's message earns `agentId` another turn only if it names them and either asks them something
 * or vetoes their option when they haven't offered an alternative and the peer didn't settle on one.
 */
function peerMessageNeedsReplyFrom(
  content: string,
  agentId: AgentId,
  agentPriorMsgs: string[],
  runtimeConfig?: CustomAgentRuntimeConfig,
): boolean {
  if (!extractMentionedAgents(content, runtimeConfig).includes(agentId)) return false;
  if (asksAgentDirectly(content, agentId, runtimeConfig)) return true;
  return (
    agentPriorMsgs.length > 0 &&
    !hasAlreadyOfferedAlternatives(agentPriorMsgs) &&
    VETO_CUE_RE.test(content) &&
    !RESOLUTION_CUE_RE.test(content)
  );
}

// The LLM sometimes reads "wrap up in one sentence" as stop; a stop must actually ask to stop.
const STOP_INTENT_RE =
  /\b(?:stop|halt|cancel|abort|pause|quiet|shut\s+up|enough|never\s*mind|nevermind|forget\s+it|hold\s+on|that'?s\s+all)\b/i;

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
  const dependsOn = AGENT_IDS.find((id) => id === raw?.dependsOn);
  const answersCoSpeaker = AGENT_IDS.find((id) => id === raw?.answersCoSpeaker);
  return {
    effort,
    thesis: thesis || 'Provide relevant domain facts from private vault.',
    ...(dependsOn ? { dependsOn } : {}),
    ...(answersCoSpeaker ? { answersCoSpeaker } : {}),
  };
}

/** Keeps only dependencies on another speaker in the same wave that is itself independent. */
function resolveSpeakerDependencies(
  speakers: SpeakerPlan[],
  decisions: Record<AgentId, AgentRouteDecision>,
): void {
  const speakerIds = new Set(speakers.map((s) => s.agentId));
  for (const speaker of speakers) {
    const target = decisions[speaker.agentId].dependsOn;
    if (target && target !== speaker.agentId && speakerIds.has(target) && !decisions[target].dependsOn) {
      speaker.dependsOn = target;
    }
  }
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

function getCurrentTurnAgentMessages(
  transcript: MessageRow[],
  turnUserMessageSeq?: number,
): MessageRow[] {
  if (turnUserMessageSeq !== undefined) {
    return transcript.filter(
      (m) => m.seq > turnUserMessageSeq && m.role === 'agent' && Boolean(m.agent_id),
    );
  }
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

const DEPENDS_ON_RULE = `"dependsOn":
   - Speaking agents stream in parallel and cannot see each other's answers. If one speaking agent's answer genuinely needs facts another speaking agent will state in this round, set its "dependsOn" to that agent's id so it waits and builds on that answer. Otherwise null, so independent answers stay parallel.
   - Direction matters: facts flow from the agent who owns the INPUT to the agent whose answer is CONDITIONAL on it. The waiting agent is the one that would otherwise have to say "it depends on…" or ask the other agent. Example: for "when is it ready and what should we plan right after?", the follow-up plan is conditional on the ready time, so the owner of the follow-up depends on the owner of that time — never the reverse.
   - Feasibility questions ("does X fit before Y?", "can we make it in time?", "will it finish by the deadline?") need a start time or deadline from one agent and a duration from another. The agent who owns the deadline or cutoff gives the verdict, so it depends on the agent who owns the duration; otherwise it would have to guess whether the plan fits.
   - Keep each speaker's thesis to the part of the question their own domain owns, so the input owner does not also plan the downstream part.
   - Never let two agents depend on each other, and never chain dependencies.`;

const STEERING_DRAFT_PREVIEW_CHARS = 300;

function buildSteeringContext(
  transcript: MessageRow[],
  activeAgents: AgentId[],
  currentUserMessage: string,
): string {
  const priorUserMessages = transcript.filter(
    (m) => m.role === 'user' && m.content.trim() !== currentUserMessage,
  );
  const previousQuestion = priorUserMessages[priorUserMessages.length - 1]?.content.trim();

  const latestDraftByAgent = new Map<AgentId, string>();
  for (const m of transcript) {
    if (m.role === 'agent' && m.status === 'streaming' && m.agent_id && activeAgents.includes(m.agent_id)) {
      latestDraftByAgent.set(m.agent_id, m.content);
    }
  }
  const draftLines = Array.from(latestDraftByAgent.entries())
    .filter(([, content]) => content.trim())
    .map(([agentId, content]) => {
      const flat = content.replace(/\s+/g, ' ').trim();
      const preview =
        flat.length > STEERING_DRAFT_PREVIEW_CHARS
          ? `${flat.slice(0, STEERING_DRAFT_PREVIEW_CHARS)}…`
          : flat;
      return `- ${agentId}: "${preview}"`;
    });

  const parts: string[] = [];
  if (previousQuestion) {
    parts.push(`The question they're answering: "${previousQuestion}"`);
  }
  if (draftLines.length > 0) {
    parts.push(`What each streaming agent has written so far:\n${draftLines.join('\n')}`);
  }
  return parts.join('\n');
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
    const mayaName = getAgentProfile('maya', runtimeConfig).name;
    const liamName = getAgentProfile('liam', runtimeConfig).name;
    const formattedRulesWithIds =
      activeRules.length > 0
        ? activeRules
            .map((r) => `- id="${r.id}" [scope=${r.scope}]: ${r.rule}`)
            .join('\n')
        : '(None)';

    const steeringContext = buildSteeringContext(input.transcript ?? [], activeAgents, trimmed);

    const systemPrompt = `You classify a message the user sent while agents in a group chat were still writing their replies.

Agents:
${agentDirectory}
${steeringContext ? `\n${steeringContext}\n` : ''}
Active standing rules:
${formattedRulesWithIds}

Return JSON with these fields:
1. "action":
   - "stop_all": the user wants everyone to stop writing and says nothing about what the replies should still contain.
   - "steer": the user is adjusting the replies in progress (currency, format, length, focus) or addressing an agent, including calling in one who isn't writing yet. Prefer this whenever the message relates to the replies in progress. A message about how to finish ("wrap up in one sentence", "finish quickly", "conclude now", "just give the bottom line") is a steer, not a stop: the replies must still end the way the user asked.
   - "new_turn": the user has moved on to a different question, so the replies in progress should be abandoned.
2. "targetAgents": the agents the message is aimed at, by name or role (e.g. "${mayaName}", "@${liamName}", or a role like "the budget person"). If it's aimed at everyone, return all streaming agents: ${JSON.stringify(activeAgents)}.
3. "directive": one line telling the targeted agents what to change, written to them in second person and keeping the user's specifics (e.g. "Quote all prices in euros.", "Keep your answer to one sentence."). Don't include agent names.
4. "removeRuleIds": ids of active rules the user cancels or contradicts (e.g. "talk normally" cancels a haiku rule; "no more [OK] tag" cancels that tag). Leave unrelated rules alone. Otherwise [].
5. "standingRule": only if the user signals the instruction should outlast this reply ("always", "from now on", "going forward", "all prices"), return { "scope": the named agent's id or "global", "rule": a short imperative }. A one-off change like "make this shorter" is not a standing rule. Never repeat an existing active rule. Otherwise null.

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

      if (raw.action === 'stop_all' && !STOP_INTENT_RE.test(trimmed)) {
        raw.action = 'steer';
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
      turnUserMessageSeq,
      previousWaveSpeakers,
      turnThesesByAgent = {},
      signal,
    } = input;
    const runtimeConfig = input.runtimeConfig ?? this.defaultRuntimeConfig;
    const isFollowUpWave = waveIndex > 0;
    const isMock = this.llm instanceof ScriptableMockLLMProvider;

    const turnUserMessage =
      turnUserMessageSeq !== undefined
        ? transcript.find((m) => m.seq === turnUserMessageSeq && m.role === 'user')
        : [...transcript].reverse().find((m) => m.role === 'user');
    const lastUserMessageContent = turnUserMessage?.content ?? latestUserMessage ?? '';

    const rulesSummary =
      activeRules.length > 0
        ? activeRules
            .map((r) => `- id="${r.id}" [scope=${r.scope}]: ${r.rule}`)
            .join('\n')
        : '(None)';

    const currentTurnAgentMsgs = getCurrentTurnAgentMessages(transcript, turnUserMessageSeq);
    const inferredPrevSpeakers: AgentId[] =
      previousWaveSpeakers ??
      (currentTurnAgentMsgs.length > 0 && currentTurnAgentMsgs[currentTurnAgentMsgs.length - 1].agent_id
        ? [currentTurnAgentMsgs[currentTurnAgentMsgs.length - 1].agent_id!]
        : []);

    const agentDirectory = buildAgentDomainDirectoryForRouter(runtimeConfig);
    const mayaName = getAgentProfile('maya', runtimeConfig).name;
    const zoeName = getAgentProfile('zoe', runtimeConfig).name;

    const systemPrompt = isFollowUpWave
      ? `You decide whether anyone should follow up after the last round of replies in a group chat, and if so who and how much. Agents reply in parallel and can't see each other's replies until they're posted.

Agents (each has a private vault you can't see; you only see its section names):
${agentDirectory}

Agents who spoke in the last round: ${JSON.stringify(inferredPrevSpeakers)}

Rules:
1. First, judge the user's latest message (quoted at the end of the input) by its intent, not its opening word. Open items or next steps listed in an agent's reply don't change this.
   - If it only greets, thanks, acknowledges, commits to an option already discussed, or sets a style rule, and the first round handled it, the exchange is resolved.
   - If it asks for a recap, status, or confirmation of facts already in the chat, it's resolved, unless the recapped plan conflicts with a constraint in a silent agent's domain that no message in the whole conversation has raised (e.g. the plan includes an option that breaks a personal, safety, or compliance rule only that agent knows about). Then schedule only that agent at "light" effort to flag that one conflict without restating the plan.
   - If it asks for something new (e.g. "ok, let's look at our other options"), apply rules 2 and 3.
2. Agents who spoke in the last round stay silent this round, so nobody replies to themselves. One exception: if a last-round speaker @asked another last-round speaker something that agent's own message didn't already answer (they couldn't see each other), schedule the asked agent with "answersCoSpeaker" set to the asker's id and a thesis to answer that question using the asker's facts. Otherwise "answersCoSpeaker" is null.
3. Any other agent speaks only if a last-round speaker:
   (a) @asked them something their domain covers,
   (b) proposed an option, schedule, or activity that their domain's constraints haven't been checked against yet this turn, or
   (c) raised a problem with an option this agent proposed earlier in the turn, so it can offer an alternative.
4. Set "isResolved": true and every agent to "silent" once the user's question is answered and every relevant cross-domain check has been made. Never schedule an agent just to say it doesn't have the information.
5. "thesis": 5–10 words naming what that agent will cover, within its own domain. Teammates see it and avoid repeating it, so make it specific. Don't include numbers, prices, or times: you can't see the vaults, so any figure would be a guess.
6. "effort": follow-ups are usually "light" or "standard"; use "deep" only if the user asked for a full breakdown.
7. ${DEPENDS_ON_RULE}

Return strict JSON:
{
  "isResolved": boolean,
  "maya": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string, "dependsOn": "liam" | "zoe" | null, "answersCoSpeaker": "liam" | "zoe" | null },
  "liam": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string, "dependsOn": "maya" | "zoe" | null, "answersCoSpeaker": "maya" | "zoe" | null },
  "zoe": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string, "dependsOn": "maya" | "liam" | null, "answersCoSpeaker": "maya" | "liam" | null }
}`
      : `You decide which agents reply to the user's latest message in a group chat, and how much each says. Agents reply in parallel and can't see each other's replies until they're posted.

Agents (each has a private vault you can't see; you only see its section names):
${agentDirectory}

Rules:
1. Who speaks:
   - Pick the agent(s) whose domain owns the question, even when it refers back to an earlier turn. Keep agents silent if they have nothing new to add from their domain. (E.g. a cost question goes to the agent whose domain covers budget; "can we finish this before that deadline?" needs both the owner of the deadline and the owner of the duration.)
   - If the user addresses agents by name ("@${mayaName} …"), only those agents speak.
   - At least one agent speaks on every message. For greetings or thanks, one agent replies at "light" effort.
   - When the user asks to choose between options that haven't been listed yet, have the agents who own those options speak now; constraint checks (by ${zoeName}) happen in the follow-up round once the options are on screen. If the question spans several option owners, schedule all of them.
   - For a recap or status question, schedule one summarizer (${mayaName}) and keep others silent unless they need to flag a specific constraint nobody has raised.
2. "thesis": 5–10 words naming what that agent will cover, within its own domain. Teammates see it and avoid repeating it, so make it specific. Don't include numbers, prices, or times: you can't see the vaults, so any figure would be a guess.
3. "effort": "light" for a single fact, greeting, or quick note; "standard" for a few facts (2–4 sentences); "deep" only when the user asks for a breakdown, comparison, or full plan.
4. "removeRuleIds": ids of active rules the user's message cancels or contradicts (e.g. "talk normally" cancels a haiku rule; "${mayaName}, stop ending with [OK]" cancels that tag). Leave unrelated rules alone. Otherwise [].
5. "standingRule": only if the message sets how agents should respond from now on: words like "always", "from now on", "going forward", "all prices", or a message that's purely a style/format instruction (e.g. "answer in bullet points"). Return { "scope": the named agent's id or "global", "rule": a short imperative }. Cancelling an old rule ("talk normally") is not a new rule. Never repeat an active rule. Otherwise null.
6. ${DEPENDS_ON_RULE}

Return strict JSON:
{
  "maya": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string, "dependsOn": "liam" | "zoe" | null },
  "liam": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string, "dependsOn": "maya" | "zoe" | null },
  "zoe": { "effort": "silent" | "light" | "standard" | "deep", "thesis": string, "dependsOn": "maya" | "liam" | null },
  "removeRuleIds": string[],
  "standingRule": { "scope": "global" | "maya" | "liam" | "zoe", "rule": string } | null
}`;

    const userPrompt = `Active Standing Rules:\n${rulesSummary}\n\nConversation Transcript:\n${formatTranscriptForRouter(transcript, runtimeConfig)}${isFollowUpWave ? `\n\nUser's latest message: ${JSON.stringify(lastUserMessageContent)}` : ''}\n\nRoute Wave ${waveIndex} now.`;

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
    const llmDecisions = { ...rawDecisions };

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
      for (const agentId of AGENT_IDS) {
        if (rawDecisions[agentId].effort === 'silent') continue;

        const priorMsgsForAgent = currentTurnAgentMsgs
          .filter((m) => m.agent_id === agentId)
          .map((m) => m.content);
        const priorThesesForAgent = turnThesesByAgent[agentId] ?? [];

        const wasTaggedWithQuestionOrVetoInPrevWave = prevWaveMsgs.some((m) =>
          peerMessageNeedsReplyFrom(m.content, agentId, priorMsgsForAgent, runtimeConfig),
        );

        if (!isMock && priorMsgsForAgent.length > 0) {
          // Never allow an agent to speak a 3rd time in the same turn, and never allow
          // a second speech in Wave 3+ (Wave 2 is the final synthesis/pivot wave for Wave 0 speakers).
          if (priorMsgsForAgent.length >= 2 || waveIndex >= 3) {
            rawDecisions[agentId] = { effort: 'silent', thesis: '' };
            continue;
          }

          if (!wasTaggedWithQuestionOrVetoInPrevWave || inferredPrevSpeakers.includes(agentId)) {
            rawDecisions[agentId] = { effort: 'silent', thesis: '' };
            continue;
          }
        }

        if (
          !(wasTaggedWithQuestionOrVetoInPrevWave && priorMsgsForAgent.length === 1 && waveIndex <= 2) &&
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
      //    AND asked a question (`?`) or vetoed the single option that speaker proposed (when that speaker
      //    hadn't already listed the alternative in Wave 0), allow that Wave 0 speaker to reply once in Wave 2!
      if (!isMock && !isGeneralRecapQuery(lastUserMessageContent)) {
        for (const msg of prevWaveMsgs) {
          const peerMentions = extractMentionedAgents(msg.content, runtimeConfig).filter(
            (id) => id !== msg.agent_id && !inferredPrevSpeakers.includes(id),
          );
          for (const targetPeerId of peerMentions) {
            const targetPriorMsgs = currentTurnAgentMsgs
              .filter((m) => m.agent_id === targetPeerId)
              .map((m) => m.content);
            const priorCount = targetPriorMsgs.length;
            const peerAskedOrVetoedTarget = peerMessageNeedsReplyFrom(
              msg.content,
              targetPeerId,
              targetPriorMsgs,
              runtimeConfig,
            );

            if (priorCount === 0 && rawDecisions[targetPeerId].effort === 'silent') {
              rawDecisions[targetPeerId] = {
                effort: 'standard',
                thesis: getDomainSafeThesisFallback(targetPeerId, runtimeConfig),
              };
              raw.isResolved = false;
            } else if (
              priorCount === 1 &&
              waveIndex === 2 &&
              peerAskedOrVetoedTarget &&
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
      // yet in the conversation, trigger Zoe in Wave 1 to check those proposals against her constraints.
      if (!isMock && waveIndex === 1) {
        const turnCombinedText = currentTurnAgentMsgs.map((m) => m.content).join('\n');
        const zoeSpokeInTurn = currentTurnAgentMsgs.some((m) => m.agent_id === 'zoe');
        const zoeSpokeEarlierInHistory = transcript.some((m) => m.agent_id === 'zoe');
        const boldMatches = turnCombinedText.match(/\*\*[^*]+\*\*/g) ?? [];
        const hasNumberedOrBulletOptions =
          /(?:^|\n)\s*(?:1[.)]|[-*•])\s+/m.test(turnCombinedText) && boldMatches.length >= 2;
        const hasMultipleOptionsOrSchedule =
          hasNumberedOrBulletOptions ||
          boldMatches.length >= 2 ||
          /\b(\d{1,2}:\d{2}\s*(?:AM|PM)|option\s+[ab12]|tier\s+[123]|cluster\s+\w+|vs\.?|versus|prefer|choose\s+between)\b/i.test(
            turnCombinedText,
          );
        if (
          !zoeSpokeInTurn &&
          hasMultipleOptionsOrSchedule &&
          !isGeneralRecapQuery(lastUserMessageContent)
        ) {
          if (rawDecisions.zoe.effort === 'silent' && !zoeSpokeEarlierInHistory) {
            rawDecisions.zoe = {
              effort: currentTurnAgentMsgs.length >= 2 ? 'standard' : 'light',
              thesis: getConstraintCheckThesis('zoe', runtimeConfig),
            };
            raw.isResolved = false;
          } else if (currentTurnAgentMsgs.length >= 2 && rawDecisions.zoe.effort === 'light') {
            rawDecisions.zoe.effort = 'standard';
          }
        }
      }

      // Exception to "no consecutive waves": parallel co-speakers could not see each other, so the
      // router may let one answer a question its co-speaker asked it, once per turn.
      for (const agentId of AGENT_IDS) {
        const decision = llmDecisions[agentId];
        const asker = decision.answersCoSpeaker;
        const agentTurnMsgCount = currentTurnAgentMsgs.filter((m) => m.agent_id === agentId).length;
        const askerReallyAsked =
          isMock ||
          prevWaveMsgs.some(
            (m) => m.agent_id === asker && asksAgentDirectly(m.content, agentId, runtimeConfig),
          );
        if (
          decision.effort !== 'silent' &&
          asker &&
          asker !== agentId &&
          inferredPrevSpeakers.includes(agentId) &&
          inferredPrevSpeakers.includes(asker) &&
          agentTurnMsgCount === 1 &&
          askerReallyAsked
        ) {
          rawDecisions[agentId] = decision;
          raw.isResolved = false;
        }
      }

      const repeatSpeakers = AGENT_IDS.filter(
        (id) =>
          rawDecisions[id].effort !== 'silent' && currentTurnAgentMsgs.some((m) => m.agent_id === id),
      );
      if (!isMock && repeatSpeakers.length > 1) {
        const addressedFirst = extractMentionedAgents(
          prevWaveMsgs.map((m) => m.content).join('\n'),
          runtimeConfig,
        ).find((id) => repeatSpeakers.includes(id));
        const keep = addressedFirst ?? repeatSpeakers[0];
        for (const id of repeatSpeakers) {
          if (id !== keep) rawDecisions[id] = { effort: 'silent', thesis: '' };
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
        isFollowUpVerificationOnDiscussedTopics(referenceText, transcript, runtimeConfig)
      ) {
        // Single-Speaker Rule for Follow-Up Verification / Misconception Checks on Already-Discussed Topics:
        // When the user asks "ok so I am gathering that the market tour should still be good and that we should be fine getting the tofu kaiseki?",
        // all the facts are already in the shared transcript! Pick AT MOST ONE agent (preferring Zoe for group suitability/safety
        // checks, or the primary LLM-chosen speaker) so two agents never output identical 3-sentence clarifications!
        const activeCandidates = AGENT_IDS.filter((id) => rawDecisions[id].effort !== 'silent');
        const singleSpeaker: AgentId =
          activeCandidates.includes('zoe')
            ? 'zoe'
            : activeCandidates[0] ?? 'zoe';
        for (const id of AGENT_IDS) {
          if (id === singleSpeaker) {
            if (rawDecisions[id].effort === 'silent') {
              rawDecisions[id] = {
                effort: 'standard',
                thesis: 'Clarify what works and what must be avoided based on the constraints already discussed.',
              };
            }
          } else {
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
          .filter((w) => w.length >= 3);

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
          !/\b(?:should\s+we|can\s+we|where|what\s+time|which\s+one|or\s+|transit|latency|benchmarks?|specs?|hours?|schedule|check\s+into|make\s+it|make\s+that|food|tour|lunch|dinner|eat|restaurant|market|sight|allerg(?:y|ies)|compliance|legal)\b/i.test(
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
          const llmScheduledMatchedAgent = matchedAgentsByVaultIndex.some(
            (id) => rawDecisions[id].effort !== 'silent',
          );
          for (const matchedId of matchedAgentsByVaultIndex) {
            if (rawDecisions[matchedId].effort === 'silent') {
              rawDecisions[matchedId] = {
                effort: 'standard',
                thesis: getDomainSafeThesisFallback(matchedId, runtimeConfig),
              };
            }
          }
          // Only silence non-matching agents if this is a pure single-dimension price query OR if the LLM
          // router failed to schedule ANY of the vault-matched agents (context-momentum misrouting, e.g.
          // scheduling only Zoe on "If we take Flight B, can we check into Ryokan Gion?").
          if (isPureSingleDimensionPriceQuery || !llmScheduledMatchedAgent || asksArrivalOrCheckInFeasibility) {
            for (const id of AGENT_IDS) {
              if (!matchedAgentsByVaultIndex.includes(id)) {
                rawDecisions[id] = { effort: 'silent', thesis: '' };
              }
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
    resolveSpeakerDependencies(speakers, rawDecisions);

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
