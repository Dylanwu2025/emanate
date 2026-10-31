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
  'travel', 'plans', 'plan', 'between', 'recommend', 'stay',
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
  return overlap >= 3 && (ratioLater >= 0.75 || ratioEarlier >= 0.75);
}

/**
 * Checks if an agent's proposed thesis in a follow-up wave repeats what that same agent
 * already covered earlier in the current turn (either by thesis or by message content).
 */
export function isSelfRepetitiveForAgent(
  proposedThesis: string,
  priorThesesForAgent: string[],
  priorMessagesForAgent: string[],
): boolean {
  const trimmed = proposedThesis.trim();
  if (!trimmed) return true;

  // Reject meta-prompts where an agent merely asks the user to confirm/choose an option it just presented
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
      // If >=80% of the proposed thesis tokens were already stated in this agent's earlier message in the turn
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
  const isClearIntent =
    /\b(?:forget|clear|remove|drop|cancel|reset|stop\s+following)\b.*\b(?:standing\s+rules?|rules?|instructions?|jpy|yen|bullet\s+points?|uppercase)\b/i.test(
      lower,
    ) ||
    /\b(?:no\s+longer\s+need\s+to|stop\s+using\s+bullet|go\s+back\s+to\s+usd|switch\s+back\s+to\s+usd)\b/i.test(
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

  const hasPersistentCue =
    /\b(?:always|never|from\s+now\s+on|going\s+forward|in\s+all\s+(?:future\s+)?(?:responses|answers|messages|turns)|every\s+time|for\s+the\s+rest\s+of)\b/i.test(
      trimmed,
    ) ||
    /\b(?:convert\s+all\s+prices|state\s+all\s+prices|quote\s+(?:all\s+)?prices|use\s+japanese\s+yen|in\s+jpy\s+instead|to\s+japanese\s+yen|in\s+bullet\s+points|use\s+bullet\s+points|answer\s+in\s+bullet|keep\s+(?:all\s+|your\s+)?(?:answers|responses)\s+(?:to|under)|in\s+uppercase|all\s+caps)\b/i.test(
      trimmed,
    );

  if (!hasPersistentCue) {
    return null;
  }

  const scope: RuleScope = mentionedAgents.length === 1 ? mentionedAgents[0] : 'global';
  const cleanedRule = trimmed
    .replace(/^(?:@?(?:maya|liam|zoe)\s*[,:—-]\s*)+/i, '')
    .trim();

  return {
    scope,
    rule: cleanedRule || trimmed,
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
  const standingRule = clearRuleScope ? null : extractStandingRuleDeterministic(trimmed, mentioned);

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
4. "standingRule": If the instruction should persist for future turns (e.g. currency preference, bullet points, length cap, "always...", "from now on...", "convert all prices..."), return { "scope": "global" | "maya" | "liam" | "zoe", "rule": string }, else null.
5. "clearRuleScope": If the user asks to forget/clear/remove a standing rule, return the scope ("global" | "maya" | "liam" | "zoe"), else null.

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
        raw.clearRuleScope && VALID_SCOPES.has(raw.clearRuleScope)
          ? raw.clearRuleScope
          : fallback.clearRuleScope;

      const standingRule = clearRuleScope
        ? null
        : normalizeStandingRule(raw.standingRule) ?? fallback.standingRule;

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
      ? `You are the A2A Follow-Up Wave Router for a 3-agent Kyoto & Osaka trip planning group chat:
- maya (Maya — Budget, Flights & Lodging: knows the $4,800 budget, Flight A 6:45 AM $650 vs Flight B 11:30 AM $820, Ryokan Gion $380/night with strict 6:00 PM check-in cutoff vs Hotel Granvia $290/night 24h check-in, Cross Hotel Osaka $260/night).
- liam (Liam — Local Guide for Food, Sights & Transit: knows KIX to Gion takes 2h 15m, KIX to Kyoto Station takes 1h 35m, Nishiki Market 11 AM–2 PM has heavy shared-grill shellfish, Arashiyama Tofu Kaiseki is 100% vegetarian & shellfish-free until 7:30 PM, sight crowd windows).
- zoe (Zoe — Group Dynamics, Dietary & Pacing: knows Sam has a severe shellfish allergy and boat/bus motion sickness, Priya is strict vegetarian, and the group has a strict rule against waking before 7:30 AM which rules out 6:45 AM Flight A).

Agents who just spoke in Wave ${waveIndex - 1}: ${JSON.stringify(inferredPrevSpeakers)}

CRITICAL RULES FOR A2A FOLLOW-UP WAVES:
1. NEVER allow an agent to reply to itself or repeat its own recommendation or call-to-action! If only one agent spoke in Wave ${waveIndex - 1} (e.g. ${JSON.stringify(inferredPrevSpeakers)}), that agent MUST be "silent" in Wave ${waveIndex}.
2. Questions directed to the USER (e.g. "Please let me know which flight you prefer!", "What do you all think?" when no peer has conflicting/missing private facts) are NOT triggers for the same agent to speak again. Wait for the user by setting "isResolved": true and all agents to "silent".
3. An agent may ONLY speak in Wave ${waveIndex} if a DIFFERENT agent in Wave ${waveIndex - 1}:
   (a) directly asked them a question (e.g. "@Liam how long is transit?"), OR
   (b) mentioned a specific option/fact that triggers a constraint in THIS agent's private vault that has NOT been stated yet in the current turn (for example: if Maya mentioned Flight A departing at 6:45 AM and Zoe hasn't vetoed it yet based on the group's no-waking-before-7:30-AM rule, Zoe MUST speak now; or if Liam revealed 2h 15m transit makes Ryokan Gion's 6:00 PM cutoff too tight, Maya can pivot to Hotel Granvia).
4. Once all cross-agent constraints for the user's question have been surfaced, set "isResolved": true and all agents to "silent".

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
- zoe (Zoe — Group Dynamics, Dietary & Pacing: private vault has Sam's severe shellfish allergy & bus/boat motion sickness, Priya's strict vegetarianism, and the group's no-waking-before-7:30-AM rule which rules out 6:45 AM Flight A).

Rules for routing Wave 0:
1. Silence Gate: If an agent's domain is not relevant to the user's message, set their effort to "silent" and thesis to "".
2. Cross-Domain Awareness:
   - When the user asks about flights or travel plans, Maya should cover the flight options/prices AND Zoe should also speak (or follow up) if 6:45 AM Flight A is on the table, since Zoe's vault has the group's no-waking-before-7:30-AM rule.
   - When the user asks about food/dining (like Nishiki Market), both Liam (food/hours) and Zoe (Sam's shellfish allergy & Priya's vegetarian diet) are relevant.
   - When the user asks a narrow single-agent question (e.g. "@Maya what is the total budget?"), only that agent speaks.
3. Effort Calibration:
   - "light": Simple factual check or quick confirmation (1 short sentence).
   - "standard": Moderate multi-fact answer (2–4 sentences).
   - "deep": Complex trade-off or full itinerary/budget breakdown.
4. Distinct Theses: Each speaking agent MUST have a distinct 1-line "thesis" focused strictly on their own private domain so agents never repeat each other.
5. Standing Rule Extraction: If the user message establishes a persistent instruction for how agents should respond (e.g. "Always answer in bullet points", "Maya, quote all prices in JPY from now on"), return it in "standingRule": { "scope": "global" | "maya" | "liam" | "zoe", "rule": "..." }, otherwise null.

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

    // Enforce hard A2A guards on follow-up waves (Wave >= 1):
    // 1. No consecutive self-reply: if an agent was the ONLY speaker in Wave N-1,
    //    no peer has spoken since their last message, so they MUST NOT speak again in Wave N!
    // 2. Self-deduplication: if an agent already spoke in this turn and their proposed thesis
    //    repeats their earlier thesis/message (or is just "confirm preferred flight"), silence them.
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
    }

    const referenceText =
      latestUserMessage ??
      (transcript.length > 0 ? transcript[transcript.length - 1].content : '');

    const speakerOrder = computeDeterministicSpeakerOrder(referenceText);
    const { decisions, speakers } = applyAsymmetricDedup(rawDecisions, speakerOrder);

    const isResolved = isFollowUpWave
      ? Boolean(raw.isResolved) || speakers.length === 0
      : speakers.length === 0;

    if (isFollowUpWave && isResolved) {
      for (const id of AGENT_IDS) {
        decisions[id] = { effort: 'silent', thesis: '' };
      }
      speakers.length = 0;
    }

    const mentionedInUserMsg = extractMentionedAgents(referenceText);
    const clearRuleScope = isFollowUpWave
      ? null
      : (raw.clearRuleScope && VALID_SCOPES.has(raw.clearRuleScope)
          ? raw.clearRuleScope
          : detectRuleClearScope(referenceText, mentionedInUserMsg));

    const standingRule =
      isFollowUpWave || clearRuleScope
        ? null
        : normalizeStandingRule(raw.standingRule) ??
          extractStandingRuleDeterministic(referenceText, mentionedInUserMsg);

    return {
      waveIndex,
      isResolved,
      decisions,
      speakers,
      standingRule,
      clearRuleScope,
    };
  }
}
