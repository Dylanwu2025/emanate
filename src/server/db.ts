import Database from 'better-sqlite3';
import { randomUUID } from 'node:crypto';

export type MessageRole = 'user' | 'agent';
export type MessageStatus = 'streaming' | 'completed';
export type AgentId = 'maya' | 'liam' | 'zoe';
export type RuleScope = 'global' | AgentId;
export type ClearRuleScope = RuleScope | 'all';

export interface MessageRow {
  seq: number;
  id: string;
  role: MessageRole;
  agent_id: AgentId | null;
  content: string;
  status: MessageStatus;
  created_at: number;
}

export interface StandingRuleRow {
  id: string;
  scope: RuleScope;
  rule: string;
  active: number;
  created_at: number;
}

export interface ReserveMessageInput {
  id?: string;
  role: MessageRole;
  agentId?: AgentId | null;
  content?: string;
  status?: MessageStatus;
  createdAt?: number;
}

export interface AddStandingRuleInput {
  id?: string;
  scope: RuleScope;
  rule: string;
  supersedeRuleIds?: string[];
  supersedeMatchingPattern?: RegExp | ((existingRule: StandingRuleRow) => boolean);
  createdAt?: number;
}

const STOPWORDS = new Set([
  'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
  'as', 'is', 'are', 'be', 'all', 'always', 'never', 'only', 'please', 'must',
  'should', 'use', 'using', 'keep', 'make', 'write', 'respond', 'answer', 'reply',
  'state', 'give', 'format', 'responses', 'answers', 'replies', 'messages', 'instead'
]);

const TOPIC_BUCKETS: Array<{ name: string; pattern: RegExp }> = [
  { name: 'currency', pattern: /\b(usd|jpy|eur|yen|euros?|dollars?|currency|currencies|prices?|costs?)\b/i },
  {
    name: 'format',
    pattern:
      /\b(bullets?|bulleted|numbered|paragraphs?|prose|tables?|json|markdown|lists?|haikus?|poems?|rhymes?|normal\s+sentences?|plain\s+text|uppercase|lowercase|all\s+caps)\b/i,
  },
  {
    name: 'prefix_suffix',
    pattern:
      /\b((?:start|end|begin|finish)(?:ing)?\s+(?:every|all|each)?\s*(?:replies|responses|messages)?\s*with|prefix|suffix|tag|header)\b|\[[^\]]+\]/i,
  },
  { name: 'length', pattern: /\b(short|brief|concise|detailed|long|sentences?|words?|one-liner)\b/i },
  { name: 'language', pattern: /\b(english|japanese|spanish|french|bilingual|translate|language)\b/i },
  { name: 'tone', pattern: /\b(formal|casual|friendly|playful|serious|emoji|emojis|enthusiastic)\b/i },
  { name: 'time_format', pattern: /\b(24-hour|12-hour|am\/pm|military time|timezone|jst)\b/i },
];

function extractTopics(text: string): Set<string> {
  const matched = new Set<string>();
  for (const bucket of TOPIC_BUCKETS) {
    if (bucket.pattern.test(text)) {
      matched.add(bucket.name);
    }
  }
  return matched;
}

function extractSignificantTokens(text: string): Set<string> {
  const lower = text.toLowerCase();
  const tokens = new Set<string>();
  const bracketMatches = lower.match(/\[[^\]]+\]/g);
  if (bracketMatches) {
    for (const tag of bracketMatches) {
      tokens.add(tag);
    }
  }
  const words = lower
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length >= 2 && !STOPWORDS.has(w));
  for (const w of words) {
    tokens.add(w);
  }
  return tokens;
}

export function rulesConflict(existingRuleText: string, newRuleText: string): boolean {
  const normExisting = existingRuleText.trim().toLowerCase();
  const normNew = newRuleText.trim().toLowerCase();
  if (normExisting === normNew) {
    return true;
  }

  const existingTopics = extractTopics(normExisting);
  const newTopics = extractTopics(normNew);
  for (const topic of existingTopics) {
    if (newTopics.has(topic)) {
      return true;
    }
  }

  const existingTokens = extractSignificantTokens(normExisting);
  const newTokens = extractSignificantTokens(normNew);
  if (existingTokens.size > 0 && newTokens.size > 0) {
    let overlap = 0;
    for (const token of existingTokens) {
      if (newTokens.has(token)) {
        overlap++;
      }
    }
    const minSize = Math.min(existingTokens.size, newTokens.size);
    if (overlap >= 2 || (minSize === 1 && overlap === 1)) {
      return true;
    }
  }

  return false;
}

export class EmanateStore {
  public readonly db: Database.Database;

  private readonly stmtInsertMessage: Database.Statement;
  private readonly stmtAppendMessageDelta: Database.Statement;
  private readonly stmtUpdateMessageContent: Database.Statement;
  private readonly stmtMarkMessageStatus: Database.Statement;
  private readonly stmtGetMessageById: Database.Statement;
  private readonly stmtListMessages: Database.Statement;
  private readonly stmtDeactivateRuleById: Database.Statement;
  private readonly stmtSelectActiveRulesByScope: Database.Statement;
  private readonly stmtInsertStandingRule: Database.Statement;
  private readonly stmtClearAllStandingRules: Database.Statement;
  private readonly stmtClearStandingRulesByScope: Database.Statement;
  private readonly stmtListActiveRulesForAgent: Database.Statement;
  private readonly stmtListAllActiveRules: Database.Statement;
  private readonly stmtListAllRules: Database.Statement;
  private readonly stmtGetSetting: Database.Statement;
  private readonly stmtSetSetting: Database.Statement;

  constructor(dbPath = ':memory:') {
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
    this.initSchema();

    this.stmtInsertMessage = this.db.prepare(`
      INSERT INTO messages (id, role, agent_id, content, status, created_at)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    this.stmtAppendMessageDelta = this.db.prepare(`
      UPDATE messages
      SET content = content || ?
      WHERE id = ?
      RETURNING seq, id, role, agent_id, content, status, created_at
    `);
    this.stmtUpdateMessageContent = this.db.prepare(`
      UPDATE messages
      SET content = ?
      WHERE id = ?
      RETURNING seq, id, role, agent_id, content, status, created_at
    `);
    this.stmtMarkMessageStatus = this.db.prepare(`
      UPDATE messages
      SET status = ?
      WHERE id = ?
      RETURNING seq, id, role, agent_id, content, status, created_at
    `);
    this.stmtGetMessageById = this.db.prepare(`
      SELECT seq, id, role, agent_id, content, status, created_at
      FROM messages
      WHERE id = ?
    `);
    this.stmtListMessages = this.db.prepare(`
      SELECT seq, id, role, agent_id, content, status, created_at
      FROM messages
      ORDER BY seq ASC
    `);
    this.stmtDeactivateRuleById = this.db.prepare(`
      UPDATE standing_rules SET active = 0 WHERE id = ?
    `);
    this.stmtSelectActiveRulesByScope = this.db.prepare(`
      SELECT id, scope, rule, active, created_at
      FROM standing_rules
      WHERE scope = ? AND active = 1
    `);
    this.stmtInsertStandingRule = this.db.prepare(`
      INSERT INTO standing_rules (id, scope, rule, active, created_at)
      VALUES (?, ?, ?, 1, ?)
    `);
    this.stmtClearAllStandingRules = this.db.prepare(`
      UPDATE standing_rules SET active = 0 WHERE active = 1
    `);
    this.stmtClearStandingRulesByScope = this.db.prepare(`
      UPDATE standing_rules SET active = 0 WHERE active = 1 AND scope = ?
    `);
    this.stmtListActiveRulesForAgent = this.db.prepare(`
      SELECT id, scope, rule, active, created_at
      FROM standing_rules
      WHERE active = 1 AND (scope = 'global' OR scope = ?)
      ORDER BY created_at ASC, rowid ASC
    `);
    this.stmtListAllActiveRules = this.db.prepare(`
      SELECT id, scope, rule, active, created_at
      FROM standing_rules
      WHERE active = 1
      ORDER BY created_at ASC, rowid ASC
    `);
    this.stmtListAllRules = this.db.prepare(`
      SELECT id, scope, rule, active, created_at
      FROM standing_rules
      ORDER BY created_at ASC, rowid ASC
    `);
    this.stmtGetSetting = this.db.prepare(`SELECT value FROM settings WHERE key = ?`);
    this.stmtSetSetting = this.db.prepare(`
      INSERT INTO settings (key, value) VALUES (?, ?)
      ON CONFLICT(key) DO UPDATE SET value = excluded.value
    `);
  }

  private initSchema(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS messages (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        id TEXT UNIQUE NOT NULL,
        role TEXT NOT NULL,
        agent_id TEXT,
        content TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS standing_rules (
        id TEXT PRIMARY KEY,
        scope TEXT NOT NULL,
        rule TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        created_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );
    `);
  }

  reserveMessage(input: ReserveMessageInput): MessageRow {
    const id = input.id ?? randomUUID();
    const role = input.role;
    const agentId = input.agentId ?? null;
    const content = input.content ?? '';
    const status = input.status ?? (role === 'user' ? 'completed' : 'streaming');
    const createdAt = input.createdAt ?? Date.now();

    const info = this.stmtInsertMessage.run(id, role, agentId, content, status, createdAt);

    return {
      seq: Number(info.lastInsertRowid),
      id,
      role,
      agent_id: agentId,
      content,
      status,
      created_at: createdAt,
    };
  }

  appendMessageDelta(id: string, delta: string): MessageRow {
    const row = this.stmtAppendMessageDelta.get(delta, id) as MessageRow | undefined;
    if (!row) {
      throw new Error(`Message not found: ${id}`);
    }
    return row;
  }

  updateMessageContent(id: string, content: string): MessageRow {
    const row = this.stmtUpdateMessageContent.get(content, id) as MessageRow | undefined;
    if (!row) {
      throw new Error(`Message not found: ${id}`);
    }
    return row;
  }

  markMessageStatus(id: string, status: MessageStatus): MessageRow {
    const row = this.stmtMarkMessageStatus.get(status, id) as MessageRow | undefined;
    if (!row) {
      throw new Error(`Message not found: ${id}`);
    }
    return row;
  }

  getMessageById(id: string): MessageRow | undefined {
    return this.stmtGetMessageById.get(id) as MessageRow | undefined;
  }

  listMessages(): MessageRow[] {
    return this.stmtListMessages.all() as MessageRow[];
  }

  addStandingRule(input: AddStandingRuleInput): StandingRuleRow {
    const id = input.id ?? randomUUID();
    const scope = input.scope;
    const rule = input.rule.trim();
    const createdAt = input.createdAt ?? Date.now();

    const tx = this.db.transaction(() => {
      if (input.supersedeRuleIds && input.supersedeRuleIds.length > 0) {
        for (const oldId of input.supersedeRuleIds) {
          this.stmtDeactivateRuleById.run(oldId);
        }
      }

      const existingInScope = this.stmtSelectActiveRulesByScope.all(scope) as StandingRuleRow[];

      for (const existing of existingInScope) {
        let shouldSupersede = false;
        if (input.supersedeMatchingPattern instanceof RegExp) {
          shouldSupersede = input.supersedeMatchingPattern.test(existing.rule);
        } else if (typeof input.supersedeMatchingPattern === 'function') {
          shouldSupersede = input.supersedeMatchingPattern(existing);
        } else {
          shouldSupersede = rulesConflict(existing.rule, rule);
        }

        if (shouldSupersede) {
          this.stmtDeactivateRuleById.run(existing.id);
        }
      }

      this.stmtInsertStandingRule.run(id, scope, rule, createdAt);
    });

    tx();

    return {
      id,
      scope,
      rule,
      active: 1,
      created_at: createdAt,
    };
  }

  supersedeStandingRule(oldRuleId: string, newRule: Omit<AddStandingRuleInput, 'supersedeRuleIds'>): StandingRuleRow {
    return this.addStandingRule({
      ...newRule,
      supersedeRuleIds: [oldRuleId],
    });
  }

  deactivateStandingRule(id: string): void {
    this.stmtDeactivateRuleById.run(id);
  }

  deactivateRulesByIds(ids: string[]): number {
    if (!ids || ids.length === 0) return 0;
    let count = 0;
    const tx = this.db.transaction(() => {
      for (const id of ids) {
        if (typeof id === 'string' && id.trim()) {
          const res = this.stmtDeactivateRuleById.run(id.trim());
          count += res.changes;
        }
      }
    });
    tx();
    return count;
  }

  clearStandingRules(scope?: ClearRuleScope): number {
    if (!scope || scope === 'all') {
      const res = this.stmtClearAllStandingRules.run();
      return res.changes;
    }
    const res = this.stmtClearStandingRulesByScope.run(scope);
    return res.changes;
  }

  listActiveStandingRules(agentId?: AgentId): StandingRuleRow[] {
    if (agentId) {
      return this.stmtListActiveRulesForAgent.all(agentId) as StandingRuleRow[];
    }
    return this.stmtListAllActiveRules.all() as StandingRuleRow[];
  }

  listAllStandingRules(): StandingRuleRow[] {
    return this.stmtListAllRules.all() as StandingRuleRow[];
  }

  getSetting(key: string): string | undefined {
    const row = this.stmtGetSetting.get(key) as { value: string } | undefined;
    return row?.value;
  }

  setSetting(key: string, value: string): void {
    this.stmtSetSetting.run(key, value);
  }

  reset(): void {
    this.db.exec(`
      DELETE FROM messages;
      DELETE FROM standing_rules;
      DELETE FROM sqlite_sequence WHERE name = 'messages';
    `);
  }

  close(): void {
    this.db.close();
  }
}
