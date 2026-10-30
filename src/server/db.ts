import Database from 'better-sqlite3';
import { randomUUID } from 'node:crypto';

export type MessageRole = 'user' | 'agent';
export type MessageStatus = 'streaming' | 'completed';
export type AgentId = 'maya' | 'liam' | 'zoe';
export type RuleScope = 'global' | AgentId;

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
  { name: 'currency', pattern: /\b(usd|jpy|eur|yen|dollars?|currency|currencies|prices?|costs?)\b/i },
  { name: 'format', pattern: /\b(bullets?|bulleted|numbered|paragraphs?|prose|tables?|json|markdown|lists?)\b/i },
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
  const words = text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOPWORDS.has(w));
  return new Set(words);
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

  constructor(dbPath = ':memory:') {
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
    this.initSchema();
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
    `);
  }

  reserveMessage(input: ReserveMessageInput): MessageRow {
    const id = input.id ?? randomUUID();
    const role = input.role;
    const agentId = input.agentId ?? null;
    const content = input.content ?? '';
    const status = input.status ?? (role === 'user' ? 'completed' : 'streaming');
    const createdAt = input.createdAt ?? Date.now();

    const stmt = this.db.prepare(`
      INSERT INTO messages (id, role, agent_id, content, status, created_at)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    const info = stmt.run(id, role, agentId, content, status, createdAt);

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
    const updateStmt = this.db.prepare(`
      UPDATE messages
      SET content = content || ?
      WHERE id = ?
    `);
    const res = updateStmt.run(delta, id);
    if (res.changes === 0) {
      throw new Error(`Message not found: ${id}`);
    }
    return this.getMessageById(id)!;
  }

  updateMessageContent(id: string, content: string): MessageRow {
    const updateStmt = this.db.prepare(`
      UPDATE messages
      SET content = ?
      WHERE id = ?
    `);
    const res = updateStmt.run(content, id);
    if (res.changes === 0) {
      throw new Error(`Message not found: ${id}`);
    }
    return this.getMessageById(id)!;
  }

  markMessageStatus(id: string, status: MessageStatus): MessageRow {
    const updateStmt = this.db.prepare(`
      UPDATE messages
      SET status = ?
      WHERE id = ?
    `);
    const res = updateStmt.run(status, id);
    if (res.changes === 0) {
      throw new Error(`Message not found: ${id}`);
    }
    return this.getMessageById(id)!;
  }

  getMessageById(id: string): MessageRow | undefined {
    const stmt = this.db.prepare(`
      SELECT seq, id, role, agent_id, content, status, created_at
      FROM messages
      WHERE id = ?
    `);
    return stmt.get(id) as MessageRow | undefined;
  }

  listMessages(): MessageRow[] {
    const stmt = this.db.prepare(`
      SELECT seq, id, role, agent_id, content, status, created_at
      FROM messages
      ORDER BY seq ASC
    `);
    return stmt.all() as MessageRow[];
  }

  addStandingRule(input: AddStandingRuleInput): StandingRuleRow {
    const id = input.id ?? randomUUID();
    const scope = input.scope;
    const rule = input.rule.trim();
    const createdAt = input.createdAt ?? Date.now();

    const tx = this.db.transaction(() => {
      if (input.supersedeRuleIds && input.supersedeRuleIds.length > 0) {
        const deactivateById = this.db.prepare(`
          UPDATE standing_rules SET active = 0 WHERE id = ?
        `);
        for (const oldId of input.supersedeRuleIds) {
          deactivateById.run(oldId);
        }
      }

      const existingInScope = this.db
        .prepare(`
          SELECT id, scope, rule, active, created_at
          FROM standing_rules
          WHERE scope = ? AND active = 1
        `)
        .all(scope) as StandingRuleRow[];

      const deactivateStmt = this.db.prepare(`
        UPDATE standing_rules SET active = 0 WHERE id = ?
      `);

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
          deactivateStmt.run(existing.id);
        }
      }

      this.db
        .prepare(`
          INSERT INTO standing_rules (id, scope, rule, active, created_at)
          VALUES (?, ?, ?, 1, ?)
        `)
        .run(id, scope, rule, createdAt);
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
    this.db.prepare(`UPDATE standing_rules SET active = 0 WHERE id = ?`).run(id);
  }

  listActiveStandingRules(agentId?: AgentId): StandingRuleRow[] {
    if (agentId) {
      const stmt = this.db.prepare(`
        SELECT id, scope, rule, active, created_at
        FROM standing_rules
        WHERE active = 1 AND (scope = 'global' OR scope = ?)
        ORDER BY created_at ASC, rowid ASC
      `);
      return stmt.all(agentId) as StandingRuleRow[];
    }

    const stmt = this.db.prepare(`
      SELECT id, scope, rule, active, created_at
      FROM standing_rules
      WHERE active = 1
      ORDER BY created_at ASC, rowid ASC
    `);
    return stmt.all() as StandingRuleRow[];
  }

  listAllStandingRules(): StandingRuleRow[] {
    const stmt = this.db.prepare(`
      SELECT id, scope, rule, active, created_at
      FROM standing_rules
      ORDER BY created_at ASC, rowid ASC
    `);
    return stmt.all() as StandingRuleRow[];
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
