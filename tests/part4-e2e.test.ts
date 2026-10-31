import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { EmanateStore } from '../src/server/db.js';
import {
  createLLMProvider,
  ScriptableMockLLMProvider,
} from '../src/server/llm.js';
import { createAppServer, type AppServerInstance } from '../src/server/index.js';
import type { OrchestratorSSEEvent, PublicWireMessage } from '../src/server/orchestrator.js';

function waitForCondition(
  predicate: () => boolean,
  timeoutMs = 8000,
  intervalMs = 10,
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (predicate()) {
      resolve();
      return;
    }
    const start = Date.now();
    const timer = setInterval(() => {
      if (predicate()) {
        clearInterval(timer);
        resolve();
      } else if (Date.now() - start > timeoutMs) {
        clearInterval(timer);
        reject(new Error('Timed out waiting for condition'));
      }
    }, intervalMs);
  });
}

const ALLOWED_PUBLIC_MESSAGE_KEYS = [
  'agentId',
  'agentName',
  'content',
  'createdAt',
  'id',
  'role',
  'seq',
  'status',
].sort();

const FORBIDDEN_WIRE_KEYS = [
  'effort',
  'thesis',
  'peerTheses',
  'priorSpeakerTheses',
  'epoch',
  'generationEpoch',
  'standingRule',
  'standingRules',
  'steeringClassification',
  'pendingContinuation',
];

describe('Part 4: Group Chat Web UI, Documentation & End-to-End Verification', () => {
  let store: EmanateStore;
  let mockLLM: ScriptableMockLLMProvider;
  let app: AppServerInstance;
  let baseUrl: string;

  beforeEach(async () => {
    store = new EmanateStore(':memory:');
    mockLLM = new ScriptableMockLLMProvider();
    app = createAppServer({ store, llm: mockLLM });
    const bound = await app.listen(0);
    baseUrl = bound.url;
  });

  afterEach(async () => {
    await app.close();
    store.close();
  });

  describe('a) Static Group Chat Web UI & Zero Exposed Mechanism Contract', () => {
    it('serves index.html, styles.css, and app.js on the same port and contains zero forbidden mechanism badges or logs', async () => {
      const htmlRes = await fetch(`${baseUrl}/`);
      expect(htmlRes.status).toBe(200);
      expect(htmlRes.headers.get('content-type')).toContain('text/html');
      const html = await htmlRes.text();

      const cssRes = await fetch(`${baseUrl}/styles.css`);
      expect(cssRes.status).toBe(200);
      expect(cssRes.headers.get('content-type')).toContain('text/css');
      const css = await cssRes.text();

      const jsRes = await fetch(`${baseUrl}/app.js`);
      expect(jsRes.status).toBe(200);
      expect(jsRes.headers.get('content-type')).toContain('application/javascript');
      const js = await jsRes.text();

      // Verify essential group chat UI elements exist
      expect(html).toContain('Kyoto &amp; Osaka Trip');
      expect(html).toContain('Maya');
      expect(html).toContain('Liam');
      expect(html).toContain('Zoe');
      expect(html).toContain('id="composer-input"');
      expect(html).toContain('id="stop-btn"');
      expect(html).toContain('id="reset-btn"');
      expect(html).toContain('data-mention="@Maya "');
      expect(html).toContain('data-mention="@Liam "');
      expect(html).toContain('data-mention="@Zoe "');

      // Verify composer input is NEVER disabled during streaming in app.js
      expect(js).not.toMatch(/inputEl\.disabled\s*=\s*true/);

      // Verify ZERO exposed mechanism strings in HTML, CSS, or client JS
      const combinedClientSource = `${html}\n${css}\n${js}`;
      const forbiddenMechanismPatterns = [
        /steering\s*log/i,
        /redirected\s*badge/i,
        /steered\s*badge/i,
        /effort\s*badge/i,
        /effort\s*tier/i,
        /applying\s*standing\s*instruction/i,
        /standing\s*rule\s*active/i,
        /generationEpoch/i,
        /peerTheses/i,
      ];
      for (const pattern of forbiddenMechanismPatterns) {
        expect(combinedClientSource).not.toMatch(pattern);
      }

      // Verify README.md and ARCHITECTURE.md exist in repo root and are comprehensive
      const repoRoot = join(process.cwd());
      const readmeText = readFileSync(join(repoRoot, 'README.md'), 'utf-8');
      const archText = readFileSync(join(repoRoot, 'ARCHITECTURE.md'), 'utf-8');
      expect(readmeText).toContain('LLM_PROVIDER');
      expect(readmeText).toContain('gpt-4o-mini');
      expect(readmeText).toContain('ARCHITECTURE.md');
      expect(archText).toContain('sequenceDiagram');
      expect(archText).toContain('standing_rules');
    });
  });

  describe('b) End-to-End Multi-Wave + Mid-Stream Steering + Reload Ordering Consistency', () => {
    it('maintains identical seq ASC ordering between live SSE stream reconciliation and full page reload (GET /api/state)', async () => {
      const sseAbort = new AbortController();
      const sseEvents: OrchestratorSSEEvent[] = [];

      // Simulate browser client state reconciler (matching src/client/app.js logic)
      const clientMessagesById = new Map<string, PublicWireMessage>();

      const sseRes = await fetch(`${baseUrl}/api/stream`, { signal: sseAbort.signal });
      const reader = sseRes.body!.getReader();
      const decoder = new TextDecoder();

      const sseReaderTask = (async () => {
        let buffer = '';
        try {
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';
            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed.startsWith('data:')) continue;
              const ev = JSON.parse(trimmed.slice(5).trim()) as OrchestratorSSEEvent;
              sseEvents.push(ev);

              if (ev.type === 'message.created') {
                clientMessagesById.set(ev.message.id, { ...ev.message });
              } else if (ev.type === 'message.delta') {
                const msg = clientMessagesById.get(ev.id);
                if (msg) {
                  msg.content = ev.replace ? ev.delta : msg.content + ev.delta;
                }
              } else if (ev.type === 'message.done') {
                clientMessagesById.set(ev.message.id, { ...ev.message });
              }
            }
          }
        } catch {
          // Closed
        }
      })();

      // Wave 0: Maya, Liam, and Zoe all speak concurrently at different speeds,
      // and we steer Maya mid-stream!
      mockLLM.enqueueJSON({
        response: {
          maya: { effort: 'standard', thesis: 'Compare Flight A and B prices' },
          liam: { effort: 'standard', thesis: 'Provide KIX to Gion 2h 15m transit time' },
          zoe: { effort: 'light', thesis: 'State 7:30 AM group wakeup rule' },
        },
      });

      mockLLM.enqueueStream({
        match: (opts) =>
          opts.messages[0].content.includes('You are Maya') &&
          !opts.messages[0].content.includes('Revising Your In-Progress Message'),
        chunks: ['Flight A is $650 ', 'and Flight B is $820 ', 'in USD...'],
        delayMs: 35,
      });

      mockLLM.enqueueStream({
        match: (opts) => opts.messages[0].content.includes('You are Liam'),
        chunks: ['KIX to Gion takes ', '2 hours 15 minutes door-to-door.'],
        delayMs: 15,
      });

      mockLLM.enqueueStream({
        match: (opts) => opts.messages[0].content.includes('You are Zoe'),
        chunks: ['Nobody in the group wakes before 7:30 AM, so 6:45 AM Flight A is out.'],
        delayMs: 5,
      });

      mockLLM.enqueueStream({
        match: (opts) =>
          opts.messages[0].content.includes('You are Maya') &&
          opts.messages[0].content.includes('Revising Your In-Progress Message'),
        chunks: ['Flight A is ¥97,500 ', 'and Flight B is ¥123,000 in JPY.'],
        delayMs: 10,
      });

      // Kick off initial POST /api/messages
      const post1Promise = fetch(`${baseUrl}/api/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: 'How should we handle Day 1 flights and airport transit?',
        }),
      });

      // Wait until Maya has flushed at least 1 chunk over SSE
      await waitForCondition(() => {
        const maya = app.orchestrator.getActiveStreams().get('maya');
        return Boolean(maya && maya.flushedContent.includes('Flight A is $650 '));
      });

      // Send mid-stream steering redirect for Maya over HTTP POST /api/messages
      const post2Promise = fetch(`${baseUrl}/api/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: '@Maya convert prices to JPY instead',
        }),
      });

      const [res1, res2] = await Promise.all([post1Promise, post2Promise]);
      expect(res1.status).toBe(200);
      expect(res2.status).toBe(200);

      await waitForCondition(() => sseEvents.some((e) => e.type === 'turn.idle'));

      // Live client sorted state (from SSE stream)
      const liveClientSorted = Array.from(clientMessagesById.values()).sort(
        (a, b) => a.seq - b.seq,
      );

      // Simulated Browser Reload: GET /api/state
      const reloadRes = await fetch(`${baseUrl}/api/state`);
      expect(reloadRes.status).toBe(200);
      const reloadData = (await reloadRes.json()) as { messages: PublicWireMessage[] };

      // Canonical reload state MUST match live SSE reconciled state in exact seq ASC order!
      expect(reloadData.messages).toEqual(liveClientSorted);
      expect(reloadData.messages.map((m) => ({ seq: m.seq, role: m.role, agentId: m.agentId }))).toEqual([
        { seq: 1, role: 'user', agentId: null },
        { seq: 2, role: 'agent', agentId: 'maya' },
        { seq: 3, role: 'agent', agentId: 'liam' },
        { seq: 4, role: 'agent', agentId: 'zoe' },
        { seq: 5, role: 'user', agentId: null },
      ]);

      expect(reloadData.messages[1].content).toBe(
        'Flight A is ¥97,500 and Flight B is ¥123,000 in JPY.',
      );

      sseAbort.abort();
      await sseReaderTask;
    });
  });

  describe('c) Sanitized Wire Format Audit (Zero Internal Metadata Leaked)', () => {
    it('ensures every SSE event and GET /api/state payload exposes strictly public chat fields', async () => {
      const sseAbort = new AbortController();
      const rawSSEPayloads: Array<Record<string, unknown>> = [];

      const sseRes = await fetch(`${baseUrl}/api/stream`, { signal: sseAbort.signal });
      const reader = sseRes.body!.getReader();
      const decoder = new TextDecoder();

      const sseTask = (async () => {
        let buf = '';
        try {
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            const lines = buf.split('\n');
            buf = lines.pop() ?? '';
            for (const line of lines) {
              const trimmed = line.trim();
              if (trimmed.startsWith('data:')) {
                rawSSEPayloads.push(JSON.parse(trimmed.slice(5).trim()));
              }
            }
          }
        } catch {
          // Closed
        }
      })();

      mockLLM.enqueueJSON({
        response: {
          maya: { effort: 'light', thesis: 'State budget' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
          standingRule: { scope: 'global', rule: 'Use bullet points' },
        },
      });
      mockLLM.enqueueStream({
        chunks: ['- Group budget is $4,800 USD.'],
        delayMs: 2,
      });
      mockLLM.enqueueJSON({
        response: {
          isResolved: true,
          maya: { effort: 'silent', thesis: '' },
          liam: { effort: 'silent', thesis: '' },
          zoe: { effort: 'silent', thesis: '' },
        },
      });

      const postRes = await fetch(`${baseUrl}/api/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: 'Always use bullet points. What is the budget?' }),
      });
      const postBody = (await postRes.json()) as Record<string, unknown>;

      // Verify POST response has no leaked internal keys
      for (const forbidden of FORBIDDEN_WIRE_KEYS) {
        expect(postBody).not.toHaveProperty(forbidden);
      }

      const stateRes = await fetch(`${baseUrl}/api/state`);
      const stateBody = (await stateRes.json()) as {
        messages: Array<Record<string, unknown>>;
      };
      expect(Object.keys(stateBody)).toEqual(['messages']);

      for (const msg of stateBody.messages) {
        expect(Object.keys(msg).sort()).toEqual(ALLOWED_PUBLIC_MESSAGE_KEYS);
        for (const forbidden of FORBIDDEN_WIRE_KEYS) {
          expect(msg).not.toHaveProperty(forbidden);
        }
      }

      await waitForCondition(() => rawSSEPayloads.some((p) => p.type === 'turn.idle'));

      for (const ev of rawSSEPayloads) {
        for (const forbidden of FORBIDDEN_WIRE_KEYS) {
          expect(ev).not.toHaveProperty(forbidden);
        }
        if (ev.type === 'message.created' || ev.type === 'message.done') {
          const msgObj = ev.message as Record<string, unknown>;
          expect(Object.keys(msgObj).sort()).toEqual(ALLOWED_PUBLIC_MESSAGE_KEYS);
        }
      }

      sseAbort.abort();
      await sseTask;
    });
  });

  describe('d) Live OpenAI E2E Server Turn & Reload Check', () => {
    it(
      'executes a live end-to-end turn over HTTP + SSE with gpt-4o-mini and verifies GET /api/state reload consistency',
      async () => {
        const liveLLM = createLLMProvider();
        if (!liveLLM.config.apiKey) {
          return;
        }

        const liveStore = new EmanateStore(':memory:');
        const liveApp = createAppServer({
          store: liveStore,
          llm: liveLLM,
          streamPaceMs: 10,
        });
        const { url: liveUrl } = await liveApp.listen(0);

        try {
          const postRes = await fetch(`${liveUrl}/api/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              content:
                '@Maya and @Zoe: Can we take the 6:45 AM Flight A to save money, and how much does it cost?',
            }),
          });
          expect(postRes.status).toBe(200);

          const stateRes = await fetch(`${liveUrl}/api/state`);
          expect(stateRes.status).toBe(200);
          const stateData = (await stateRes.json()) as { messages: PublicWireMessage[] };

          expect(stateData.messages.length).toBeGreaterThanOrEqual(3);
          const seqs = stateData.messages.map((m) => m.seq);
          expect(seqs).toEqual([...seqs].sort((a, b) => a - b));

          const combined = stateData.messages.map((m) => m.content).join('\n');
          // Maya knows Flight A is $650; Zoe knows the group cannot wake before 7:30 AM
          expect(combined).toMatch(/650/);
          expect(combined).toMatch(/7:30[\s-]*AM|wak(?:e|ing)|early/i);
        } finally {
          await liveApp.close();
          liveStore.close();
        }
      },
      45000,
    );

    it(
      'works seamlessly when private markdown vaults are swapped to a completely different scenario (B2B Cloud Architecture Launch)',
      async () => {
        const liveLLM = createLLMProvider();
        if (!liveLLM.config.apiKey) {
          return;
        }

        const customStore = new EmanateStore(':memory:');
        const customApp = createAppServer({
          store: customStore,
          llm: liveLLM,
          streamPaceMs: 10,
          runtimeConfig: {
            profileOverrides: {
              maya: {
                roleTitle: 'Cloud FinOps & Infrastructure Budget',
                personality: 'Direct, cost-focused FinOps lead.',
              },
              liam: {
                roleTitle: 'Database & Latency Architect',
                personality: 'Performance-obsessed systems engineer.',
              },
              zoe: {
                roleTitle: 'Security, HIPAA & Compliance Officer',
                personality: 'Strict compliance and data residency specialist.',
              },
            },
            vaultOverrides: {
              maya: `# Maya — Private Vault (Cloud FinOps)
## Monthly Infrastructure Budget
- Total monthly budget cap: $15,000/month.
## Cluster Options
1. **Cluster Alpha (Multi-Tenant Shared GPU)**: $9,000/month, deployed in us-west-1.
2. **Cluster Beta (Dedicated Single-Tenant H100)**: $14,200/month, deployed in eu-central-1.`,
              liam: `# Liam — Private Vault (Database & Latency)
## Replication & Latency Benchmarks
- **Cluster Alpha (us-west-1)** has p99 latency of 180ms for European users and uses unencrypted shared memory buffers.
- **Cluster Beta (eu-central-1)** has p99 latency of 22ms and supports dedicated NVMe encryption.`,
              zoe: `# Zoe — Private Vault (Security & Compliance)
## Non-Negotiable Compliance Rules
- **EU Data Residency & Isolation**: Our enterprise healthcare contract strictly forbids multi-tenant shared GPUs or hosting outside the EU (us-west-1 is strictly vetoed). All workloads must run on dedicated single-tenant hardware in eu-central-1.`,
            },
          },
        });

        const { url: customUrl } = await customApp.listen(0);
        try {
          const postRes = await fetch(`${customUrl}/api/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              content:
                'Should we deploy on Cluster Alpha to save money, or do our latency and compliance rules require Cluster Beta?',
            }),
          });
          expect(postRes.status).toBe(200);

          const stateRes = await fetch(`${customUrl}/api/state`);
          const stateData = (await stateRes.json()) as { messages: PublicWireMessage[] };

          expect(stateData.messages.length).toBeGreaterThanOrEqual(3);
          const combined = stateData.messages.map((m) => `[${m.agentName}]: ${m.content}`).join('\n');

          // Verify synthesis across the swapped non-travel vaults:
          // Maya: $9,000 / $14,200 / $15,000
          expect(combined).toMatch(/9,000|14,200|15,000/);
          // Liam or Zoe: eu-central-1 / us-west-1 / multi-tenant / 22ms / 180ms
          expect(combined).toMatch(/eu-central-1|us-west-1|multi-tenant|dedicated|22\s*ms|180\s*ms/i);
          // Zero Kyoto/Osaka leakage!
          expect(combined).not.toMatch(/Kyoto|Osaka|Ryokan|Nishiki/i);
        } finally {
          await customApp.close();
          customStore.close();
        }
      },
      45000,
    );
  });
});
