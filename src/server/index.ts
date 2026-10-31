import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { CustomAgentRuntimeConfig } from './agents.js';
import { EmanateStore } from './db.js';
import { createLLMProvider, type LLMProvider } from './llm.js';
import { GroupChatOrchestrator, type OrchestratorSSEEvent } from './orchestrator.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CLIENT_DIR = join(__dirname, '..', 'client');

const STATIC_ASSET_CACHE = new Map<string, string>([
  ['index.html', readFileSync(join(CLIENT_DIR, 'index.html'), 'utf-8')],
  ['styles.css', readFileSync(join(CLIENT_DIR, 'styles.css'), 'utf-8')],
  ['app.js', readFileSync(join(CLIENT_DIR, 'app.js'), 'utf-8')],
]);

export interface AppServerOptions {
  store?: EmanateStore;
  llm?: LLMProvider;
  orchestrator?: GroupChatOrchestrator;
  dbPath?: string;
  streamPaceMs?: number;
  runtimeConfig?: CustomAgentRuntimeConfig;
}

export interface AppServerInstance {
  server: Server;
  store: EmanateStore;
  llm: LLMProvider;
  orchestrator: GroupChatOrchestrator;
  listen: (port?: number, host?: string) => Promise<{ port: number; url: string }>;
  close: () => Promise<void>;
}

function setCorsHeaders(res: ServerResponse): void {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Accept');
}

function sendJSON(res: ServerResponse, statusCode: number, payload: unknown): void {
  setCorsHeaders(res);
  res.writeHead(statusCode, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(payload));
}

function sendStaticFile(
  res: ServerResponse,
  fileName: string,
  contentType: string,
): void {
  setCorsHeaders(res);
  const content =
    STATIC_ASSET_CACHE.get(fileName) ?? readFileSync(join(CLIENT_DIR, fileName), 'utf-8');
  res.writeHead(200, {
    'Content-Type': contentType,
    'Cache-Control': 'no-cache',
  });
  res.end(content);
}

const MAX_BODY_BYTES = 1024 * 1024;

async function readRequestBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  let totalBytes = 0;
  for await (const chunk of req) {
    const buf = typeof chunk === 'string' ? Buffer.from(chunk) : chunk;
    totalBytes += buf.length;
    if (totalBytes > MAX_BODY_BYTES) {
      throw new Error('Request body too large.');
    }
    chunks.push(buf);
  }
  return Buffer.concat(chunks).toString('utf-8');
}

export function createAppServer(options: AppServerOptions = {}): AppServerInstance {
  const store = options.store ?? new EmanateStore(options.dbPath ?? ':memory:');
  const llm = options.llm ?? createLLMProvider();
  const orchestrator =
    options.orchestrator ??
    new GroupChatOrchestrator({
      store,
      llm,
      streamPaceMs: options.streamPaceMs,
      runtimeConfig: options.runtimeConfig,
    });

  const sseClients = new Set<ServerResponse>();

  const broadcastSSE = (event: OrchestratorSSEEvent) => {
    const serialized = `data: ${JSON.stringify(event)}\n\n`;
    for (const client of sseClients) {
      if (!client.writableEnded) {
        client.write(serialized);
      }
    }
  };

  const unsubscribeOrchestrator = orchestrator.onEvent(broadcastSSE);

  const server = createServer(async (req, res) => {
    setCorsHeaders(res);

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
    const pathname = url.pathname;

    try {
      if (req.method === 'GET' && (pathname === '/' || pathname === '/index.html')) {
        sendStaticFile(res, 'index.html', 'text/html; charset=utf-8');
        return;
      }

      if (req.method === 'GET' && pathname === '/styles.css') {
        sendStaticFile(res, 'styles.css', 'text/css; charset=utf-8');
        return;
      }

      if (req.method === 'GET' && pathname === '/app.js') {
        sendStaticFile(res, 'app.js', 'application/javascript; charset=utf-8');
        return;
      }

      if (req.method === 'GET' && (pathname === '/api/state' || pathname === '/api/messages')) {
        sendJSON(res, 200, {
          messages: orchestrator.getPublicMessages(),
        });
        return;
      }

      if (req.method === 'GET' && pathname === '/api/stream') {
        res.writeHead(200, {
          'Content-Type': 'text/event-stream; charset=utf-8',
          'Cache-Control': 'no-cache, no-transform',
          Connection: 'keep-alive',
          'X-Accel-Buffering': 'no',
        });
        res.flushHeaders?.();
        res.write(': connected\n\n');

        sseClients.add(res);
        req.on('close', () => {
          sseClients.delete(res);
        });
        return;
      }

      if (req.method === 'POST' && pathname === '/api/messages') {
        const rawBody = await readRequestBody(req);
        let parsed: { content?: string; message?: string } = {};
        try {
          parsed = rawBody ? JSON.parse(rawBody) : {};
        } catch {
          sendJSON(res, 400, { error: 'Invalid JSON request body.' });
          return;
        }

        const content = (parsed.content ?? parsed.message ?? '').trim();
        if (!content) {
          sendJSON(res, 400, { error: 'Message content cannot be empty.' });
          return;
        }

        const turnResult = await orchestrator.handleUserMessage(content);
        sendJSON(res, 200, {
          ok: true,
          userMessage: turnResult.userMessage,
          messages: orchestrator.getPublicMessages(),
        });
        return;
      }

      if (req.method === 'POST' && pathname === '/api/stop') {
        const messages = await orchestrator.stopAll();
        sendJSON(res, 200, {
          ok: true,
          messages,
        });
        return;
      }

      if (req.method === 'POST' && pathname === '/api/reset') {
        await orchestrator.reset();
        sendJSON(res, 200, {
          ok: true,
          messages: [],
        });
        return;
      }

      sendJSON(res, 404, { error: `Not found: ${req.method} ${pathname}` });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Internal server error';
      if (!res.headersSent) {
        sendJSON(res, 500, { error: message });
      }
    }
  });

  const listen = (port = 0, host = '127.0.0.1'): Promise<{ port: number; url: string }> => {
    return new Promise((resolve, reject) => {
      server.once('error', reject);
      server.listen(port, host, () => {
        server.off('error', reject);
        const addr = server.address();
        const boundPort = typeof addr === 'object' && addr ? addr.port : port;
        resolve({
          port: boundPort,
          url: `http://${host}:${boundPort}`,
        });
      });
    });
  };

  const close = (): Promise<void> => {
    unsubscribeOrchestrator();
    for (const client of sseClients) {
      client.end();
    }
    sseClients.clear();
    return new Promise((resolve) => {
      server.close(() => resolve());
    });
  };

  return {
    server,
    store,
    llm,
    orchestrator,
    listen,
    close,
  };
}

const isDirectEntry =
  process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];

if (isDirectEntry) {
  const port = Number(process.env.PORT ?? 3000);
  const dbPath = process.env.DB_PATH ?? 'emanate.db';
  const defaultPace = process.env.STREAM_PACE_MS ? Number(process.env.STREAM_PACE_MS) : 25;
  const app = createAppServer({ dbPath, streamPaceMs: defaultPace });
  app.listen(port, '0.0.0.0').then(({ port: boundPort }) => {
    console.log(`Emanate Group Chat running at http://localhost:${boundPort}`);
  });
}
