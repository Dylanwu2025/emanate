import dotenv from 'dotenv';

dotenv.config();

export type LLMProviderType = 'openai' | 'anthropic' | 'gemini' | 'openai-compatible' | 'mock';

export interface LLMConfig {
  provider: LLMProviderType;
  model: string;
  apiKey: string;
  baseUrl?: string;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  name?: string;
}

export interface StreamChatOptions {
  messages: ChatMessage[];
  maxTokens?: number;
  temperature?: number;
  signal?: AbortSignal;
}

export interface CompleteJSONOptions {
  messages: ChatMessage[];
  maxTokens?: number;
  temperature?: number;
  signal?: AbortSignal;
}

export interface LLMProvider {
  readonly config: LLMConfig;
  streamChat(options: StreamChatOptions): AsyncIterable<string>;
  completeJSON<T = unknown>(options: CompleteJSONOptions): Promise<T>;
}

export function resolveLLMConfigFromEnv(env: NodeJS.ProcessEnv = process.env): LLMConfig {
  const rawProvider = (env.LLM_PROVIDER || 'openai').trim().toLowerCase();
  const validProviders: LLMProviderType[] = [
    'openai',
    'anthropic',
    'gemini',
    'openai-compatible',
    'mock',
  ];
  if (!validProviders.includes(rawProvider as LLMProviderType)) {
    throw new Error(
      `Unsupported LLM_PROVIDER "${rawProvider}". Expected one of: ${validProviders.join(', ')}`,
    );
  }

  const provider = rawProvider as LLMProviderType;
  const defaultModels: Record<LLMProviderType, string> = {
    openai: 'gpt-4o-mini',
    anthropic: 'claude-3-5-haiku-latest',
    gemini: 'gemini-2.5-flash',
    'openai-compatible': 'gpt-4o-mini',
    mock: 'mock-model',
  };

  const model = (env.LLM_MODEL || defaultModels[provider]).trim();
  const apiKey = (env.LLM_API_KEY || '').trim();
  const rawBaseUrl = (env.LLM_BASE_URL || '').trim();

  return {
    provider,
    model,
    apiKey,
    baseUrl: rawBaseUrl ? rawBaseUrl.replace(/\/+$/, '') : undefined,
  };
}

export function parseJSONResponse<T>(raw: string): T {
  const trimmed = raw.trim();
  const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  const candidate = fencedMatch ? fencedMatch[1].trim() : trimmed;
  try {
    return JSON.parse(candidate) as T;
  } catch {
    const firstBrace = candidate.indexOf('{');
    const lastBrace = candidate.lastIndexOf('}');
    if (firstBrace >= 0 && lastBrace > firstBrace) {
      return JSON.parse(candidate.slice(firstBrace, lastBrace + 1)) as T;
    }
    throw new SyntaxError(`Failed to parse JSON response: ${raw}`);
  }
}

export function normalizeAlternatingMessages(
  messages: ChatMessage[],
  fallbackUserContent: string,
): Array<{ role: 'user' | 'assistant'; content: string }> {
  const coalesced: Array<{ role: 'user' | 'assistant'; content: string }> = [];

  for (const msg of messages) {
    if (msg.role === 'system') continue;
    const role: 'user' | 'assistant' = msg.role === 'assistant' ? 'assistant' : 'user';
    const prev = coalesced[coalesced.length - 1];
    if (prev && prev.role === role) {
      prev.content = `${prev.content}\n\n${msg.content}`;
    } else {
      coalesced.push({ role, content: msg.content });
    }
  }

  if (coalesced.length === 0) {
    return [{ role: 'user', content: fallbackUserContent }];
  }

  if (coalesced[0].role === 'assistant') {
    coalesced.unshift({ role: 'user', content: fallbackUserContent });
  }

  return coalesced;
}

function safeExtractDelta(
  extractDelta: (dataPayload: string) => string | null,
  payload: string,
): string | null {
  try {
    return extractDelta(payload);
  } catch {
    return null;
  }
}

async function* parseSSEStream(
  body: ReadableStream<Uint8Array>,
  extractDelta: (dataPayload: string) => string | null,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const onAbort = () => {
    reader.cancel().catch(() => {});
  };
  signal?.addEventListener('abort', onAbort, { once: true });

  try {
    while (true) {
      if (signal?.aborted) {
        throw signal.reason ?? new DOMException('Aborted', 'AbortError');
      }
      const { value, done } = await reader.read();
      if (done) break;
      if (signal?.aborted) {
        throw signal.reason ?? new DOMException('Aborted', 'AbortError');
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const payload = trimmed.slice(5).trim();
        if (!payload || payload === '[DONE]') continue;
        const delta = safeExtractDelta(extractDelta, payload);
        if (signal?.aborted) {
          throw signal.reason ?? new DOMException('Aborted', 'AbortError');
        }
        if (delta) {
          yield delta;
        }
      }
    }

    const tail = (buffer + decoder.decode()).trim();
    if (tail.startsWith('data:')) {
      const payload = tail.slice(5).trim();
      if (payload && payload !== '[DONE]') {
        const delta = safeExtractDelta(extractDelta, payload);
        if (delta) {
          yield delta;
        }
      }
    }
  } finally {
    signal?.removeEventListener('abort', onAbort);
    reader.releaseLock();
  }
}

export class HttpLLMProvider implements LLMProvider {
  public readonly config: LLMConfig;
  private readonly fetchImpl: typeof fetch;

  constructor(config: LLMConfig, fetchImpl: typeof fetch = globalThis.fetch) {
    this.config = config;
    this.fetchImpl = fetchImpl;
  }

  public getEndpointUrl(isStream: boolean): string {
    const { provider, model, baseUrl, apiKey } = this.config;
    if (provider === 'openai' || provider === 'openai-compatible') {
      const base = baseUrl || 'https://api.openai.com/v1';
      return `${base}/chat/completions`;
    }
    if (provider === 'anthropic') {
      const base = baseUrl || 'https://api.anthropic.com/v1';
      return `${base}/messages`;
    }
    if (provider === 'gemini') {
      const base = baseUrl || 'https://generativelanguage.googleapis.com/v1beta';
      const action = isStream ? 'streamGenerateContent?alt=sse&' : 'generateContent?';
      return `${base}/models/${encodeURIComponent(model)}:${action}key=${encodeURIComponent(apiKey)}`;
    }
    throw new Error(`HTTP endpoint not supported for provider: ${provider}`);
  }

  async *streamChat(options: StreamChatOptions): AsyncIterable<string> {
    const { provider, model, apiKey } = this.config;
    const { messages, maxTokens = 400, temperature = 0.4, signal } = options;

    if (signal?.aborted) {
      throw signal.reason ?? new DOMException('Aborted', 'AbortError');
    }

    const url = this.getEndpointUrl(true);

    if (provider === 'openai' || provider === 'openai-compatible') {
      const res = await this.fetchImpl(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
        },
        body: JSON.stringify({
          model,
          messages: messages.map((m) => ({ role: m.role, content: m.content })),
          max_tokens: maxTokens,
          temperature,
          stream: true,
        }),
        signal,
      });

      if (!res.ok || !res.body) {
        const errText = await res.text().catch(() => '');
        throw new Error(`OpenAI stream failed (${res.status}): ${errText}`);
      }

      yield* parseSSEStream(
        res.body,
        (payload) => {
          const parsed = JSON.parse(payload);
          return parsed.choices?.[0]?.delta?.content ?? null;
        },
        signal,
      );
      return;
    }

    if (provider === 'anthropic') {
      const systemText = messages
        .filter((m) => m.role === 'system')
        .map((m) => m.content)
        .join('\n\n');
      const chatMessages = normalizeAlternatingMessages(messages, 'Continue.');

      const res = await this.fetchImpl(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model,
          ...(systemText ? { system: systemText } : {}),
          messages: chatMessages,
          max_tokens: maxTokens,
          temperature,
          stream: true,
        }),
        signal,
      });

      if (!res.ok || !res.body) {
        const errText = await res.text().catch(() => '');
        throw new Error(`Anthropic stream failed (${res.status}): ${errText}`);
      }

      yield* parseSSEStream(
        res.body,
        (payload) => {
          const parsed = JSON.parse(payload);
          if (parsed.type === 'content_block_delta' && parsed.delta?.type === 'text_delta') {
            return parsed.delta.text ?? null;
          }
          return null;
        },
        signal,
      );
      return;
    }

    if (provider === 'gemini') {
      const systemText = messages
        .filter((m) => m.role === 'system')
        .map((m) => m.content)
        .join('\n\n');
      const contents = normalizeAlternatingMessages(messages, 'Continue.').map((m) => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      }));

      const res = await this.fetchImpl(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...(systemText ? { systemInstruction: { parts: [{ text: systemText }] } } : {}),
          contents,
          generationConfig: {
            maxOutputTokens: maxTokens,
            temperature,
          },
        }),
        signal,
      });

      if (!res.ok || !res.body) {
        const errText = await res.text().catch(() => '');
        throw new Error(`Gemini stream failed (${res.status}): ${errText}`);
      }

      yield* parseSSEStream(
        res.body,
        (payload) => {
          const parsed = JSON.parse(payload);
          return parsed.candidates?.[0]?.content?.parts?.[0]?.text ?? null;
        },
        signal,
      );
    }
  }

  async completeJSON<T = unknown>(options: CompleteJSONOptions): Promise<T> {
    const { provider, model, apiKey } = this.config;
    const { messages, maxTokens = 500, temperature = 0.1, signal } = options;

    if (signal?.aborted) {
      throw signal.reason ?? new DOMException('Aborted', 'AbortError');
    }

    const url = this.getEndpointUrl(false);

    if (provider === 'openai' || provider === 'openai-compatible') {
      const res = await this.fetchImpl(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
        },
        body: JSON.stringify({
          model,
          messages: messages.map((m) => ({ role: m.role, content: m.content })),
          max_tokens: maxTokens,
          temperature,
          response_format: { type: 'json_object' },
        }),
        signal,
      });

      if (!res.ok) {
        const errText = await res.text().catch(() => '');
        throw new Error(`OpenAI completeJSON failed (${res.status}): ${errText}`);
      }

      const data = (await res.json()) as {
        choices?: Array<{ message?: { content?: string } }>;
      };
      const raw = data.choices?.[0]?.message?.content ?? '{}';
      return parseJSONResponse<T>(raw);
    }

    if (provider === 'anthropic') {
      const systemText = messages
        .filter((m) => m.role === 'system')
        .map((m) => m.content)
        .join('\n\n');
      const chatMessages = normalizeAlternatingMessages(messages, '{}');

      const res = await this.fetchImpl(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model,
          ...(systemText
            ? { system: `${systemText}\n\nRespond ONLY with a valid JSON object.` }
            : { system: 'Respond ONLY with a valid JSON object.' }),
          messages: chatMessages,
          max_tokens: maxTokens,
          temperature,
        }),
        signal,
      });

      if (!res.ok) {
        const errText = await res.text().catch(() => '');
        throw new Error(`Anthropic completeJSON failed (${res.status}): ${errText}`);
      }

      const data = (await res.json()) as {
        content?: Array<{ type: string; text?: string }>;
      };
      const raw = data.content?.find((c) => c.type === 'text')?.text ?? '{}';
      return parseJSONResponse<T>(raw);
    }

    if (provider === 'gemini') {
      const systemText = messages
        .filter((m) => m.role === 'system')
        .map((m) => m.content)
        .join('\n\n');
      const contents = normalizeAlternatingMessages(messages, '{}').map((m) => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      }));

      const res = await this.fetchImpl(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...(systemText ? { systemInstruction: { parts: [{ text: systemText }] } } : {}),
          contents,
          generationConfig: {
            maxOutputTokens: maxTokens,
            temperature,
            responseMimeType: 'application/json',
          },
        }),
        signal,
      });

      if (!res.ok) {
        const errText = await res.text().catch(() => '');
        throw new Error(`Gemini completeJSON failed (${res.status}): ${errText}`);
      }

      const data = (await res.json()) as {
        candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
      };
      const raw = data.candidates?.[0]?.content?.parts?.[0]?.text ?? '{}';
      return parseJSONResponse<T>(raw);
    }

    throw new Error(`Unsupported provider for completeJSON: ${provider}`);
  }
}

export interface ScriptedStreamStep {
  chunks: string[];
  delayMs?: number;
  match?: RegExp | ((options: StreamChatOptions) => boolean);
}

export interface ScriptedJSONStep<T = unknown> {
  response: T | ((options: CompleteJSONOptions) => T | Promise<T>);
  delayMs?: number;
  match?: RegExp | ((options: CompleteJSONOptions) => boolean);
}

export function sleepAbortable(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) {
    if (signal?.aborted) {
      return Promise.reject(signal.reason ?? new DOMException('Aborted', 'AbortError'));
    }
    return Promise.resolve();
  }

  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason ?? new DOMException('Aborted', 'AbortError'));
      return;
    }

    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);

    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason ?? new DOMException('Aborted', 'AbortError'));
    };

    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

export class ScriptableMockLLMProvider implements LLMProvider {
  public readonly config: LLMConfig;
  public readonly streamCalls: StreamChatOptions[] = [];
  public readonly jsonCalls: CompleteJSONOptions[] = [];

  private streamQueue: ScriptedStreamStep[] = [];
  private jsonQueue: ScriptedJSONStep[] = [];
  private defaultStreamChunks: string[] = ['Mock ', 'stream ', 'response.'];
  private defaultChunkDelayMs = 5;

  constructor(config?: Partial<LLMConfig>) {
    this.config = {
      provider: config?.provider ?? 'mock',
      model: config?.model ?? 'mock-model',
      apiKey: config?.apiKey ?? '',
      baseUrl: config?.baseUrl,
    };
  }

  enqueueStream(step: ScriptedStreamStep): this {
    this.streamQueue.push(step);
    return this;
  }

  enqueueJSON<T>(step: ScriptedJSONStep<T>): this {
    this.jsonQueue.push(step as ScriptedJSONStep);
    return this;
  }

  setDefaultStream(chunks: string[], delayMs = 5): this {
    this.defaultStreamChunks = chunks;
    this.defaultChunkDelayMs = delayMs;
    return this;
  }

  private pickStreamStep(options: StreamChatOptions): ScriptedStreamStep {
    const combinedText = options.messages.map((m) => m.content).join('\n');
    const matchIdx = this.streamQueue.findIndex((step) => {
      if (!step.match) return true;
      if (step.match instanceof RegExp) return step.match.test(combinedText);
      return step.match(options);
    });

    if (matchIdx >= 0) {
      const [picked] = this.streamQueue.splice(matchIdx, 1);
      return picked;
    }

    return {
      chunks: this.defaultStreamChunks,
      delayMs: this.defaultChunkDelayMs,
    };
  }

  private pickJSONStep(options: CompleteJSONOptions): ScriptedJSONStep | undefined {
    const combinedText = options.messages.map((m) => m.content).join('\n');
    const matchIdx = this.jsonQueue.findIndex((step) => {
      if (!step.match) return true;
      if (step.match instanceof RegExp) return step.match.test(combinedText);
      return step.match(options);
    });

    if (matchIdx >= 0) {
      const [picked] = this.jsonQueue.splice(matchIdx, 1);
      return picked;
    }

    return undefined;
  }

  hasMatchingSteeringJSONStep(options: CompleteJSONOptions): boolean {
    const combinedText = options.messages.map((m) => m.content).join('\n');
    return this.jsonQueue.some((step) => {
      if (step.match) {
        if (step.match instanceof RegExp) return step.match.test(combinedText);
        return step.match(options);
      }
      if (typeof step.response === 'object' && step.response !== null) {
        return 'action' in step.response;
      }
      return false;
    });
  }

  async *streamChat(options: StreamChatOptions): AsyncIterable<string> {
    this.streamCalls.push(options);
    const { signal } = options;

    if (signal?.aborted) {
      throw signal.reason ?? new DOMException('Aborted', 'AbortError');
    }

    const step = this.pickStreamStep(options);
    const delayMs = step.delayMs ?? this.defaultChunkDelayMs;

    for (const chunk of step.chunks) {
      await sleepAbortable(delayMs, signal);
      if (signal?.aborted) {
        throw signal.reason ?? new DOMException('Aborted', 'AbortError');
      }
      yield chunk;
    }
  }

  async completeJSON<T = unknown>(options: CompleteJSONOptions): Promise<T> {
    this.jsonCalls.push(options);
    const { signal } = options;

    if (signal?.aborted) {
      throw signal.reason ?? new DOMException('Aborted', 'AbortError');
    }

    const step = this.pickJSONStep(options);
    if (!step) {
      return {} as T;
    }

    if (step.delayMs && step.delayMs > 0) {
      await sleepAbortable(step.delayMs, signal);
    }

    if (typeof step.response === 'function') {
      return (await (step.response as (opts: CompleteJSONOptions) => T | Promise<T>)(options)) as T;
    }

    return step.response as T;
  }
}

export function createLLMProvider(
  env: NodeJS.ProcessEnv = process.env,
  fetchImpl?: typeof fetch,
): LLMProvider {
  const config = resolveLLMConfigFromEnv(env);
  if (config.provider === 'mock') {
    return new ScriptableMockLLMProvider(config);
  }
  return new HttpLLMProvider(config, fetchImpl);
}
