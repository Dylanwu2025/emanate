# Emanate — Multi-Turn Agent Steering Group Chat

A real-time, multi-agent group chat web application where three distinct agents (**Maya**, **Liam**, and **Zoe**) with strictly isolated private knowledge vaults collaborate concurrently in a shared conversation. Users can steer any or all agents **in flight while they are actively streaming**—pivoting the trajectory of an ongoing response within the exact same message bubble, establishing persistent standing rules across page reloads, or stopping generation immediately—without exposing any internal orchestration mechanics in the UI.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the complete system architecture and sequence diagrams.

---

## 1. Quickstart

### Prerequisites
- **Node.js** `v20+` and **npm**

### Install & Run
```bash
# 1. Install dependencies
npm install

# 2. Configure your LLM API key in .env
cp .env.example .env
# Edit .env and set LLM_API_KEY="your-api-key"

# 3. Start the web app & API server (defaults to http://localhost:3000)
npm start
```

Open **[http://localhost:3000](http://localhost:3000)** in your browser.

### Run Verification & Test Suite
```bash
# TypeScript strict typecheck
npm run typecheck

# Full automated test suite (Part 1–4 unit, integration & E2E tests)
npm test
```

---

## 2. Swapping LLM Providers & Models (Zero Code Changes)

All LLM calls (`streamChat` and `completeJSON`) route through a unified provider-agnostic gateway in `src/server/llm.ts` controlled purely via environment variables in `.env`:

```bash
LLM_PROVIDER="openai"          # "openai" | "anthropic" | "gemini" | "openai-compatible"
LLM_MODEL="gpt-4o-mini"        # Model slug for the chosen provider
LLM_API_KEY="sk-..."           # Your API key
LLM_BASE_URL=""                # Optional custom base URL (e.g. OpenRouter, local vLLM, Ollama)
STREAM_PACE_MS="25"            # Optional token pacing (ms) for smooth visual mid-stream steering
```

- **Tested Live Configuration**: Built and verified end-to-end against **OpenAI `gpt-4o-mini`** (`LLM_PROVIDER="openai"`, `LLM_MODEL="gpt-4o-mini"`).
- **Switching to Anthropic**: Set `LLM_PROVIDER="anthropic"`, `LLM_MODEL="claude-3-5-haiku-latest"`, and `LLM_API_KEY="sk-ant-..."`.
- **Switching to Google Gemini**: Set `LLM_PROVIDER="gemini"`, `LLM_MODEL="gemini-2.5-flash"`, and `LLM_API_KEY="AIza..."`.
- **Switching to OpenRouter / Local Models**: Set `LLM_PROVIDER="openai-compatible"`, `LLM_BASE_URL="https://openrouter.ai/api/v1"`, `LLM_MODEL="..."`, and `LLM_API_KEY="..."`.

---

## 3. The Group: 3 Agents with Isolated Private Contexts

The group is planning a 5-day trip to **Kyoto & Osaka** for four friends (Sam, Priya, Alex, and Jordan). Each agent only has access to its own private markdown file under `src/server/contexts/<agent>.md` plus the shared chat transcript and active standing rules:

| Agent | Role & Personality | Exclusive Private Vault (`src/server/contexts/*.md`) |
| :--- | :--- | :--- |
| **Maya** | **Budget, Flights & Lodging** — Direct, numbers-driven, cost-conscious. | $4,800 total group budget ($1,200/person); exchange rate `1 USD = 150 JPY`; **Flight A** (ANA 6:45 AM, $650, lands 11:00 AM) vs. **Flight B** (JAL 11:30 AM, $820, lands 3:45 PM); **Ryokan Gion** ($380/night, strict **6:00 PM check-in cutoff**) vs. **Hotel Granvia** ($290/night, 24h check-in at Kyoto Station); **Cross Hotel Osaka** ($260/night). |
| **Liam** | **Local Guide (Food, Sights & Transit)** — Warm, food-obsessed, timing-focused. | **KIX to Gion** takes **2 hours 15 minutes** door-to-door after customs; **KIX to Kyoto Station** takes **1 hour 35 minutes**; **Nishiki Market Street Food Tour** (11:00 AM–2:00 PM, heavy shared-grill shellfish/seafood); **Arashiyama Tofu Kaiseki** (100% vegetarian & shellfish-free, last seating 7:30 PM); Fushimi Inari & Kiyomizu-dera crowd windows. |
| **Zoe** | **Group Dynamics, Dietary & Pacing** — Empathetic, protective of group health & sleep. | **Sam** has a **severe anaphylactic shellfish allergy** (cannot be around shared seafood grills) and boat/bus motion sickness; **Priya** is **strictly vegetarian**; unanimous group rule against **waking up before 7:30 AM** (ruling out 6:45 AM flights). |

---

## 4. Sample Prompts to Test Every Capability

1. **Cross-Agent Private Vault Synthesis + Bounded A2A Follow-Up Waves**:
   > *"Should we take Flight B and stay at Ryokan Gion on Day 1?"*
   - **What happens**: Maya shares Flight B's price ($820, lands 3:45 PM) and Ryokan Gion's strict 6:00 PM cutoff; Liam reveals KIX-to-Gion transit takes 2h 15m (arriving at 6:00 PM sharp with zero buffer); in the automatic A2A follow-up wave, Maya recommends Hotel Granvia ($290/night, 24h check-in) and Zoe vetoes the 6:45 AM Flight A due to the group's 7:30 AM sleep rule. The exchange resolves and ends on its own.

2. **Selective Per-Agent Mid-Stream Steering (Same-Bubble Continuation)**:
   > First send: *"@Maya and @Liam give us a full breakdown of flight and hotel prices in USD plus airport transit times."*
   > While both are actively streaming, immediately send: *"Maya, convert all prices to Japanese Yen instead"*
   - **What happens**: Liam's stream continues uninterrupted. Maya's stream pauses at the exact character already flushed to the screen and seamlessly pivots into Japanese Yen (`¥97,500`, `¥123,000`, `¥57,000`) **inside the exact same message bubble** (`id` and `seq` unchanged).

3. **Global Mid-Stream Steering**:
   > While multiple agents are streaming, send: *"Actually, wrap up in one short uppercase sentence"*
   - **What happens**: All currently active streams pivot in-place within their existing bubbles to finish in uppercase.

4. **Persistent Standing Instructions Across Reloads**:
   > Send: *"Always answer in concise bullet points from now on"* (or *"Maya, always quote prices in JPY"*)
   - **What happens**: Saved in SQLite `standing_rules` (automatically superseding older conflicting rules in the same scope). Refresh the browser (`Cmd+R`) and ask a new question—both the exact message ordering (`seq ASC`) and the standing rule persist.

5. **Pre-Response Effort Calibration & Silence Gate (Stretch)**:
   > Send: *"@Maya what is the total group budget?"*
   - **What happens**: Liam and Zoe stay completely silent (zero bubbles created), and Maya responds with a calibrated `light` 1-sentence answer.

6. **Instant Stop-All**:
   > Click the **Stop** button or type `stop` while agents are streaming.
   - **What happens**: All active streams and any pending A2A follow-up waves halt immediately (`<5ms`), freezing each message cleanly at its flushed text.

---

## 5. Deliberate Design Decisions & Scope Cuts

- **Single Integer `seq` Defended Ordering**: Every message reserves an auto-incrementing integer `seq` in SQLite **before** its LLM stream starts. Both live SSE reconciliation and full page reloads (`GET /api/state`) sort strictly by `seq ASC`, guaranteeing identical message ordering across reloads regardless of which concurrent stream finishes first.
- **Zero Exposed Mechanism in the UI**: The DOM and wire payloads (`PublicWireMessage`) contain strictly standard chat fields (`seq`, `id`, `role`, `agentId`, `agentName`, `content`, `status`, `createdAt`). Internal orchestration concepts (`effort`, `thesis`, `peerTheses`, `epoch`, `standingRules`, steering actions) are never sent to or rendered in the browser.
- **Focused Scope**: Built with a zero-bloat 2-table SQLite schema (`messages`, `standing_rules`) and a single-port Node HTTP + SSE server serving the group chat web UI directly, avoiding unnecessary external queues, multi-container infrastructure, or exposed debug dashboards.
