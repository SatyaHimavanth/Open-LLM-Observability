# Open-LLM-Observability — Low-Level Architecture

## Package Structure

```
universal_agent_obs/
├── __init__.py              # Entry: auto-installs all interceptors
├── __main__.py              # Delegates to server.__main__
├── core.py                  # Span model, context, emit queue, worker
├── dspy.py                  # DSPy TraceContextCallbackHandler
├── crewai.py                # CrewAI TraceContextCallbackHandler
├── google.py                # Google ADK/GenAI TraceContextCallbackHandler
├── langchain.py             # LangChain TraceContextCallbackHandler
├── openai.py                # OpenAI TraceContextCallbackHandler
├── interceptors/
│   ├── agents.py            # AutoGen + OpenAI Agents SDK patching
│   ├── crewai.py            # CrewAI Crew class patching
│   ├── dspy.py              # DSPy LM.__call__ patching
│   ├── google_adk.py        # Google ADK Runner patching
│   ├── google_genai.py      # Google GenAI Models patching
│   ├── http.py              # httpx/requests fallback patching
│   ├── langchain.py         # LangChain BaseCallbackHandler + LangGraph
│   └── openai_sdk.py        # OpenAI Completions.create patching
└── server/
    ├── __main__.py           # Uvicorn startup with Windows compatibility
    ├── app.py                # FastAPI application (REST + WebSocket + SPA)
    ├── storage.py            # SQLAlchemy models and CRUD
    └── static/
        ├── index.html        # Single-page dashboard (dark theme)
        └── model_prices_and_context_window.json  # Cost lookup table
```

---

## Module-by-Module Breakdown

### `core.py` — The Engine

**Responsibilities:**
- Defines the `Span` dataclass (the universal event unit)
- Manages trace context via Python `contextvars` (trace_id, span_id, project, user, tags, metadata)
- Provides `emit()` for non-blocking span submission to a bounded queue (10k max)
- Runs a background daemon thread (`_worker`) that POSTs spans to the collector
- Exposes `flush()` for graceful shutdown (registered via `atexit`)
- Utility functions: `detect_provider()`, `detect_framework_from_stack()`

**Key Flow:**
```python
emit(span) → _emit_queue.put_nowait(span.to_dict())
                          ↓ (background thread)
              _worker() → _post_span(span_dict)
                          ↓
              urllib.request POST → {SERVER_URL}/ingest
```

**Context Propagation:**
```python
# Set context for current thread/async task
set_context(project_name="my-project", user={...}, tags=[...])

# Or scoped context manager
with trace_context(project_name="my-project"):
    agent.run(...)
```

Context values are resolved at `span.to_dict()` time, allowing interceptors to set context before emitting and restore it after.

---

### `__init__.py` — Auto-Installer

On `import universal_agent_obs`:
1. Checks `AGENT_OBS` env var (enabled by default)
2. Iterates over all interceptors in a fixed order:
   - langchain → crewai → autogen → openai-agents → google-adk → google-genai → openai-sdk → dspy → http
3. Each interceptor's `install()` tries to import its target framework
4. If the framework is present, patches are applied silently
5. If not, returns `False` and moves on
6. Prints a summary of active interceptors

---

### Interceptors (Monkey-Patching Layer)

All interceptors follow the same pattern:
1. `install()` → tries to import the target library
2. `_patch_*()` → replaces class methods with wrapped versions
3. Wrapped methods: emit `*_start` span → call original → emit `*_end` span
4. Context (project, user, tags) is extracted from callback objects or contextvars

#### `interceptors/dspy.py`
- **Target:** `dspy.clients.lm.LM`
- **Patched methods:** `__call__` (+ async variant)
- **How:** Wraps the `with_callbacks`-decorated `__call__`. Emits `llm_start` with messages, then `llm_end` with response content + token usage extracted from `self.history[-1]['usage']`.
- **Context:** Reads `_obs_callbacks` attribute on the LM instance, or global `dspy.settings.callbacks`.

#### `interceptors/langchain.py`
- **Target:** LangChain's `BaseCallbackHandler` + LangGraph's `Pregel`
- **Patched methods:** All callback hooks (`on_llm_start`, `on_tool_start`, etc.) + `Pregel.invoke`/`ainvoke`/`stream`/`astream`
- **How:** Registers a custom callback handler that emits spans for every LLM, tool, chain, and agent event.
- **Context:** Extracted from `RunnableConfig.metadata` and `RunnableConfig.tags`, or from `TraceContextCallbackHandler` in the callbacks list.

#### `interceptors/crewai.py`
- **Target:** `crewai.Crew`
- **Patched methods:** `__init__`, `kickoff`
- **How:** Injects LiteLLM success/failure callbacks. On kickoff, emits agent-level spans.
- **Context:** From `TraceContextCallbackHandler` passed as `callbacks` kwarg.

#### `interceptors/google_adk.py`
- **Target:** `google.adk.runners.Runner`
- **Patched methods:** `run_async`
- **How:** Wraps the async generator to emit spans for each event (agent/tool/LLM).
- **Context:** From `callbacks` kwarg on `run_async`.

#### `interceptors/google_genai.py`
- **Target:** `google.genai.models.Models`
- **Patched methods:** `generate_content` (sync + async + streaming variants)
- **How:** Wraps content generation to emit `llm_start`/`llm_end` with messages, model, tokens.
- **Context:** From `callbacks` kwarg on `generate_content`.

#### `interceptors/openai_sdk.py`
- **Target:** `openai.resources.chat.completions.Completions`
- **Patched methods:** `create`
- **How:** Accepts a `callbacks` kwarg, emits `llm_start`/`llm_end` with messages, model, token usage.
- **Context:** From `TraceContextCallbackHandler` in `callbacks`.

#### `interceptors/agents.py`
- **Target:** AutoGen `ConversableAgent` + OpenAI Agents SDK `Runner`
- **Patched methods:** Various agent entry points
- **How:** Wraps agent execution to emit agent/LLM/tool spans.
- **Context:** From `callbacks` on the agent instance.

#### `interceptors/http.py`
- **Target:** `httpx.Client.send` + `requests.Session.request`
- **Patched methods:** HTTP send/request
- **How:** Detects LLM API URLs (OpenAI, Anthropic, etc.), parses request/response bodies, emits spans. Acts as a catch-all fallback.
- **Context:** From thread-local contextvars.

---

### Context Callback Helpers

Each framework module (`dspy.py`, `langchain.py`, `crewai.py`, `google.py`, `openai.py`) exports a `TraceContextCallbackHandler` class:

```python
class TraceContextCallbackHandler:
    def __init__(self, *, user=None, project_name=None, tags=None, metadata=None): ...
    def obs_context(self) -> dict: ...  # Returns {project_name, user, tags, metadata}
```

These are lightweight carriers that interceptors look for via:
1. `kwargs.get("callbacks")` in function calls
2. `getattr(instance, "_obs_callbacks")` on LM/Agent objects
3. `getattr(instance, "callbacks")` on framework objects

The `context_from_callbacks()` helper iterates callbacks, calls `.obs_context()`, and merges results into a dict that gets passed to `_set_context()`.

---

### `server/app.py` — Collector + Dashboard

**FastAPI application** with these responsibilities:

1. **Ingest endpoint** (`POST /ingest`): Receives span dicts, validates auth, stores in DB, broadcasts via WebSocket.
2. **Trace API** (`GET /traces`, `GET /traces/{id}`): Returns aggregated trace summaries with span trees, token rollups, cost estimation.
3. **Project API** (`GET /projects`, `POST /projects`): Project-level aggregations.
4. **Auth** (`POST /register`, `POST /login`): User management with client ID/secret for agent auth.
5. **WebSocket** (`WS /ws`): Live span streaming to the dashboard.
6. **Cost estimation**: Uses `model_prices_and_context_window.json` to estimate cost from token counts.
7. **Static SPA** (`GET /`): Serves `index.html` — a dark-themed dashboard with trace list, detail view, settings.

**Authentication:**
- Agents authenticate via `x-client-id` + `x-client-secret` headers on `/ingest`
- Dashboard users authenticate via `/login` endpoint
- Default admin credentials: `admin` / `password`

---

### `server/storage.py` — Persistence

**SQLAlchemy models:**
- `SpanRecord`: Stores all span fields as JSON columns. Indexed by `trace_id`, `project_name`.
- `UserRecord`: Dashboard user accounts with hashed passwords, client_id/secret.
- `ProjectRecord`: Project metadata.

**Database:** SQLite by default (`agent_obs.sqlite3`), Postgres-ready via `AGENT_OBS_DB_URL`.

---

### `server/__main__.py` — Server Entry Point

- Sets Windows event loop policy (`WindowsSelectorEventLoopPolicy`)
- Configures Uvicorn with smooth signal handling
- Watches stdin for EOF to support graceful shutdown in subprocesses
- Accepts `--host`, `--port`, `--reload` CLI args

---

## Data Flow (End-to-End)

```
1. Agent code runs (e.g., dspy.Predict(BasicQA)(question="..."))
2. Interceptor wrapper fires:
   a. Extracts messages, model, callbacks
   b. Resolves trace context (trace_id, parent_span, project, user, tags)
   c. Emits llm_start span → queue
   d. Calls original method
   e. Extracts response + token usage
   f. Emits llm_end span → queue
3. Background worker thread:
   a. Reads span from queue
   b. Serializes to JSON
   c. POSTs to {AGENT_OBS_URL}/ingest with auth headers
4. Server /ingest handler:
   a. Validates client credentials
   b. Stores SpanRecord in DB
   c. Broadcasts to WebSocket subscribers
5. Dashboard:
   a. Fetches /traces → shows trace list
   b. Fetches /traces/{id} → shows span tree
   c. WebSocket → live updates
```

---

## Trace Context Hierarchy

```
trace_id (UUID) — groups all spans from one logical operation
├── span_id (UUID) — unique per span
├── parent_span (UUID | null) — forms parent-child tree
├── event — lifecycle event (llm_start, llm_end, tool_start, agent_start, etc.)
└── resource — category (llm, tool, agent, chain)
```

The server reconstructs the tree from `parent_span` references and computes:
- Total latency (first span start → last span end)
- Token rollups (sum of all llm_end token counts)
- Cost estimation (tokens × model price from lookup table)
- Agent list (unique agent_name values)
- Model list (unique model values)

---

## Error Handling Philosophy

- **Never crash the agent**: All interceptor code is wrapped in try/except
- **Silent degradation**: If queue is full, spans are dropped. If server is unreachable, spans are lost.
- **No retries**: The worker thread moves on after a failed POST (observability is best-effort)
- **Graceful flush**: `atexit.register(flush)` attempts to drain the queue before process exit