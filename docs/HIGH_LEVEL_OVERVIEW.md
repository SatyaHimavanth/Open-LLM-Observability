# Open-LLM-Observability — High-Level Overview

## What It Is

Open-LLM-Observability (`universal_agent_obs`) is a **zero-code, framework-agnostic observability platform** for Python LLM applications. It captures traces from LLM calls, agent runs, tool invocations, and multi-agent workflows with a single import — no SDK wrapping or decorator changes required.

## How It Works (30-Second Summary)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Your Application                             │
│                                                                      │
│   import universal_agent_obs   ← one-line integration                │
│   # ... your LangChain / DSPy / CrewAI / OpenAI / Google code ...    │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ auto-patches at import time
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Interceptor Layer                                │
│  Monkey-patches framework classes to emit Span events:               │
│  • LangChain (BaseCallbackHandler + LangGraph Pregel)                │
│  • DSPy (LM.__call__)                                                │
│  • CrewAI (Crew.kickoff + LiteLLM)                                   │
│  • Google ADK (Runner.run_async) + GenAI (Models.generate_content)   │
│  • OpenAI SDK (Completions.create) + OpenAI Agents (Runner.run)      │
│  • HTTP fallback (httpx.Client.send / requests.Session.request)      │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ non-blocking queue
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Core (emit queue + worker)                      │
│  • Span dataclass with trace_id / parent_span tree structure         │
│  • contextvar-based trace propagation (project, user, tags)          │
│  • Background thread POSTs spans to collector via HTTP               │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ POST /ingest
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   Collector Server (FastAPI)                         │
│  • Receives and stores spans in SQLite (or Postgres)                 │
│  • Aggregates spans into traces with cost/token rollups              │
│  • Serves a browser dashboard (SPA) for viewing traces               │
│  • WebSocket for live trace streaming                                │
│  • REST API for querying traces, spans, projects, stats              │
└──────────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

1. **Zero-code instrumentation** — `import universal_agent_obs` is all you need. The package auto-detects installed frameworks and patches them at import time.

2. **Non-intrusive** — All patches are wrapped in try/except. If a framework isn't installed or patching fails, the agent runs normally. Observability never crashes the application.

3. **Lightweight core** — Only `fastapi`, `httpx`, `uvicorn`, `sqlalchemy` are required. Framework SDKs are optional extras.

4. **Context propagation** — Trace context (project, user, tags, metadata) flows through Python's `contextvars` and optional `TraceContextCallbackHandler` objects.

5. **Asynchronous emission** — Spans are put into a bounded queue and shipped by a daemon thread. Agent latency is unaffected.

## Supported Frameworks

| Framework | Interceptor | Context Helper |
|-----------|-------------|----------------|
| LangChain / LangGraph | `interceptors/langchain.py` | `universal_agent_obs.langchain.TraceContextCallbackHandler` |
| DSPy | `interceptors/dspy.py` | `universal_agent_obs.dspy.TraceContextCallbackHandler` |
| CrewAI | `interceptors/crewai.py` | `universal_agent_obs.crewai.TraceContextCallbackHandler` |
| Google ADK | `interceptors/google_adk.py` | `universal_agent_obs.google.TraceContextCallbackHandler` |
| Google GenAI | `interceptors/google_genai.py` | (same as above) |
| OpenAI SDK | `interceptors/openai_sdk.py` | `universal_agent_obs.openai.TraceContextCallbackHandler` |
| OpenAI Agents | `interceptors/agents.py` | (same as above) |
| Any HTTP-based LLM | `interceptors/http.py` | `universal_agent_obs.set_context(...)` |

## Quick Start

```bash
# Install
uv sync

# Start collector + dashboard
uv run agent-obs

# In your agent code
import universal_agent_obs
# ... your agent runs normally, traces appear at http://localhost:4317
```

## Configuration

All config is via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `AGENT_OBS` | `1` | Enable/disable tracing |
| `AGENT_OBS_URL` | `http://localhost:4317` | Collector endpoint |
| `AGENT_OBS_CLIENT_ID` | (empty) | Auth client ID |
| `AGENT_OBS_CLIENT_SECRET` | (empty) | Auth client secret |
| `AGENT_OBS_PROJECT` | `default` | Project name for spans |
| `AGENT_OBS_DB_URL` | `sqlite:///./agent_obs.sqlite3` | Database URL |

## Data Model

```
Trace (trace_id)
├── Span: llm_start (messages, model, framework)
│   └── Span: llm_end (response, tokens, latency_ms, cost)
├── Span: tool_start (tool_name, tool_input)
│   └── Span: tool_end (tool_output)
├── Span: agent_start (agent_name)
│   ├── Span: llm_start → llm_end
│   ├── Span: tool_start → tool_end
│   └── Span: agent_end
```

Each `Span` carries: `span_id`, `trace_id`, `parent_span`, `event`, `resource`, `framework`, `project_name`, `user`, `tags`, `metadata`, `model`, `tokens`, `latency_ms`, `error`, timestamps.