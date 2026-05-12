# Universal Agent Observability

Universal Agent Observability traces Python LLM calls, agent runs, tool calls, and multi-agent workflows with a small import in your application.

It ships with:

- LangChain/LangGraph tracing for model, agent stream, and tool spans
- HTTP fallback tracing for direct LLM SDK calls
- A FastAPI collector with a live browser UI
- SQLite persistence through SQLAlchemy, with a Postgres-ready database URL
- Project, user, tag, and metadata attribution for LangSmith-style run grouping

---

## Sample Images

### Login Page
![Login Page](assets/Login_Page.png)
### Home Page
![Home Page](assets/Home_Page.png)
### Traces Page
![Traces Page](assets/Traces_Page.png)
### Example Trace
![Example Trace](assets/Example_Trace.png)
### Settings Page
![Settings Page](assets/Settings_Page.png)

---

## Quick Start

Install the core collector/tracer:

```bash
uv sync
```

Start the collector:

```bash
uv run agent-obs
```

Then open the dashboard:

```text
http://localhost:4317
```

* Log in with default credentials `admin` / `password`.
* Go to **Settings** to retrieve your unique **Client ID** and **Client Secret**.

Configure your environment with these credentials:

```bash
export AGENT_OBS_URL=http://localhost:4317
export AGENT_OBS_CLIENT_ID=<your-client-id>
export AGENT_OBS_CLIENT_SECRET=<your-client-secret>
```

Run the included LangChain test agent:

```bash
uv sync --extra agent
uv run -m samples.langchain.simple_agent.agent
```

The core install only includes `fastapi`, `httpx`, `uvicorn`, and `sqlalchemy`.
Framework libraries are optional so the observability module can stay lightweight.

## Optional Installs

Use extras for the frameworks you want to trace:

```bash
# LangChain/LangGraph interceptor support
uv sync --extra langchain

# Google ADK and Google GenAI support
uv sync --extra google

# LiteLLM-backed cost calculation
uv sync --extra cost

# CrewAI support
uv sync --extra crewai

# AutoGen support
uv sync --extra autogen

# OpenAI Agents SDK support
uv sync --extra openai-agents

# Postgres storage driver
uv sync --extra postgres

# Everything
uv sync --all-extras
```

## Add additional optional dependencies
```bash
uv add --optional <extra_name> <package>
```

## Add Tracing To An Agent

Import the package before creating your model, tools, or agent:

```python
import universal_agent_obs

# your normal agent code
```

The collector URL defaults to `http://localhost:4317`.

## Projects, Users, Tags, And Metadata

Set the project name once in the environment:

```powershell
$env:AGENT_OBS_PROJECT="default"
uv run agent-obs
```

For LangChain/LangGraph runs, pass user details, tags, and metadata through the normal runnable config:

```python
config = {
    "tags": ["production", "support-agent"],
    "metadata": {
        "user": {
            "id": "user-123",
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "account": "acme",
            "role": "admin",
        },
        "environment": "local",
    },
}

agent.invoke({"messages": [{"role": "user", "content": "Hello"}]}, config=config)
```

You can also use the explicit callback helper:

```python
from universal_agent_obs.langchain import TraceContextCallbackHandler

config = {
    "callbacks": [
        TraceContextCallbackHandler(
            user={"email": "ada@example.com", "role": "admin"},
            tags=["support-agent"],
        )
    ]
}
```

The UI shows project and user labels in the run list and trace detail. The API also exposes project rollups for dashboard-style views.

## Google ADK And Google GenAI Context

Google traces accept the same kind of context through a lightweight callback helper:

```python
from universal_agent_obs.google import TraceContextCallbackHandler

trace_callback = TraceContextCallbackHandler(
    user={
        "id": "demo-user",
        "name": "Demo User",
        "email": "demo.user@example.com",
        "account": "demo-account",
        "role": "tester",
    },
    tags=["sample", "google"],
    metadata={"environment": "local"},
)
```

Pass it to direct Google GenAI calls:

```python
client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Hi",
    callbacks=[trace_callback],
)
```

For ADK, use a runner and pass the callback into `run_async`:

```python
async for event in runner.run_async(
    user_id="demo-user",
    session_id=session.id,
    new_message=message,
    callbacks=[trace_callback],
):
    ...
```

## CrewAI Context

CrewAI traces accept the same lightweight trace context via the provided callback helper. Create a `TraceContextCallbackHandler` and pass it into your CrewAI router or agent invocation as part of the `callbacks` list.

```python
from universal_agent_obs.crewai import TraceContextCallbackHandler

trace_callback = TraceContextCallbackHandler(
    user={
        "id": "demo-user",
        "name": "Demo User",
        "email": "demo.user@example.com",
    },
    tags=["sample", "crewai"],
    metadata={"environment": "local"},
)

# Pass `trace_callback` into your CrewAI invocation.
# Example (pseudo): router_agent.kickoff(input, callbacks=[trace_callback])
```

## OpenAI SDK / OpenAI Agents Context

Use the OpenAI helper when instrumenting direct OpenAI SDK calls or the OpenAI Agents `Runner`. The handler works with both chat completion calls and agent runs — pass it in via the `callbacks` parameter.

```python
from universal_agent_obs.openai import TraceContextCallbackHandler

trace_callback = TraceContextCallbackHandler(
    user={"id": "demo-user", "name": "Demo User", "email": "demo.user@example.com"},
    tags=["sample", "openai"],
    metadata={"environment": "local"},
)

# OpenAI SDK chat completion example:
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather?"}],
    callbacks=[trace_callback],
)

# OpenAI Agents runner example:
# runner.run(..., callbacks=[trace_callback])
```

## Configuration

| Env var | Default | Description |
|---|---|---|
| `AGENT_OBS` | `1` | Set to `0`, `false`, or `no` to disable tracing. |
| `AGENT_OBS_URL` | `http://localhost:4317` | Collector ingest URL used by instrumented agents. |
| `AGENT_OBS_CLIENT_ID` | `""` | The client ID provided in your user settings for API auth. |
| `AGENT_OBS_CLIENT_SECRET` | `""` | The client secret provided in your user settings for API auth. |
| `AGENT_FRAMEWORK` | `custom` | Framework label for spans without a framework-specific interceptor. |
| `AGENT_OBS_PROJECT` | `default` | Project name assigned to emitted traces. `LANGSMITH_PROJECT` is honored as a fallback. |
| `AGENT_OBS_DB_URL` | `sqlite:///./agent_obs.sqlite3` | SQLAlchemy database URL for collector persistence. |

SQLite is used by default. To use Postgres later:

```bash
uv sync --extra postgres
export AGENT_OBS_DB_URL=postgresql+psycopg://user:password@localhost:5432/agent_obs
uv run agent-obs
```

On PowerShell:

```powershell
$env:AGENT_OBS_DB_URL="postgresql+psycopg://user:password@localhost:5432/agent_obs"
uv run agent-obs
```

## Server Commands

```bash
# Package module
uv run -m universal_agent_obs

# Server submodule
uv run -m universal_agent_obs.server

# CLI script
uv run agent-obs --host 0.0.0.0 --port 4317
```

## API Endpoints

```text
POST   /ingest        Receive spans from agents
POST   /register      Create a new user account
POST   /login         Authenticate and fetch user details
GET    /traces        List trace summaries
GET    /traces/{id}   Get one trace with spans and tree
GET    /projects      List project summaries
POST   /projects      Create a new project
GET    /spans         List raw spans
GET    /stats         Server statistics
DELETE /traces        Archive all traces
DELETE /traces/{id}   Archive one trace
DELETE /projects/{name} Archive all traces in a project
DELETE /admin/traces  Permanently clear traces
WS     /ws            Live trace updates
GET    /              Trace viewer UI
```

## Project Layout

```text
samples/
  langchain_example/
    simple_llm/
        llm.py
    simple_agent/
        __init__.py
        __main__.py
        agent.py
        prompts.py
        tools.py
    multi_agent/
        __init__.py
        __main__.py
        agent.py
        prompts.py
        tools.py
    llms.py
  google_example/
    simple_llm/
    simple_agent/

universal_agent_obs/
  __init__.py         # Auto-installs interceptors on import
  __main__.py         # uv run -m universal_agent_obs
  core.py             # Span dataclass, context, emit queue
  langchain.py        # LangChain trace context callback helper
  interceptors/
    langchain.py      # LangChain/LangGraph tracing
    http.py           # HTTP LLM fallback tracing
    google_adk.py     # Google ADK runner/event tracing
    google_genai.py   # Google GenAI SDK tracing
    crewai.py         # CrewAI hooks
    agents.py         # AutoGen/OpenAI Agents hooks
  server/
    app.py            # FastAPI collector
    storage.py        # SQLAlchemy persistence
    static/index.html # Trace viewer UI
```

## Manual Spans

```python
import universal_agent_obs as obs

trace_id = obs.new_trace()
span = obs.Span(
    trace_id=trace_id,
    event="custom_step",
    resource="agent",
    agent_name="MyAgent",
    meta={"step": "lookup"},
).end()
obs.emit(span)
```

## Current Status:
- Tested simple langchain agent and multi agent and observed the traces in the dashboard.
- need to add crewai and other agents test cases.
