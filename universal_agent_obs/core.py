"""
core.py — Span context, emitter, and shared utilities.
All framework interceptors import from here.
"""

import atexit
import contextlib
import os, time, uuid, json, queue, threading, asyncio, traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# ── Config (from env) ────────────────────────────────────────────────────────
SERVER_URL  = os.getenv("AGENT_OBS_URL",  "http://localhost:4317")
FRAMEWORK   = os.getenv("AGENT_FRAMEWORK", "custom")
ENABLED     = os.getenv("AGENT_OBS", "1") not in ("0", "false", "False", "no")
PROJECT_NAME = os.getenv("AGENT_OBS_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "default"
CLIENT_ID   = os.getenv("AGENT_OBS_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("AGENT_OBS_CLIENT_SECRET", "")

def _client_credentials() -> tuple[str, str]:
    return (
        os.getenv("AGENT_OBS_CLIENT_ID", CLIENT_ID),
        os.getenv("AGENT_OBS_CLIENT_SECRET", CLIENT_SECRET),
    )

if ENABLED and not all(_client_credentials()):
    print("\n[WARNING] AGENT_OBS_CLIENT_ID and AGENT_OBS_CLIENT_SECRET are not set.")
    print("[WARNING] Traces will not be loaded into your dashboard. Please log in to your Agent Observability dashboard to get your credentials.\n")

# ── Span dataclass ───────────────────────────────────────────────────────────
@dataclass
class Span:
    span_id:      str            = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id:     Optional[str]  = None
    parent_span:  Optional[str]  = None
    event:        str            = ""          # e.g. llm_start, tool_end
    resource:     str            = ""          # llm | tool | agent | chain
    framework:    str            = FRAMEWORK
    project_name: Optional[str]  = None
    user:         Optional[dict] = None
    tags:         Optional[list] = None
    metadata:     Optional[dict] = None
    agent_name:   Optional[str]  = None
    model:        Optional[str]  = None
    provider:     Optional[str]  = None
    messages:     Optional[list] = None
    response:     Optional[dict] = None
    tool_name:    Optional[str]  = None
    tool_input:   Optional[dict] = None
    tool_output:  Optional[str]  = None
    tokens:       Optional[dict] = None
    latency_ms:   Optional[float]= None
    error:        Optional[str]  = None
    meta:         dict           = field(default_factory=dict)
    ts_start:     float          = field(default_factory=time.time)
    ts_end:       Optional[float]= None

    def end(self):
        self.ts_end    = time.time()
        self.latency_ms = (self.ts_end - self.ts_start) * 1000
        return self

    def to_dict(self):
        data = asdict(self)
        data["project_name"] = data.get("project_name") or _current_project()
        data["user"] = data.get("user") or _current_user()
        data["tags"] = data.get("tags") or _current_tags()
        data["metadata"] = data.get("metadata") or _current_metadata()
        return {k: v for k, v in data.items() if v is not None}


# ── contextvars context ─────────────────────────────────────────────────────
from contextvars import ContextVar

_trace_id_var:     ContextVar[Optional[str]] = ContextVar("obs_trace_id",     default=None)
_span_id_var:      ContextVar[Optional[str]] = ContextVar("obs_span_id",      default=None)
_project_name_var: ContextVar[str]           = ContextVar("obs_project_name", default=PROJECT_NAME)
_user_var:         ContextVar[Optional[dict]] = ContextVar("obs_user",         default=None)
_tags_var:         ContextVar[Optional[list]] = ContextVar("obs_tags",         default=None)
_metadata_var:     ContextVar[Optional[dict]] = ContextVar("obs_metadata",     default=None)
_framework_var:    ContextVar[Optional[str]] = ContextVar("obs_framework",    default=None)

def _current_trace() -> Optional[str]:
    return _trace_id_var.get()

def _current_span() -> Optional[str]:
    return _span_id_var.get()

def _current_project() -> str:
    return _project_name_var.get() or PROJECT_NAME

def _current_framework() -> Optional[str]:
    return _framework_var.get()

def _current_user() -> Optional[dict]:
    return _user_var.get()

def _current_tags() -> Optional[list]:
    tags = _tags_var.get()
    return list(tags) if tags else None

def _current_metadata() -> Optional[dict]:
    metadata = _metadata_var.get()
    return dict(metadata) if metadata else None

def _set_context(
    *,
    project_name: Optional[str] = None,
    user: Optional[dict] = None,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None,
    framework: Optional[str] = None,
):
    """Set trace attributes for the current context."""
    previous_values = (
        _project_name_var.get(),
        _user_var.get(),
        _tags_var.get(),
        _metadata_var.get(),
    )
    tokens = []
    if project_name is not None:
        tokens.append(_project_name_var.set(project_name))
    if user is not None:
        tokens.append(_user_var.set(_safe_json(user)))
    if tags is not None:
        tokens.append(_tags_var.set(_safe_json(tags)))
    if metadata is not None:
        tokens.append(_metadata_var.set(_safe_json(metadata)))
    if framework is not None:
        tokens.append(_framework_var.set(framework))
    return (previous_values, tokens)

def _restore_context(state):
    if not state or not isinstance(state, tuple) or len(state) != 2:
        return
    _, tokens = state
    for token in reversed(tokens):
        token.var.reset(token)

def set_context(
    *,
    project_name: Optional[str] = None,
    user: Optional[dict] = None,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None,
):
    """Attach project, user, tags, and metadata to subsequently emitted spans."""
    return _set_context(project_name=project_name, user=user, tags=tags, metadata=metadata)

@contextlib.contextmanager
def trace_context(
    *,
    project_name: Optional[str] = None,
    user: Optional[dict] = None,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None,
):
    previous = _set_context(project_name=project_name, user=user, tags=tags, metadata=metadata)
    try:
        yield
    finally:
        _restore_context(previous)

def _safe_json(value: Any):
    if value is None:
        return None
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)[:500]

def _set_trace(trace_id: str, span_id: str):
    """Set trace context for current context."""
    old_trace = _trace_id_var.get()
    old_span  = _span_id_var.get()
    t1 = _trace_id_var.set(trace_id)
    t2 = _span_id_var.set(span_id)
    return (old_trace, old_span, t1, t2)

def _restore_trace(state):
    """Restore trace context saved by _set_trace."""
    if not state or len(state) != 4:
        return
    _, _, t1, t2 = state
    _trace_id_var.reset(t1)
    _span_id_var.reset(t2)

def new_trace() -> str:
    tid = str(uuid.uuid4())
    _trace_id_var.set(tid)
    _span_id_var.set(None)
    return tid

def get_or_new_trace() -> str:
    tid = _current_trace()
    if not tid:
        tid = new_trace()
    return tid

def detect_provider(model: str) -> str:
    if not model: return "unknown"
    m = model.lower()
    if any(x in m for x in ("gpt", "o1", "o3", "o4")): return "openai"
    if "claude" in m:   return "anthropic"
    if "gemini" in m:   return "google"
    if "mistral" in m:  return "mistral"
    if "llama" in m:    return "meta"
    if "command" in m:  return "cohere"
    return "unknown"

def detect_framework_from_stack() -> str:
    """Walk the call stack to identify which framework is active."""
    import re
    frames = "".join(f.filename for f in traceback.extract_stack())
    patterns = {
        "langchain":  r"langchain",
        "crewai":     r"crewai",
        "autogen":    r"autogen",
        "google_adk": r"google[/\\]adk|google\.adk",
        "openai_agents": r"agents[/\\]",
        "llamaindex": r"llama_index",
    }
    for name, pat in patterns.items():
        if re.search(pat, frames, re.I):
            return name
    return FRAMEWORK


# ── Async emit queue ─────────────────────────────────────────────────────────
_emit_queue: queue.Queue = queue.Queue(maxsize=10_000)

def _post_span(span_dict: dict):
    import urllib.request

    data = json.dumps(span_dict).encode()
    headers = {"Content-Type": "application/json"}
    client_id, client_secret = _client_credentials()
    if client_id and client_secret:
        headers["x-client-id"] = client_id
        headers["x-client-secret"] = client_secret

    req  = urllib.request.Request(
        f"{SERVER_URL}/ingest",
        data=data,
        headers=headers,
        method="POST",
    )
    urllib.request.urlopen(req, timeout=5)


def _worker():
    _auth_warned = False
    while True:
        span_dict = None
        try:
            span_dict = _emit_queue.get(timeout=1)
            if span_dict is None:   # shutdown signal
                break
            _post_span(span_dict)
        except queue.Empty:
            continue
        except Exception as e:
            if not _auth_warned and "401" in str(e):
                print(f"[agent-obs] WARNING: Ingest rejected (401 Unauthorized). Set AGENT_OBS_CLIENT_ID and AGENT_OBS_CLIENT_SECRET env vars.")
                _auth_warned = True
        finally:
            if span_dict is not None:
                _emit_queue.task_done()

_worker_thread = threading.Thread(target=_worker, daemon=True, name="obs-emitter")
_worker_thread.start()

def emit(span: Span):
    """Non-blocking emit — drops silently if queue is full."""
    if not ENABLED:
        return
    try:
        _emit_queue.put_nowait(span.to_dict())
    except queue.Full:
        pass

def flush(timeout: float = 10.0):
    """Best-effort flush for short-lived agent processes."""
    if not ENABLED:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _emit_queue.unfinished_tasks == 0:
            return
        time.sleep(0.05)


atexit.register(flush)