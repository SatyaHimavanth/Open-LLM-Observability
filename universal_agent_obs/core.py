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


# ── Thread-local + asyncio context ──────────────────────────────────────────
_local = threading.local()

def _current_trace() -> Optional[str]:
    """Return the current trace_id for this thread/task."""
    # Check asyncio task first
    try:
        task = asyncio.current_task()
        if task and hasattr(task, "_obs_trace_id"):
            return task._obs_trace_id
    except RuntimeError:
        pass
    return getattr(_local, "trace_id", None)

def _current_span() -> Optional[str]:
    try:
        task = asyncio.current_task()
        if task and hasattr(task, "_obs_span_id"):
            return task._obs_span_id
    except RuntimeError:
        pass
    return getattr(_local, "span_id", None)

def _task_attr(name: str, default=None):
    try:
        task = asyncio.current_task()
        if task and hasattr(task, name):
            return getattr(task, name)
    except RuntimeError:
        pass
    return default

def _current_project() -> str:
    return _task_attr("_obs_project_name") or getattr(_local, "project_name", None) or PROJECT_NAME

def _current_user() -> Optional[dict]:
    return _task_attr("_obs_user") or getattr(_local, "user", None)

def _current_tags() -> Optional[list]:
    tags = _task_attr("_obs_tags") or getattr(_local, "tags", None)
    return list(tags) if tags else None

def _current_metadata() -> Optional[dict]:
    metadata = _task_attr("_obs_metadata") or getattr(_local, "metadata", None)
    return dict(metadata) if metadata else None

def _set_context(
    *,
    project_name: Optional[str] = None,
    user: Optional[dict] = None,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None,
):
    """Set trace attributes for the current thread and asyncio task."""
    previous = (_current_project(), _current_user(), _current_tags(), _current_metadata())
    project_name = project_name or previous[0] or PROJECT_NAME
    user = user if user is not None else previous[1]
    tags = tags if tags is not None else previous[2]
    metadata = metadata if metadata is not None else previous[3]

    _local.project_name = project_name
    _local.user = _safe_json(user)
    _local.tags = _safe_json(tags)
    _local.metadata = _safe_json(metadata)
    try:
        task = asyncio.current_task()
        if task:
            task._obs_project_name = project_name
            task._obs_user = _local.user
            task._obs_tags = _local.tags
            task._obs_metadata = _local.metadata
    except RuntimeError:
        pass
    return previous

def _restore_context(previous):
    project_name, user, tags, metadata = previous or (PROJECT_NAME, None, None, None)
    _local.project_name = project_name or PROJECT_NAME
    _local.user = user
    _local.tags = tags
    _local.metadata = metadata
    try:
        task = asyncio.current_task()
        if task:
            task._obs_project_name = _local.project_name
            task._obs_user = user
            task._obs_tags = tags
            task._obs_metadata = metadata
    except RuntimeError:
        pass

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
    """Set trace context for current thread and asyncio task."""
    previous = (_current_trace(), _current_span())
    _local.trace_id = trace_id
    _local.span_id  = span_id
    try:
        task = asyncio.current_task()
        if task:
            task._obs_trace_id = trace_id
            task._obs_span_id  = span_id
    except RuntimeError:
        pass
    return previous

def _restore_trace(previous):
    """Restore trace context saved by _set_trace."""
    trace_id, span_id = previous or (None, None)
    _local.trace_id = trace_id
    _local.span_id = span_id
    try:
        task = asyncio.current_task()
        if task:
            task._obs_trace_id = trace_id
            task._obs_span_id = span_id
    except RuntimeError:
        pass

def new_trace() -> str:
    tid = str(uuid.uuid4())
    _local.trace_id = tid
    _local.span_id  = None
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
    while True:
        span_dict = None
        try:
            span_dict = _emit_queue.get(timeout=1)
            if span_dict is None:   # shutdown signal
                break
            _post_span(span_dict)
        except queue.Empty:
            continue
        except Exception:
            pass   # observability must never crash the agent
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
