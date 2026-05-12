import asyncio
import os
import time
import uuid
from importlib.resources import files
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .storage import (
    archive_all,
    archive_project,
    archive_trace,
    clear_all,
    get_session,
    get_trace_spans,
    init_db,
    list_span_payloads,
    list_trace_ids,
    upsert_span,
    UserRecord,
    ProjectRecord,
)

from pydantic import BaseModel



app = FastAPI(title="Agent Observability", version="0.1.0")
DEFAULT_PROJECT_NAME = os.getenv("AGENT_OBS_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "default"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# Create default user if not exists
with get_session() as session:
    if not session.query(UserRecord).filter_by(username="admin").first():
        default_user = UserRecord(
            id=str(uuid.uuid4()),
            username="admin",
            password="password",
            client_id=str(uuid.uuid4()),
            client_secret=str(uuid.uuid4()),
        )
        session.add(default_user)
        session.commit()

_ws_clients: list[tuple[WebSocket, Optional[str]]] = []


async def _broadcast(data: dict, user_id: Optional[str] = None):
    dead = []
    for ws, uid in _ws_clients:
        if not uid or not user_id or uid == user_id:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append((ws, uid))
    for item in dead:
        if item in _ws_clients:
            _ws_clients.remove(item)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, user_id: Optional[str] = None):
    await websocket.accept()
    _ws_clients.append((websocket, user_id))
    with get_session() as session:
        await websocket.send_json({
            "type": "init",
            "traces": [],
            "projects": _project_summaries(session, user_id=user_id),
        })
    try:
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        if (websocket, user_id) in _ws_clients:
            _ws_clients.remove((websocket, user_id))


class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(req: LoginRequest):
    with get_session() as session:
        user = session.query(UserRecord).filter_by(username=req.username, password=req.password).first()
        if not user:
            raise HTTPException(401, "Invalid credentials")
        return {
            "id": user.id,
            "username": user.username,
            "client_id": user.client_id,
            "client_secret": user.client_secret,
        }

class RegisterRequest(BaseModel):
    username: str
    password: str

@app.post("/register")
def register(req: RegisterRequest):
    with get_session() as session:
        if session.query(UserRecord).filter_by(username=req.username).first():
            raise HTTPException(400, "User already exists")
        new_user = UserRecord(
            id=str(uuid.uuid4()),
            username=req.username,
            password=req.password,
            client_id=str(uuid.uuid4()),
            client_secret=str(uuid.uuid4()),
        )
        session.add(new_user)
        session.commit()
        return {
            "id": new_user.id,
            "username": new_user.username,
            "client_id": new_user.client_id,
            "client_secret": new_user.client_secret,
        }

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

@app.post("/ingest")
async def ingest(request: Request, payload: dict):
    # Retrieve client id and secret from headers
    client_id = request.headers.get("x-client-id")
    client_secret = request.headers.get("x-client-secret")
    
    with get_session() as session:
        user = None
        if client_id and client_secret:
            user = session.query(UserRecord).filter_by(client_id=client_id, client_secret=client_secret).first()
            
        if not user:
            raise HTTPException(401, "Invalid or missing client credentials")

    span = _enrich(payload)
    if user:
        span["system_user_id"] = user.id

    with get_session() as session:
        upsert_span(session, span)
        summary = _trace_summary(session, span["trace_id"])

    # Extract user_id from span to broadcast to correct user
    span_user_id = span.get("system_user_id")
    await _broadcast({"type": "span", "span": span, "summary": summary}, user_id=span_user_id)
    return {"ok": True, "span_id": span["span_id"], "trace_id": span["trace_id"]}



@app.get("/traces")
def list_traces(
    page: int = 1,
    per_page: int = 50,
    limit: Optional[int] = None,
    framework: Optional[str] = None,
    project: Optional[str] = None,
    q: Optional[str] = None,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None,
    include_archived: bool = False,
):
    page = max(page, 1)
    per_page = min(max(limit or per_page, 1), 100)
    with get_session() as session:
        return _list_trace_page(
            session,
            page=page,
            per_page=per_page,
            framework=framework,
            project=project,
            q=q,
            user_email=user_email,
            user_id=user_id,
            include_archived=include_archived,
        )


@app.get("/projects")
def list_projects(user_id: Optional[str] = None, include_archived: bool = False):
    with get_session() as session:
        # Include explicit projects and inferred ones
        if user_id:
            explicit_projects = session.query(ProjectRecord).filter_by(user_id=user_id).all()
        else:
            explicit_projects = session.query(ProjectRecord).all()
        summaries = _project_summaries(session, user_id=user_id, include_archived=include_archived)
        
        # Merge explicit projects that might not have traces yet
        summary_names = {s["name"] for s in summaries}
        for p in explicit_projects:
            if p.name not in summary_names:
                summaries.append({
                    "name": p.name,
                    "trace_count": 0,
                    "most_recent_run": p.created_at,
                    "error_rate": 0,
                    "p50_latency_ms": 0,
                    "p99_latency_ms": 0,
                    "total_tokens": 0,
                    "total_cost": 0,
                    "users": [],
                })
        return sorted(summaries, key=lambda x: x["most_recent_run"], reverse=True)

class ProjectCreateRequest(BaseModel):
    name: str

@app.post("/projects")
def create_project(req: ProjectCreateRequest, user_id: Optional[str] = None):
    with get_session() as session:
        if session.query(ProjectRecord).filter_by(name=req.name).first():
            raise HTTPException(400, "Project already exists")
        new_project = ProjectRecord(
            id=str(uuid.uuid4()),
            name=req.name,
            user_id=user_id or "default"
        )
        session.add(new_project)
        session.commit()
        return {"id": new_project.id, "name": new_project.name}


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str, include_archived: bool = False):
    with get_session() as session:
        spans = get_trace_spans(session, trace_id, include_archived=include_archived)
        if not spans:
            raise HTTPException(404, "trace not found")
        return {
            "summary": _summary_from_spans(trace_id, spans),
            "spans": sorted(spans, key=lambda s: s.get("ts_start", 0)),
            "tree": _build_tree(spans),
        }


@app.get("/spans")
def list_spans(trace_id: Optional[str] = None, limit: int = 500, include_archived: bool = False):
    with get_session() as session:
        return list_span_payloads(
            session,
            trace_id=trace_id,
            limit=limit,
            include_archived=include_archived,
        )


@app.get("/stats")
def stats():
    with get_session() as session:
        trace_ids = list_trace_ids(session, limit=100_000)
        summaries = [_trace_summary(session, tid) for tid in trace_ids]
    return {
        "traces": len(summaries),
        "spans": sum(s["span_count"] for s in summaries),
        "total_cost": round(sum(s["total_cost"] for s in summaries), 8),
        "frameworks": sorted({s["framework"] for s in summaries if s.get("framework")}),
        "projects": sorted({s["project_name"] for s in summaries if s.get("project_name")}),
        "ws_clients": len(_ws_clients),
    }


@app.delete("/traces")
def archive_traces():
    with get_session() as session:
        archived = archive_all(session)
    return {"ok": True, "archived": archived}


@app.delete("/traces/{trace_id}")
def archive_one_trace(trace_id: str):
    with get_session() as session:
        archived = archive_trace(session, trace_id)
    if not archived:
        raise HTTPException(404, "trace not found")
    return {"ok": True, "archived": archived, "trace_id": trace_id}


@app.delete("/projects/{project_name}")
def archive_one_project(project_name: str):
    with get_session() as session:
        archived = archive_project(session, project_name)
    if not archived:
        raise HTTPException(404, "project not found")
    return {"ok": True, "archived": archived, "project_name": project_name}


@app.delete("/admin/traces")
def clear_traces_permanently():
    with get_session() as session:
        clear_all(session)
    return {"ok": True, "permanent": True}


@app.get("/", response_class=HTMLResponse)
def ui():
    html = files("universal_agent_obs.server.static").joinpath("index.html").read_text(
        encoding="utf-8"
    )
    return HTMLResponse(content=html)


def _enrich(span: dict) -> dict:
    span.setdefault("span_id", str(uuid.uuid4()))
    span.setdefault("trace_id", span["span_id"])
    span.setdefault("project_name", DEFAULT_PROJECT_NAME)
    span.setdefault("tags", [])
    span.setdefault("metadata", {})
    span.setdefault("archived", False)
    span.setdefault("ts_start", time.time())
    span["received_at"] = time.time()

    tokens = span.get("tokens") or {}
    model = span.get("model") or _model_from_meta(span.get("meta") or "")
    cost = _estimate_cost(
        model,
        tokens.get("prompt") or 0,
        tokens.get("completion") or 0,
    )
    if cost is not None:
        span["cost_usd"] = cost

    return span


def _model_from_meta(meta: dict) -> str:
    if not isinstance(meta, dict):
        return ""
    for params in (meta.get("invocation_params") or {}, meta.get("options") or {}):
        if not isinstance(params, dict):
            continue
        model = (
            params.get("model")
            or params.get("model_name")
            or params.get("ls_model_name")
            or params.get("deployment_name")
        )
        if model:
            return str(model)
    return ""


def _estimate_cost(model: str, prompt: int, completion: int) -> Optional[float]:
    litellm_cost = _estimate_cost_litellm(model, prompt, completion)
    if litellm_cost is not None:
        return litellm_cost

    model = (model or "").lower()
    prices = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.5, 10.0),
        "o1-preview": (15.0, 60.0),
        "o1-mini": (1.1, 4.4),
        "o1": (15.0, 60.0),
        "gpt-4.1-nano": (0.10, 0.40),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1": (2.00, 8.00),
        "claude-3-5-sonnet": (3.0, 15.0),
        "claude-3-5-haiku": (0.25, 1.25),
        "claude-3-opus": (15.0, 75.0),
        "claude-3-sonnet": (3.0, 15.0),
        "claude-3-haiku": (0.25, 1.25),
        "gpt-4-turbo": (10.0, 30.0),
        "gpt-3.5": (0.5, 1.5),
        "gemini-1.5-pro": (1.25, 5.0), # Updated pricing
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-2.0-flash": (0.10, 0.40),
        "mistral-large": (2.0, 6.0),
    }
    for key, (inp, out) in prices.items():
        if key in model:
            return round((prompt * inp + completion * out) / 1_000_000, 8)
    return None


def _estimate_cost_litellm(model: str, prompt: int, completion: int) -> Optional[float]:
    if not model or not (prompt or completion):
        return None
    try:
        from litellm import completion_cost

        return round(float(completion_cost(
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
        )), 8)
    except Exception:
        return None


def _list_trace_summaries(
    session,
    limit: int = 100,
    offset: int = 0,
    framework: Optional[str] = None,
    project: Optional[str] = None,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None,
    include_archived: bool = False,
) -> list[dict]:
    summaries = [
        _trace_summary(session, tid, include_archived=include_archived)
        for tid in list_trace_ids(
            session,
            limit=limit,
            offset=offset,
            framework=framework,
            project=project,
            user_email=user_email,
            user_id=user_id,
            include_archived=include_archived,
        )
    ]
    return summaries


def _list_trace_page(
    session,
    page: int = 1,
    per_page: int = 50,
    framework: Optional[str] = None,
    project: Optional[str] = None,
    q: Optional[str] = None,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None,
    include_archived: bool = False,
) -> dict:
    # Search and aggregate stats are summary-level concepts, so this keeps
    # frontend payloads paged while still reporting totals across the filter.
    all_ids = list_trace_ids(
        session,
        limit=100_000,
        framework=framework,
        project=project,
        user_email=user_email,
        user_id=user_id,
        include_archived=include_archived,
    )
    summaries = [_trace_summary(session, tid, include_archived=include_archived) for tid in all_ids]
    if q:
        needle = q.lower()
        summaries = [summary for summary in summaries if _summary_matches(summary, needle)]

    total = len(summaries)
    offset = (page - 1) * per_page
    items = summaries[offset:offset + per_page]
    return {
        "items": items,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": max(1, (total + per_page - 1) // per_page),
        "stats": _trace_collection_stats(summaries),
    }


def _summary_matches(summary: dict, needle: str) -> bool:
    user = summary.get("user") or {}
    haystack = " ".join([
        str(summary.get("trace_id") or ""),
        str(summary.get("framework") or ""),
        str(summary.get("project_name") or ""),
        " ".join(summary.get("agents") or []),
        " ".join(summary.get("models") or []),
        " ".join(summary.get("tags") or []),
        str(user.get("id") or ""),
        str(user.get("name") or ""),
        str(user.get("email") or ""),
    ]).lower()
    return needle in haystack


def _trace_collection_stats(summaries: list[dict]) -> dict:
    total = len(summaries)
    done = sum(1 for s in summaries if s.get("status") == "done")
    errors = sum(1 for s in summaries if s.get("status") == "error")
    return {
        "total_cost": sum(s.get("total_cost") or 0 for s in summaries),
        "success_count": done,
        "error_count": errors,
        "success_rate": round(done / total, 4) if total else 0,
        "error_rate": round(errors / total, 4) if total else 0,
    }


def _project_summaries(session, user_id: Optional[str] = None, include_archived: bool = False) -> list[dict]:
    trace_ids = list_trace_ids(session, limit=100_000, user_id=user_id, include_archived=include_archived)
    summaries = [_trace_summary(session, tid, include_archived=include_archived) for tid in trace_ids]
    grouped: dict[str, list[dict]] = {}
    for summary in summaries:
        grouped.setdefault(summary.get("project_name") or DEFAULT_PROJECT_NAME, []).append(summary)

    projects = []
    for name, rows in grouped.items():
        latencies = sorted((r.get("duration_ms") or 0) for r in rows)
        error_count = sum(1 for r in rows if r.get("status") == "error")
        projects.append({
            "name": name,
            "trace_count": len(rows),
            "most_recent_run": max(r.get("updated_at") or 0 for r in rows),
            "error_rate": round(error_count / len(rows), 4) if rows else 0,
            "p50_latency_ms": _percentile(latencies, 50),
            "p99_latency_ms": _percentile(latencies, 99),
            "total_tokens": sum((r.get("total_tokens") or {}).get("total") or 0 for r in rows),
            "total_cost": round(sum(r.get("total_cost") or 0 for r in rows), 6),
            "users": sorted(
                user
                for user in {
                    (r.get("user") or {}).get("email")
                    or (r.get("user") or {}).get("name")
                    or (r.get("user") or {}).get("id")
                    for r in rows
                    if r.get("user")
                }
                if user
            ),
        })
    return sorted(projects, key=lambda p: p["most_recent_run"], reverse=True)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0
    index = min(len(values) - 1, max(0, round((percentile / 100) * (len(values) - 1))))
    return values[index]


def _trace_summary(session, trace_id: str, include_archived: bool = False) -> dict:
    spans = get_trace_spans(session, trace_id, include_archived=include_archived)
    if not spans:
        raise HTTPException(404, "trace not found")
    return _summary_from_spans(trace_id, spans)


def _summary_from_spans(trace_id: str, spans: list[dict]) -> dict:
    started_at = min(s.get("ts_start", time.time()) for s in spans)
    updated_at = max(s.get("received_at", s.get("ts_start", started_at)) for s in spans)
    agents = sorted({s["agent_name"] for s in spans if s.get("agent_name")})
    models = sorted({s["model"] for s in spans if s.get("model")})
    project_name = next((s.get("project_name") for s in spans if s.get("project_name")), DEFAULT_PROJECT_NAME)
    user = next((s.get("user") for s in spans if s.get("user")), None)
    if user is None:
        system_user_id = next((s.get("system_user_id") for s in spans if s.get("system_user_id")), None)
        if system_user_id:
            user = {"id": system_user_id, "source": "system"}
    tags = sorted({
        tag
        for s in spans
        for tag in (s.get("tags") or [])
        if tag
    })
    metadata = {}
    for span in spans:
        if isinstance(span.get("metadata"), dict):
            metadata.update(span["metadata"])
    archived = all(s.get("archived") for s in spans)
    llm_calls = sum(
        1 for s in spans if s.get("resource") == "llm" and s.get("event") in {"llm_end", "llm_error"}
    )
    tool_calls = sum(
        1
        for s in spans
        if s.get("resource") == "tool" and s.get("event") in {"tool_end", "tool_error"}
    )
    total_tokens = {"prompt": 0, "completion": 0, "reasoning": 0, "total": 0}

    for span in spans:
        tokens = span.get("tokens") or {}
        total_tokens["prompt"] += tokens.get("prompt") or 0
        total_tokens["completion"] += tokens.get("completion") or 0
        total_tokens["reasoning"] += tokens.get("reasoning") or 0
        total_tokens["total"] += tokens.get("total") or 0

    status = "running"
    error = None
    root_starts = {s["span_id"] for s in spans if not s.get("parent_span")}
    all_events = {s.get("event") for s in spans}

    # Check if root-level spans completed
    root_completed = False
    for span in spans:
        if span.get("error"):
            status = "error"
            error = span["error"]
        elif span.get("event") in {
            "agent_finish",
            "conversation_end",
            "task_complete",
            "agent_stream_end",
            "agent_invoke_end",
            "agent_end",
            "handoff", # OpenAI Agents handoff might be terminal for a specific agent
        } and status != "error":
            # For handoff, only mark as done if it's a root-level event (not usually)
            # but we include it in terminal patterns.
            status = "done"
            root_completed = True
        elif span.get("event", "").endswith("_end") and (not span.get("parent_span") or span.get("parent_span") in root_starts) and status != "error":
            status = "done"
            root_completed = True

    # Detect cancelled/stale: has start events but no corresponding root-level end,
    # and hasn't received new spans in over 2 minutes (increased from 30s for long-running steps)
    if status == "running":
        staleness = time.time() - updated_at
        has_starts = any(e for e in all_events if e and ("_start" in e or e.startswith("llm_start") or e.startswith("chain_start")))
        if has_starts and staleness > 120:
            status = "cancelled"

    return {
        "trace_id": trace_id,
        "project_name": project_name,
        "user": user,
        "tags": tags,
        "metadata": metadata,
        "archived": archived,
        "framework": next((s.get("framework") for s in spans if s.get("framework")), "unknown"),
        "started_at": started_at,
        "updated_at": updated_at,
        "span_count": len(spans),
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "agents": agents,
        "models": models,
        "total_tokens": total_tokens,
        "total_cost": sum(s.get("cost_usd") or 0 for s in spans),
        "total_latency_ms": sum(s.get("latency_ms") or 0 for s in spans),
        "duration_ms": (updated_at - started_at) * 1000,
        "status": status,
        "error": error,
    }


def _build_tree(spans: list[dict]) -> list[dict]:
    by_id = {s["span_id"]: {**s, "children": []} for s in spans}
    roots = []
    for span in by_id.values():
        parent_id = span.get("parent_span")
        if parent_id and parent_id in by_id:
            by_id[parent_id]["children"].append(span)
        else:
            roots.append(span)
    return roots
