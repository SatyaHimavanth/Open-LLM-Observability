"""
Google ADK interceptor.

Patches google.adk Agent classes to inject tracing callbacks for
agent lifecycle, model calls, and tool executions.
"""

import time
import uuid
from typing import Any, Optional

from ..core import (
    Span,
    detect_provider,
    emit,
    _current_span,
    _current_trace,
    _current_user,
    _restore_context,
    _restore_trace,
    _set_context,
    _set_trace,
)

_INSTALLED = False


def install():
    try:
        _patch_adk()
        return True
    except ImportError:
        return False


def _patch_adk():
    global _INSTALLED
    if _INSTALLED:
        return

    from google.adk.runners import Runner

    if getattr(Runner.run_async, "_agent_obs_patched", False):
        return

    _orig_run_async = Runner.run_async

    async def _patched_run_async(self, *args, **kwargs):
        """Wrap Runner.run_async to emit agent lifecycle spans."""
        agent_instance = getattr(self, "agent", None)
        agent_label = getattr(agent_instance, "name", None) or type(agent_instance).__name__ if agent_instance else "ADKAgent"
        model_by_agent = _agent_model_map(agent_instance)
        agent_model = model_by_agent.get(agent_label)

        span_id = str(uuid.uuid4())
        trace_id = _current_trace() or span_id
        parent_span = _current_span()
        user_id = kwargs.get("user_id")
        previous_context = _set_context(user=_current_user() or ({"id": user_id} if user_id else None))
        previous = _set_trace(trace_id, span_id)
        t0 = time.perf_counter()
        input_message = _safe_message(kwargs.get("new_message"))

        emit(Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_span=parent_span,
            event="agent_invoke_start",
            resource="agent",
            framework="google-adk",
            agent_name=agent_label,
            model=agent_model,
            provider=detect_provider(agent_model),
            messages=[input_message] if input_message else None,
            meta={
                "capture_via": "google_adk_patch",
                "input": {"messages": [input_message]} if input_message else None,
                "agents": _agent_metadata(agent_instance),
            },
        ))

        try:
            async for event in _orig_run_async(self, *args, **kwargs):
                # Emit sub-spans for model/tool events from the ADK event stream
                _emit_adk_event(event, trace_id, span_id, agent_label, model_by_agent)
                yield event

            latency = (time.perf_counter() - t0) * 1000
            emit(Span(
                span_id=span_id + "-end",
                trace_id=trace_id,
                parent_span=span_id,
                event="agent_invoke_end",
                resource="agent",
                framework="google-adk",
                agent_name=agent_label,
                model=agent_model,
                provider=detect_provider(agent_model),
                latency_ms=latency,
                meta={"capture_via": "google_adk_patch"},
            ))
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            emit(Span(
                span_id=span_id + "-err",
                trace_id=trace_id,
                parent_span=span_id,
                event="agent_error",
                resource="agent",
                framework="google-adk",
                agent_name=agent_label,
                model=agent_model,
                provider=detect_provider(agent_model),
                latency_ms=latency,
                error=str(exc),
                meta={"capture_via": "google_adk_patch"},
            ))
            raise
        finally:
            _restore_trace(previous)
            _restore_context(previous_context)

    _patched_run_async._agent_obs_patched = True
    Runner.run_async = _patched_run_async
    _INSTALLED = True


def _emit_adk_event(
    event,
    trace_id: str,
    parent_span: str,
    agent_label: str,
    model_by_agent: dict[str, str],
):
    """Extract tracing info from ADK event stream events."""
    try:
        # ADK events have .author, .content (with .parts), etc.
        author = getattr(event, "author", None) or agent_label
        content = getattr(event, "content", None)

        if content is None:
            return

        parts = getattr(content, "parts", None) or []
        if not parts:
            return

        usage = getattr(event, "usage_metadata", None)
        model = model_by_agent.get(author) or model_by_agent.get(agent_label)
        model_meta = _event_meta(event)
        model_response = _model_response(parts)
        if usage and model_response:
            emit(Span(
                span_id=str(uuid.uuid4()),
                trace_id=trace_id,
                parent_span=parent_span,
                event="llm_end",
                resource="llm",
                framework="google-adk",
                agent_name=author,
                model=model,
                provider=detect_provider(model),
                response=model_response,
                tokens=_usage_tokens(usage),
                meta={**model_meta, "capture_via": "google_adk_event"},
                ts_start=getattr(event, "timestamp", None) or time.time(),
            ))

        for part in parts:
            # Tool call request (function_call)
            fc = getattr(part, "function_call", None)
            if fc:
                tool_name = getattr(fc, "name", None) or "unknown_tool"
                tool_args = getattr(fc, "args", None)
                emit(Span(
                    span_id=str(uuid.uuid4()),
                    trace_id=trace_id,
                    parent_span=parent_span,
                    event="tool_start",
                    resource="tool",
                    framework="google-adk",
                    agent_name=author,
                    tool_name=tool_name,
                    tool_input={"input": dict(tool_args) if tool_args else {}},
                    meta={**model_meta, "capture_via": "google_adk_event"},
                    ts_start=getattr(event, "timestamp", None) or time.time(),
                ))
                continue

            # Tool call response (function_response)
            fr = getattr(part, "function_response", None)
            if fr:
                tool_name = getattr(fr, "name", None) or "unknown_tool"
                tool_response = getattr(fr, "response", None)
                emit(Span(
                    span_id=str(uuid.uuid4()),
                    trace_id=trace_id,
                    parent_span=parent_span,
                    event="tool_end",
                    resource="tool",
                    framework="google-adk",
                    agent_name=author,
                    tool_name=tool_name,
                    tool_output=str(tool_response)[:2000] if tool_response else None,
                    meta={**model_meta, "capture_via": "google_adk_event"},
                    ts_start=getattr(event, "timestamp", None) or time.time(),
                ))
                continue
    except Exception:
        pass  # never crash the agent


def _agent_model_map(agent) -> dict[str, str]:
    models = {}

    def visit(node):
        if not node:
            return
        name = getattr(node, "name", None) or type(node).__name__
        model = _model_name(getattr(node, "model", None))
        if model:
            models[name] = model
        for child in getattr(node, "sub_agents", None) or []:
            visit(child)

    visit(agent)
    return models


def _agent_metadata(agent) -> list[dict]:
    agents = []

    def visit(node):
        if not node:
            return
        tools = []
        for tool in getattr(node, "tools", None) or []:
            tools.append({
                "name": getattr(tool, "__name__", None) or getattr(tool, "name", None) or type(tool).__name__,
                "description": getattr(tool, "__doc__", None) or getattr(tool, "description", None),
            })
        agents.append({
            "name": getattr(node, "name", None) or type(node).__name__,
            "description": getattr(node, "description", None),
            "instruction": getattr(node, "instruction", None),
            "model": _model_name(getattr(node, "model", None)),
            "tools": [tool for tool in tools if tool.get("name")],
        })
        for child in getattr(node, "sub_agents", None) or []:
            visit(child)

    visit(agent)
    return agents


def _model_name(model) -> Optional[str]:
    if model is None:
        return None
    if isinstance(model, str):
        return model
    return (
        getattr(model, "model", None)
        or getattr(model, "model_name", None)
        or getattr(model, "name", None)
        or str(model)
    )


def _usage_tokens(usage) -> Optional[dict]:
    if not usage:
        return None
    return {
        "prompt": getattr(usage, "prompt_token_count", None),
        "completion": getattr(usage, "candidates_token_count", None),
        "reasoning": getattr(usage, "thoughts_token_count", None) or 0,
        "tool": getattr(usage, "tool_use_prompt_token_count", None) or 0,
        "cached": getattr(usage, "cached_content_token_count", None) or 0,
        "total": getattr(usage, "total_token_count", None),
    }


def _model_response(parts) -> Optional[dict]:
    texts = []
    tool_calls = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(str(text))
        fc = getattr(part, "function_call", None)
        if fc:
            tool_calls.append({
                "name": getattr(fc, "name", None),
                "args": _safe(getattr(fc, "args", None)),
            })
    response = {}
    if texts:
        response["content"] = "\n".join(texts)[:4000]
    if tool_calls:
        response["tool_calls"] = tool_calls
    return response or None


def _event_meta(event) -> dict:
    finish_reason = getattr(event, "finish_reason", None)
    if hasattr(finish_reason, "name"):
        finish_reason = finish_reason.name
    return {
        key: value
        for key, value in {
            "event_id": getattr(event, "id", None),
            "invocation_id": getattr(event, "invocation_id", None),
            "branch": getattr(event, "branch", None),
            "partial": getattr(event, "partial", None),
            "turn_complete": getattr(event, "turn_complete", None),
            "finish_reason": finish_reason,
        }.items()
        if value not in (None, "")
    }


def _safe(value):
    if value is None:
        return None
    try:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return dict(value)
        return value
    except Exception:
        return str(value)[:500]


def _safe_message(message):
    if message is None:
        return None
    try:
        if hasattr(message, "model_dump"):
            data = message.model_dump()
        else:
            data = message
        return _normalize_content(data)
    except Exception:
        return {"role": "user", "content": str(message)[:4000]}


def _normalize_content(value):
    if isinstance(value, dict):
        role = value.get("role") or "user"
        parts = value.get("parts") or []
        texts = []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                texts.append(part["text"])
        return {"role": role, "content": "\n".join(texts) if texts else value}
    return value
