"""
LangChain interceptor.

Installs a BaseCallbackHandler into LangChain LLM/chat model instances so
agent, chain, tool, and LLM runs are emitted with LangChain's run_id tree.
"""

import uuid
import time
from typing import Any, Dict, Optional

from ..core import (
    Span,
    emit,
    _current_span,
    _current_trace,
    _restore_context,
    _restore_trace,
    _set_context,
    _set_trace,
)

_INSTALLED = False
_trace_map: Dict[str, str] = {}
_active_agent_trace_id: Optional[str] = None
_active_agent_span_id: Optional[str] = None
_active_agent_context: dict = {}


def install():
    try:
        _patch_langchain()
        return True
    except ImportError:
        return False


def _patch_langchain():
    global _INSTALLED
    if _INSTALLED:
        return

    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel, BaseLLM
    from langchain_core.outputs import LLMResult
    from langchain_core.tools import BaseTool

    class _ObsHandler(BaseCallbackHandler):
        """Silently observes every LangChain event. Never alters behavior."""

        def __init__(self):
            self._t: Dict[str, float] = {}
            self._ctx: Dict[str, tuple] = {}
            self._obs_ctx: Dict[str, tuple] = {}
            self._models: Dict[str, Optional[str]] = {}

        def _start_context(
            self,
            run_id,
            parent_run_id=None,
            obs_context: Optional[dict] = None,
        ) -> tuple[str, Optional[str]]:
            parent_span = str(parent_run_id) if parent_run_id else _current_span()
            trace_id = _root(parent_run_id, run_id)
            self._t[str(run_id)] = time.perf_counter()
            self._obs_ctx[str(run_id)] = _set_context(**(obs_context or _active_agent_context))
            self._ctx[str(run_id)] = _set_trace(trace_id, str(run_id))
            return trace_id, parent_span

        def _finish_context(self, run_id) -> float:
            t0 = self._t.pop(str(run_id), time.perf_counter())
            return (time.perf_counter() - t0) * 1000

        def _restore_finished_context(self, run_id):
            _restore_trace(self._ctx.pop(str(run_id), None))
            _restore_context(self._obs_ctx.pop(str(run_id), None))

        def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kw):
            trace_id, parent_span = self._start_context(
                run_id, parent_run_id, _obs_context_from_callback_kwargs(kw)
            )
            model = _model_name(serialized, kw)
            self._models[str(run_id)] = model
            emit(Span(
                span_id=str(run_id),
                trace_id=trace_id,
                parent_span=parent_span,
                event="llm_start",
                resource="llm",
                framework="langchain",
                model=model,
                messages=prompts,
                meta=_llm_start_meta(kw, serialized),
            ))

        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kw):
            trace_id, parent_span = self._start_context(
                run_id, parent_run_id, _obs_context_from_callback_kwargs(kw)
            )
            model = _model_name(serialized, kw)
            self._models[str(run_id)] = model
            emit(Span(
                span_id=str(run_id),
                trace_id=trace_id,
                parent_span=parent_span,
                event="llm_start",
                resource="llm",
                framework="langchain",
                model=model,
                messages=[[_message_to_dict(m) for m in batch] for batch in messages],
                meta=_llm_start_meta(kw, serialized),
            ))

        def on_llm_end(self, response: LLMResult, *, run_id, parent_run_id=None, **kw):
            latency = self._finish_context(run_id)
            usage = (response.llm_output or {}).get("token_usage") or (
                response.llm_output or {}
            ).get("usage", {})
            # Fallback: extract usage from AIMessage.usage_metadata (Ollama, etc.)
            if not usage:
                for batch in (response.generations or []):
                    for gen in batch:
                        msg = getattr(gen, "message", None)
                        if msg is not None:
                            um = getattr(msg, "usage_metadata", None)
                            if um and isinstance(um, dict) and um.get("total_tokens"):
                                usage = {
                                    "prompt_tokens": um.get("input_tokens") or 0,
                                    "completion_tokens": um.get("output_tokens") or 0,
                                    "total_tokens": um.get("total_tokens") or 0,
                                }
                                break
                    if usage:
                        break
            emit(Span(
                span_id=str(run_id) + "-end",
                trace_id=_root(parent_run_id, run_id),
                parent_span=str(run_id),
                event="llm_end",
                resource="llm",
                framework="langchain",
                model=self._models.get(str(run_id)),
                latency_ms=latency,
                tokens=_token_usage(usage),
                response={"generations": [
                    [_generation_to_dict(g) for g in batch] for batch in response.generations
                ]},
                meta={"capture_via": "langchain_callback"},
            ))
            self._models.pop(str(run_id), None)
            self._restore_finished_context(run_id)

        def on_llm_error(self, error, *, run_id, parent_run_id=None, **kw):
            self._finish_context(run_id)
            emit(Span(
                span_id=str(run_id) + "-err",
                trace_id=_root(parent_run_id, run_id),
                parent_span=str(run_id),
                event="llm_error",
                resource="llm",
                framework="langchain",
                model=self._models.get(str(run_id)),
                error=str(error),
                meta={"capture_via": "langchain_callback"},
            ))
            self._models.pop(str(run_id), None)
            self._restore_finished_context(run_id)

        def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kw):
            trace_id, parent_span = self._start_context(
                run_id, parent_run_id, _obs_context_from_callback_kwargs(kw)
            )
            emit(Span(
                span_id=str(run_id),
                trace_id=trace_id,
                parent_span=parent_span,
                event="tool_start",
                resource="tool",
                framework="langchain",
                tool_name=_serialized_name(serialized),
                tool_input={"input": input_str},
                meta={"capture_via": "langchain_callback"},
            ))

        def on_tool_end(self, output, *, run_id, parent_run_id=None, **kw):
            latency = self._finish_context(run_id)
            emit(Span(
                span_id=str(run_id) + "-end",
                trace_id=_root(parent_run_id, run_id),
                parent_span=str(run_id),
                event="tool_end",
                resource="tool",
                framework="langchain",
                tool_output=str(output)[:2000],
                latency_ms=latency,
                meta={"capture_via": "langchain_callback"},
            ))
            self._restore_finished_context(run_id)

        def on_tool_error(self, error, *, run_id, parent_run_id=None, **kw):
            self._finish_context(run_id)
            emit(Span(
                span_id=str(run_id) + "-err",
                trace_id=_root(parent_run_id, run_id),
                parent_span=str(run_id),
                event="tool_error",
                resource="tool",
                framework="langchain",
                error=str(error),
                meta={"capture_via": "langchain_callback"},
            ))
            self._restore_finished_context(run_id)

        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kw):
            trace_id, parent_span = self._start_context(
                run_id, parent_run_id, _obs_context_from_callback_kwargs(kw)
            )
            emit(Span(
                span_id=str(run_id),
                trace_id=trace_id,
                parent_span=parent_span,
                event="chain_start",
                resource="chain",
                framework="langchain",
                agent_name=_serialized_name(serialized),
                meta={"inputs": _safe(inputs), "capture_via": "langchain_callback"},
            ))

        def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kw):
            latency = self._finish_context(run_id)
            emit(Span(
                span_id=str(run_id) + "-end",
                trace_id=_root(parent_run_id, run_id),
                parent_span=str(run_id),
                event="chain_end",
                resource="chain",
                framework="langchain",
                latency_ms=latency,
                meta={"outputs": _safe(outputs), "capture_via": "langchain_callback"},
            ))
            self._restore_finished_context(run_id)

        def on_chain_error(self, error, *, run_id, parent_run_id=None, **kw):
            self._finish_context(run_id)
            emit(Span(
                span_id=str(run_id) + "-err",
                trace_id=_root(parent_run_id, run_id),
                parent_span=str(run_id),
                event="chain_error",
                resource="chain",
                framework="langchain",
                error=str(error),
                meta={"capture_via": "langchain_callback"},
            ))
            self._restore_finished_context(run_id)

        def on_agent_action(self, action, *, run_id, parent_run_id=None, **kw):
            emit(Span(
                span_id=str(run_id) + "-action",
                trace_id=_root(parent_run_id, run_id),
                parent_span=str(parent_run_id) if parent_run_id else None,
                event="agent_action",
                resource="agent",
                framework="langchain",
                tool_name=getattr(action, "tool", None),
                tool_input={"input": getattr(action, "tool_input", None)},
                meta={
                    "log": str(getattr(action, "log", ""))[:500],
                    "capture_via": "langchain_callback",
                },
            ))

        def on_agent_finish(self, finish, *, run_id, parent_run_id=None, **kw):
            emit(Span(
                span_id=str(run_id) + "-finish",
                trace_id=_root(parent_run_id, run_id),
                parent_span=str(parent_run_id) if parent_run_id else None,
                event="agent_finish",
                resource="agent",
                framework="langchain",
                meta={
                    "output": _safe(getattr(finish, "return_values", None)),
                    "capture_via": "langchain_callback",
                },
            ))

    _handler = _ObsHandler()

    def _inject(orig):
        if getattr(orig, "_agent_obs_patched", False):
            return orig

        def patched(self, *args, **kwargs):
            callbacks = list(kwargs.get("callbacks") or [])
            if not any(isinstance(c, _ObsHandler) for c in callbacks):
                callbacks.append(_handler)
            kwargs["callbacks"] = callbacks
            return orig(self, *args, **kwargs)

        patched._agent_obs_patched = True
        return patched

    BaseLLM.__init__ = _inject(BaseLLM.__init__)
    BaseChatModel.__init__ = _inject(BaseChatModel.__init__)
    _patch_tools(BaseTool)
    _patch_langgraph()
    _INSTALLED = True


def _root(parent_run_id, run_id) -> str:
    rid = str(run_id)
    pid = str(parent_run_id) if parent_run_id else None
    if pid and pid in _trace_map:
        _trace_map[rid] = _trace_map[pid]
    elif rid not in _trace_map:
        _trace_map[rid] = _current_trace() or rid
    return _trace_map[rid]


def _patch_tools(BaseTool):
    if getattr(BaseTool.run, "_agent_obs_patched", False):
        return

    orig_run = BaseTool.run
    orig_arun = BaseTool.arun

    def run(self, tool_input, *args, **kwargs):
        return _run_tool_span(self, tool_input, lambda: orig_run(self, tool_input, *args, **kwargs))

    async def arun(self, tool_input, *args, **kwargs):
        return await _arun_tool_span(
            self,
            tool_input,
            lambda: orig_arun(self, tool_input, *args, **kwargs),
        )

    run._agent_obs_patched = True
    arun._agent_obs_patched = True
    BaseTool.run = run
    BaseTool.arun = arun


def _patch_langgraph():
    try:
        from langgraph.pregel.main import Pregel
    except ImportError:
        return

    if getattr(Pregel.stream, "_agent_obs_patched", False):
        return

    orig_stream = Pregel.stream
    orig_invoke = Pregel.invoke

    def stream(self, input, *args, **kwargs):
        config = _extract_config(args, kwargs)
        span_id, trace_id, previous, previous_obs, t0 = _agent_start(
            self, "agent_stream_start", input, config
        )
        ended = False
        failed = False
        try:
            for chunk in orig_stream(self, input, *args, **kwargs):
                yield chunk
            _agent_end(self, span_id, trace_id, t0, "agent_stream_end")
            ended = True
        except Exception as exc:
            failed = True
            _agent_error(span_id, trace_id, exc)
            raise
        finally:
            if not ended and not failed:
                _agent_end(self, span_id, trace_id, t0, "agent_stream_end")
            _restore_trace(previous)
            _restore_context(previous_obs)
            _clear_active_agent(trace_id, span_id)

    def invoke(self, input, *args, **kwargs):
        config = _extract_config(args, kwargs)
        span_id, trace_id, previous, previous_obs, t0 = _agent_start(
            self, "agent_invoke_start", input, config
        )
        try:
            result = orig_invoke(self, input, *args, **kwargs)
            _agent_end(self, span_id, trace_id, t0, "agent_invoke_end", result)
            return result
        except Exception as exc:
            _agent_error(span_id, trace_id, exc)
            raise
        finally:
            _restore_trace(previous)
            _restore_context(previous_obs)
            _clear_active_agent(trace_id, span_id)

    stream._agent_obs_patched = True
    invoke._agent_obs_patched = True
    Pregel.stream = stream
    Pregel.invoke = invoke


def _run_tool_span(tool, tool_input, call):
    span_id = str(uuid.uuid4())
    trace_id = _current_trace() or _active_agent_trace_id or span_id
    parent_span = _current_span() or _active_agent_span_id
    previous_obs = _set_context(**_active_agent_context)
    previous = _set_trace(trace_id, span_id)
    t0 = time.perf_counter()
    emit(Span(
        span_id=span_id,
        trace_id=trace_id,
        parent_span=parent_span,
        event="tool_start",
        resource="tool",
        framework="langchain",
        tool_name=getattr(tool, "name", type(tool).__name__),
        tool_input={"input": _safe(tool_input)},
        meta={"capture_via": "langchain_tool_patch"},
    ))
    try:
        output = call()
        emit(Span(
            span_id=span_id + "-end",
            trace_id=trace_id,
            parent_span=span_id,
            event="tool_end",
            resource="tool",
            framework="langchain",
            tool_name=getattr(tool, "name", type(tool).__name__),
            tool_output=str(output)[:2000],
            latency_ms=(time.perf_counter() - t0) * 1000,
            meta={"capture_via": "langchain_tool_patch"},
        ))
        return output
    except Exception as exc:
        emit(Span(
            span_id=span_id + "-err",
            trace_id=trace_id,
            parent_span=span_id,
            event="tool_error",
            resource="tool",
            framework="langchain",
            tool_name=getattr(tool, "name", type(tool).__name__),
            error=str(exc),
            meta={"capture_via": "langchain_tool_patch"},
        ))
        raise
    finally:
        _restore_trace(previous)
        _restore_context(previous_obs)


async def _arun_tool_span(tool, tool_input, call):
    span_id = str(uuid.uuid4())
    trace_id = _current_trace() or _active_agent_trace_id or span_id
    parent_span = _current_span() or _active_agent_span_id
    previous_obs = _set_context(**_active_agent_context)
    previous = _set_trace(trace_id, span_id)
    t0 = time.perf_counter()
    emit(Span(
        span_id=span_id,
        trace_id=trace_id,
        parent_span=parent_span,
        event="tool_start",
        resource="tool",
        framework="langchain",
        tool_name=getattr(tool, "name", type(tool).__name__),
        tool_input={"input": _safe(tool_input)},
        meta={"capture_via": "langchain_tool_patch"},
    ))
    try:
        output = await call()
        emit(Span(
            span_id=span_id + "-end",
            trace_id=trace_id,
            parent_span=span_id,
            event="tool_end",
            resource="tool",
            framework="langchain",
            tool_name=getattr(tool, "name", type(tool).__name__),
            tool_output=str(output)[:2000],
            latency_ms=(time.perf_counter() - t0) * 1000,
            meta={"capture_via": "langchain_tool_patch"},
        ))
        return output
    except Exception as exc:
        emit(Span(
            span_id=span_id + "-err",
            trace_id=trace_id,
            parent_span=span_id,
            event="tool_error",
            resource="tool",
            framework="langchain",
            tool_name=getattr(tool, "name", type(tool).__name__),
            error=str(exc),
            meta={"capture_via": "langchain_tool_patch"},
        ))
        raise
    finally:
        _restore_trace(previous)
        _restore_context(previous_obs)


def _agent_start(agent, event, input, config=None):
    global _active_agent_context, _active_agent_span_id, _active_agent_trace_id

    span_id = str(uuid.uuid4())
    trace_id = _current_trace() or span_id
    obs_context = _obs_context_from_config(config)
    previous_obs = _set_context(**obs_context)
    previous = _set_trace(trace_id, span_id)
    _active_agent_trace_id = trace_id
    _active_agent_span_id = span_id
    _active_agent_context = obs_context
    t0 = time.perf_counter()
    emit(Span(
        span_id=span_id,
        trace_id=trace_id,
        parent_span=previous[1],
        event=event,
        resource="agent",
        framework="langchain",
        agent_name=type(agent).__name__,
        meta={"input": _safe(input), "capture_via": "langgraph_pregel_patch"},
    ))
    return span_id, trace_id, previous, previous_obs, t0


def _clear_active_agent(trace_id, span_id):
    global _active_agent_context, _active_agent_span_id, _active_agent_trace_id

    if _active_agent_trace_id == trace_id and _active_agent_span_id == span_id:
        _active_agent_trace_id = None
        _active_agent_span_id = None
        _active_agent_context = {}


def _extract_config(args, kwargs) -> Optional[dict]:
    config = kwargs.get("config")
    if config is None and args and isinstance(args[0], dict):
        candidate = args[0]
        if "metadata" in candidate or "tags" in candidate or "callbacks" in candidate:
            config = candidate
    return config if isinstance(config, dict) else None


def _obs_context_from_callback_kwargs(callback_kwargs: dict) -> dict:
    metadata = callback_kwargs.get("metadata") or {}
    tags = callback_kwargs.get("tags") or []
    context = _obs_context_from_metadata(metadata, tags)
    return context or _active_agent_context


def _obs_context_from_config(config: Optional[dict]) -> dict:
    if not isinstance(config, dict):
        return {}
    metadata = config.get("metadata") or {}
    tags = config.get("tags") or []
    return _obs_context_from_metadata(metadata, tags)


def _obs_context_from_metadata(metadata: dict, tags=None) -> dict:
    if not isinstance(metadata, dict):
        metadata = {}

    project_name = (
        metadata.get("project_name")
        or metadata.get("project")
        or metadata.get("langsmith_project")
    )
    user = metadata.get("user") if isinstance(metadata.get("user"), dict) else None
    if user is None:
        user = {
            key: metadata.get(key) or metadata.get(f"user_{key}")
            for key in ("id", "name", "email", "account", "role")
            if metadata.get(key) or metadata.get(f"user_{key}")
        }
    user = user or None

    tag_values = []
    for value in list(tags or []) + list(metadata.get("tags") or []):
        if value is not None and value not in tag_values:
            tag_values.append(str(value))

    hidden = {
        "project",
        "project_name",
        "langsmith_project",
        "user",
        "tags",
        "id",
        "name",
        "email",
        "account",
        "role",
        "user_id",
        "user_name",
        "user_email",
        "user_account",
        "user_role",
    }
    trace_metadata = {key: _safe(value) for key, value in metadata.items() if key not in hidden}

    context = {
        "project_name": project_name,
        "user": user,
        "tags": tag_values or None,
        "metadata": trace_metadata or None,
    }
    return {key: value for key, value in context.items() if value is not None}


def _agent_end(agent, span_id, trace_id, t0, event, output=None):
    meta = {"capture_via": "langgraph_pregel_patch"}
    if output is not None:
        meta["output"] = _safe(output)
    emit(Span(
        span_id=span_id + "-end",
        trace_id=trace_id,
        parent_span=span_id,
        event=event,
        resource="agent",
        framework="langchain",
        agent_name=type(agent).__name__,
        latency_ms=(time.perf_counter() - t0) * 1000,
        meta=meta,
    ))


def _agent_error(span_id, trace_id, error):
    emit(Span(
        span_id=span_id + "-err",
        trace_id=trace_id,
        parent_span=span_id,
        event="agent_error",
        resource="agent",
        framework="langchain",
        error=str(error),
        meta={"capture_via": "langgraph_pregel_patch"},
    ))


def _model_name(serialized: dict, callback_kwargs: Optional[dict] = None) -> Optional[str]:
    callback_kwargs = callback_kwargs or {}
    invocation_params = callback_kwargs.get("invocation_params") or {}
    options = callback_kwargs.get("options") or {}
    for params in (invocation_params, options):
        model = (
            params.get("model")
            or params.get("model_name")
            or params.get("ls_model_name")
            or params.get("deployment_name")
        )
        if model:
            return str(model)
    if not isinstance(serialized, dict):
        return None
    kwargs = serialized.get("kwargs", {})
    return kwargs.get("model_name") or kwargs.get("model") or kwargs.get("deployment_name")


def _model_identity(serialized: dict) -> dict:
    if not isinstance(serialized, dict):
        return {}
    identity = {
        "model_class": serialized.get("name") or (serialized.get("id") or [""])[-1],
        "model_module": ".".join((serialized.get("id") or [])[:-1]),
    }
    return {key: value for key, value in identity.items() if value}


def _serialized_name(serialized: dict) -> Optional[str]:
    if not isinstance(serialized, dict):
        return None
    return serialized.get("name") or (serialized.get("id") or ["?"])[-1]


def _message_to_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return str(obj)


def _generation_to_dict(generation):
    message = getattr(generation, "message", None)
    text = getattr(generation, "text", None)
    content = getattr(message, "content", None) if message is not None else text
    result = {
        "text": str(content or text or "")[:4000],
        "type": type(generation).__name__,
    }
    if message is not None:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            result["tool_calls"] = _safe(tool_calls)
    finish_reason = (getattr(generation, "generation_info", None) or {}).get("finish_reason")
    if finish_reason:
        result["finish_reason"] = finish_reason
    return result


def _llm_start_meta(callback_kwargs: dict, serialized: Optional[dict] = None) -> dict:
    meta = {"capture_via": "langchain_callback"}
    identity = _model_identity(serialized or {})
    invocation_params = _compact_invocation_params(
        callback_kwargs.get("invocation_params") or {}
    )
    options = _compact_invocation_params(callback_kwargs.get("options") or {})

    if invocation_params:
        meta["invocation_params"] = invocation_params
    if options:
        meta["options"] = options
    if identity:
        meta.update(identity)
    if invocation_params.get("tools"):
        meta["tools"] = invocation_params["tools"]
    elif options.get("tools"):
        meta["tools"] = options["tools"]
    return meta


def _compact_invocation_params(params: dict) -> dict:
    keep = {
        "model",
        "model_name",
        "ls_model_name",
        "deployment_name",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
    }
    compact = {key: _safe(value) for key, value in params.items() if key in keep}
    return {key: value for key, value in compact.items() if value is not None}


def _token_usage(usage: dict) -> Optional[dict]:
    if not usage:
        return None
    completion_tokens_details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = completion_tokens_details.get("reasoning_tokens") or 0
    return {
        "prompt": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "completion": usage.get("completion_tokens") or usage.get("output_tokens"),
        "reasoning": reasoning_tokens,
        "total": usage.get("total_tokens"),
    }


def _safe(obj) -> Any:
    try:
        import json

        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)[:500]
