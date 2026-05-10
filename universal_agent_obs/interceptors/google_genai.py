"""Google GenAI SDK interceptor."""

import time
import uuid
from typing import Optional

from ..core import (
    Span,
    detect_provider,
    emit,
    get_or_new_trace,
    _current_span,
    _restore_trace,
    _set_trace,
)

_INSTALLED = False


def install():
    try:
        _patch_google_genai()
        return True
    except ImportError:
        return False


def _patch_google_genai():
    global _INSTALLED
    if _INSTALLED:
        return

    from google.genai.models import AsyncModels, Models

    _patch_sync_models(Models)
    _patch_async_models(AsyncModels)
    _INSTALLED = True


def _patch_sync_models(Models):
    if getattr(Models.generate_content, "_agent_obs_patched", False):
        return

    orig_generate_content = Models.generate_content
    orig_generate_content_stream = Models.generate_content_stream

    def generate_content(self, *args, **kwargs):
        if _current_span():
            return orig_generate_content(self, *args, **kwargs)

        model = kwargs.get("model")
        contents = kwargs.get("contents")
        start_span_id, trace_id, t0 = _emit_start(model, contents, "google_genai_sdk")
        previous = _set_trace(trace_id, start_span_id)
        try:
            response = orig_generate_content(self, *args, **kwargs)
            _emit_end(start_span_id, trace_id, model, response, (time.perf_counter() - t0) * 1000)
            return response
        except Exception as exc:
            _emit_error(start_span_id, trace_id, model, exc, (time.perf_counter() - t0) * 1000)
            raise
        finally:
            _restore_trace(previous)

    def generate_content_stream(self, *args, **kwargs):
        if _current_span():
            yield from orig_generate_content_stream(self, *args, **kwargs)
            return

        model = kwargs.get("model")
        contents = kwargs.get("contents")
        start_span_id, trace_id, t0 = _emit_start(model, contents, "google_genai_sdk_stream")
        previous = _set_trace(trace_id, start_span_id)
        chunks = []
        try:
            for chunk in orig_generate_content_stream(self, *args, **kwargs):
                chunks.append(chunk)
                yield chunk
            _emit_end(start_span_id, trace_id, model, chunks, (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            _emit_error(start_span_id, trace_id, model, exc, (time.perf_counter() - t0) * 1000)
            raise
        finally:
            _restore_trace(previous)

    generate_content._agent_obs_patched = True
    generate_content_stream._agent_obs_patched = True
    Models.generate_content = generate_content
    Models.generate_content_stream = generate_content_stream


def _patch_async_models(AsyncModels):
    if getattr(AsyncModels.generate_content, "_agent_obs_patched", False):
        return

    orig_generate_content = AsyncModels.generate_content
    orig_generate_content_stream = AsyncModels.generate_content_stream

    async def generate_content(self, *args, **kwargs):
        if _current_span():
            return await orig_generate_content(self, *args, **kwargs)

        model = kwargs.get("model")
        contents = kwargs.get("contents")
        start_span_id, trace_id, t0 = _emit_start(model, contents, "google_genai_sdk")
        previous = _set_trace(trace_id, start_span_id)
        try:
            response = await orig_generate_content(self, *args, **kwargs)
            _emit_end(start_span_id, trace_id, model, response, (time.perf_counter() - t0) * 1000)
            return response
        except Exception as exc:
            _emit_error(start_span_id, trace_id, model, exc, (time.perf_counter() - t0) * 1000)
            raise
        finally:
            _restore_trace(previous)

    async def generate_content_stream(self, *args, **kwargs):
        if _current_span():
            async for chunk in orig_generate_content_stream(self, *args, **kwargs):
                yield chunk
            return

        model = kwargs.get("model")
        contents = kwargs.get("contents")
        start_span_id, trace_id, t0 = _emit_start(model, contents, "google_genai_sdk_stream")
        previous = _set_trace(trace_id, start_span_id)
        chunks = []
        try:
            async for chunk in orig_generate_content_stream(self, *args, **kwargs):
                chunks.append(chunk)
                yield chunk
            _emit_end(start_span_id, trace_id, model, chunks, (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            _emit_error(start_span_id, trace_id, model, exc, (time.perf_counter() - t0) * 1000)
            raise
        finally:
            _restore_trace(previous)

    generate_content._agent_obs_patched = True
    generate_content_stream._agent_obs_patched = True
    AsyncModels.generate_content = generate_content
    AsyncModels.generate_content_stream = generate_content_stream


def _emit_start(model: Optional[str], contents, capture_via: str):
    span_id = str(uuid.uuid4())
    trace_id = get_or_new_trace()
    emit(Span(
        span_id=span_id,
        trace_id=trace_id,
        parent_span=_current_span(),
        event="llm_start",
        resource="llm",
        framework="google-genai",
        model=model,
        provider=detect_provider(model),
        messages=_safe(contents),
        meta={"capture_via": capture_via},
    ))
    return span_id, trace_id, time.perf_counter()


def _emit_end(parent_span: str, trace_id: str, model: Optional[str], response, latency_ms: float):
    emit(Span(
        span_id=f"{parent_span}-end",
        trace_id=trace_id,
        parent_span=parent_span,
        event="llm_end",
        resource="llm",
        framework="google-genai",
        model=model or _response_model(response),
        provider=detect_provider(model or _response_model(response)),
        response={"content": _response_text(response)} if _response_text(response) else None,
        tokens=_response_tokens(response),
        latency_ms=latency_ms,
        meta={"capture_via": "google_genai_sdk"},
    ))


def _emit_error(parent_span: str, trace_id: str, model: Optional[str], error: Exception, latency_ms: float):
    emit(Span(
        span_id=f"{parent_span}-err",
        trace_id=trace_id,
        parent_span=parent_span,
        event="llm_error",
        resource="llm",
        framework="google-genai",
        model=model,
        provider=detect_provider(model),
        error=str(error),
        latency_ms=latency_ms,
        meta={"capture_via": "google_genai_sdk"},
    ))


def _response_model(response) -> Optional[str]:
    item = _last_response(response)
    return getattr(item, "model_version", None) or getattr(item, "modelVersion", None)


def _response_text(response) -> Optional[str]:
    texts = []
    for item in _responses(response):
        text = getattr(item, "text", None)
        if text:
            texts.append(str(text))
    return "\n".join(texts)[:4000] if texts else None


def _response_tokens(response) -> Optional[dict]:
    usage = None
    for item in _responses(response):
        usage = getattr(item, "usage_metadata", None) or usage
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


def _responses(response):
    if isinstance(response, list):
        return response
    return [response]


def _last_response(response):
    items = _responses(response)
    return items[-1] if items else None


def _safe(value):
    try:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return value
    except Exception:
        return str(value)[:500]
