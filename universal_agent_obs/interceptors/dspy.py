"""
interceptors/dspy.py — Patches dspy.LM calls.
"""
import time
import uuid
from ..core import Span, emit, get_or_new_trace, _current_span, detect_provider, _set_context, _restore_context

def install():
    try:
        import dspy
        _patch_dspy()
        return True
    except ImportError:
        return False

def _patch_dspy():
    import dspy
    from dspy.clients.lm import LM
    # Patch a variety of common LM method names to ensure we catch calls
    sync_names = ["__call__", "predict", "generate", "run", "invoke", "complete"]
    async_names = ["acall", "apredict", "agenerate", "arun", "ainvoke", "acomplete"]

    def _make_sync_wrapper(orig):
        def wrapper(self, *args, **kwargs):
            prompt = kwargs.get("prompt") or (args[0] if args else None)
            messages = kwargs.get("messages") or (args[1] if len(args) > 1 else None)

            # attempt to extract trace context from any callbacks passed by the user
            callbacks = None
            for key in ("obs_callbacks", "callbacks", "trace_callbacks", "trace_callback"):
                if key in kwargs:
                    callbacks = kwargs.get(key)
                    break
            if callbacks is None:
                callbacks = getattr(self, "_obs_callbacks", None) or getattr(self, "callbacks", None)

            try:
                from ..dspy import context_from_callbacks
                cb_ctx = context_from_callbacks(callbacks)
            except Exception:
                cb_ctx = {}

            trace_id = get_or_new_trace()
            parent_span = _current_span()
            start_span_id = str(uuid.uuid4())
            previous_ctx = _set_context(framework="dspy", **cb_ctx)

            model_name = getattr(self, "model", None) or getattr(self, "kwargs", {}).get("model")

            emit(Span(
                span_id=start_span_id,
                trace_id=trace_id,
                parent_span=parent_span,
                event="llm_start",
                resource="llm",
                framework="dspy",
                model=model_name,
                messages=messages or [{"role": "user", "content": prompt}],
                meta={"capture_via": "dspy_interceptor"},
            ))

            history_len = len(getattr(self, 'history', []))
            t0 = time.perf_counter()
            try:
                response = orig(self, *args, **kwargs)
                latency = (time.perf_counter() - t0) * 1000

                resp_content = response if isinstance(response, (str, list)) else str(response)

                # Extract token usage from DSPy's history entry
                tokens = None
                try:
                    history = getattr(self, 'history', [])
                    if len(history) > history_len:
                        entry = history[-1]
                        usage = entry.get('usage', {})
                        if usage:
                            tokens = {
                                "prompt": usage.get('prompt_tokens', 0),
                                "completion": usage.get('completion_tokens', 0),
                                "reasoning": usage.get('reasoning_tokens', 0) or usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0) if isinstance(usage.get('completion_tokens_details'), dict) else 0,
                                "total": usage.get('total_tokens', 0),
                            }
                except Exception:
                    pass

                emit(Span(
                    trace_id=trace_id,
                    parent_span=start_span_id,
                    event="llm_end",
                    resource="llm",
                    framework="dspy",
                    model=model_name,
                    response={"content": resp_content},
                    tokens=tokens,
                    latency_ms=latency,
                    meta={"capture_via": "dspy_interceptor"},
                ))
                return response
            except Exception as e:
                latency = (time.perf_counter() - t0) * 1000
                emit(Span(
                    trace_id=trace_id,
                    parent_span=start_span_id,
                    event="llm_error",
                    resource="llm",
                    framework="dspy",
                    model=model_name,
                    error=str(e),
                    latency_ms=latency,
                    meta={"capture_via": "dspy_interceptor"},
                ))
                raise
            finally:
                _restore_context(previous_ctx)

        return wrapper

    def _make_async_wrapper(orig):
        async def wrapper(self, *args, **kwargs):
            prompt = kwargs.get("prompt") or (args[0] if args else None)
            messages = kwargs.get("messages") or (args[1] if len(args) > 1 else None)

            # attempt to extract trace context from any callbacks passed by the user
            callbacks = None
            for key in ("obs_callbacks", "callbacks", "trace_callbacks", "trace_callback"):
                if key in kwargs:
                    callbacks = kwargs.get(key)
                    break
            if callbacks is None:
                callbacks = getattr(self, "_obs_callbacks", None) or getattr(self, "callbacks", None)

            try:
                from ..dspy import context_from_callbacks
                cb_ctx = context_from_callbacks(callbacks)
            except Exception:
                cb_ctx = {}

            trace_id = get_or_new_trace()
            parent_span = _current_span()
            start_span_id = str(uuid.uuid4())
            previous_ctx = _set_context(framework="dspy", **cb_ctx)

            model_name = getattr(self, "model", None) or getattr(self, "kwargs", {}).get("model")

            emit(Span(
                span_id=start_span_id,
                trace_id=trace_id,
                parent_span=parent_span,
                event="llm_start",
                resource="llm",
                framework="dspy",
                model=model_name,
                messages=messages or [{"role": "user", "content": prompt}],
                meta={"capture_via": "dspy_interceptor"},
            ))

            history_len = len(getattr(self, 'history', []))
            t0 = time.perf_counter()
            try:
                response = await orig(self, *args, **kwargs)
                latency = (time.perf_counter() - t0) * 1000

                resp_content = response if isinstance(response, (str, list)) else str(response)

                # Extract token usage from DSPy's history entry
                tokens = None
                try:
                    history = getattr(self, 'history', [])
                    if len(history) > history_len:
                        entry = history[-1]
                        usage = entry.get('usage', {})
                        if usage:
                            tokens = {
                                "prompt": usage.get('prompt_tokens', 0),
                                "completion": usage.get('completion_tokens', 0),
                                "reasoning": usage.get('reasoning_tokens', 0) or usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0) if isinstance(usage.get('completion_tokens_details'), dict) else 0,
                                "total": usage.get('total_tokens', 0),
                            }
                except Exception:
                    pass

                emit(Span(
                    trace_id=trace_id,
                    parent_span=start_span_id,
                    event="llm_end",
                    resource="llm",
                    framework="dspy",
                    model=model_name,
                    response={"content": resp_content},
                    tokens=tokens,
                    latency_ms=latency,
                    meta={"capture_via": "dspy_interceptor"},
                ))
                return response
            except Exception as e:
                latency = (time.perf_counter() - t0) * 1000
                emit(Span(
                    trace_id=trace_id,
                    parent_span=start_span_id,
                    event="llm_error",
                    resource="llm",
                    framework="dspy",
                    model=model_name,
                    error=str(e),
                    latency_ms=latency,
                    meta={"capture_via": "dspy_interceptor"},
                ))
                raise
            finally:
                _restore_context(previous_ctx)

        return wrapper

    # Attach wrappers for all detected method names
    for name in sync_names:
        if hasattr(LM, name):
            orig = getattr(LM, name)
            try:
                setattr(LM, name, _make_sync_wrapper(orig))
            except Exception:
                pass

    for name in async_names:
        if hasattr(LM, name):
            orig = getattr(LM, name)
            try:
                setattr(LM, name, _make_async_wrapper(orig))
            except Exception:
                pass
