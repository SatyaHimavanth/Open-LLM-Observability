"""
interceptors/dspy.py — Patches dspy.LM calls.
"""
import time
import uuid
from ..core import Span, emit, get_or_new_trace, _current_span, detect_provider

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

    _orig_call = LM.__call__
    _orig_acall = LM.acall

    def _patched_call(self, prompt, **kwargs):
        trace_id = get_or_new_trace()
        parent_span = _current_span()
        start_span_id = str(uuid.uuid4())

        # Determine model name
        model_name = getattr(self, "model", None) or getattr(self, "kwargs", {}).get("model")

        emit(Span(
            span_id=start_span_id,
            trace_id=trace_id,
            parent_span=parent_span,
            event="llm_start",
            resource="llm",
            framework="dspy",
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            meta={"capture_via": "dspy_interceptor"},
        ))

        t0 = time.perf_counter()
        try:
            response = _orig_call(self, prompt, **kwargs)
            latency = (time.perf_counter() - t0) * 1000

            # response is typically a list of strings or similar in DSPy
            resp_content = response if isinstance(response, (str, list)) else str(response)

            emit(Span(
                trace_id=trace_id,
                parent_span=start_span_id,
                event="llm_end",
                resource="llm",
                framework="dspy",
                model=model_name,
                response={"content": resp_content},
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

    async def _patched_acall(self, prompt, **kwargs):
        trace_id = get_or_new_trace()
        parent_span = _current_span()
        start_span_id = str(uuid.uuid4())

        model_name = getattr(self, "model", None) or getattr(self, "kwargs", {}).get("model")

        emit(Span(
            span_id=start_span_id,
            trace_id=trace_id,
            parent_span=parent_span,
            event="llm_start",
            resource="llm",
            framework="dspy",
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            meta={"capture_via": "dspy_interceptor"},
        ))

        t0 = time.perf_counter()
        try:
            response = await _orig_acall(self, prompt, **kwargs)
            latency = (time.perf_counter() - t0) * 1000

            resp_content = response if isinstance(response, (str, list)) else str(response)

            emit(Span(
                trace_id=trace_id,
                parent_span=start_span_id,
                event="llm_end",
                resource="llm",
                framework="dspy",
                model=model_name,
                response={"content": resp_content},
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

    LM.__call__ = _patched_call
    LM.acall = _patched_acall
