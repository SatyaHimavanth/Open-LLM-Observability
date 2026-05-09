"""
interceptors/crewai.py — Patches Crew.__init__ + LiteLLM callbacks.
"""
import time, uuid
from ..core import Span, emit, get_or_new_trace, _current_span, _restore_trace, _set_trace


def install():
    try:
        _patch_crewai()
        return True
    except ImportError:
        return False


def _patch_crewai():
    from crewai import Crew
    import litellm

    _orig = Crew.__init__
    _orig_kickoff = getattr(Crew, "kickoff", None)

    def _patched(self, *args, **kwargs):
        _trace_id = str(uuid.uuid4())

        user_step = kwargs.get("step_callback")
        user_task = kwargs.get("task_callback")

        def _step(agent_output):
            emit(Span(
                trace_id   = _trace_id,
                event      = "agent_step", resource="agent",
                framework  = "crewai",
                agent_name = str(agent_output.agent) if hasattr(agent_output, "agent") else None,
                tool_name  = getattr(agent_output, "tool", None),
                tool_input = {"input": str(getattr(agent_output, "tool_input", ""))},
                meta       = {
                    "thought": str(getattr(agent_output, "thought", ""))[:500],
                    "result":  str(getattr(agent_output, "result", ""))[:500],
                    "capture_via": "crewai_step_callback",
                },
            ))
            if user_step:
                user_step(agent_output)

        def _task(task_output):
            emit(Span(
                trace_id   = _trace_id,
                event      = "task_complete", resource="agent",
                framework  = "crewai",
                agent_name = str(getattr(task_output, "agent", "")) or None,
                meta       = {
                    "task_id":     str(getattr(task_output, "name", "")),
                    "output":      str(getattr(task_output, "raw", ""))[:1000],
                    "capture_via": "crewai_task_callback",
                },
            ))
            if user_task:
                user_task(task_output)

        kwargs["step_callback"] = _step
        kwargs["task_callback"] = _task
        _orig(self, *args, **kwargs)
        self._obs_trace_id = _trace_id

    Crew.__init__ = _patched

    if _orig_kickoff:
        def _patched_kickoff(self, *args, **kwargs):
            trace_id = getattr(self, "_obs_trace_id", None) or str(uuid.uuid4())
            self._obs_trace_id = trace_id
            previous = _set_trace(trace_id, "crewai")
            try:
                return _orig_kickoff(self, *args, **kwargs)
            finally:
                _restore_trace(previous)

        Crew.kickoff = _patched_kickoff

    # LiteLLM callbacks catch token usage for every LLM call
    class _LiteLLMObs:
        def log_pre_api_call(self, model, messages, kwargs):
            pass  # captured on end

        def log_success_event(self, kwargs, resp, start_time, end_time):
            usage = getattr(resp, "usage", {})
            if hasattr(usage, "__dict__"):
                usage = usage.__dict__
            emit(Span(
                trace_id   = get_or_new_trace(),
                parent_span= _current_span(),
                event      = "llm_end", resource="llm",
                framework  = "crewai",
                model      = kwargs.get("model"),
                latency_ms = (end_time - start_time).total_seconds() * 1000,
                tokens     = {
                    "prompt":     getattr(resp.usage, "prompt_tokens", None)
                                  if hasattr(resp, "usage") else None,
                    "completion": getattr(resp.usage, "completion_tokens", None)
                                  if hasattr(resp, "usage") else None,
                },
                meta       = {"capture_via": "crewai_litellm"},
            ))

        def log_failure_event(self, kwargs, resp, start_time, end_time):
            emit(Span(
                trace_id = get_or_new_trace(),
                parent_span = _current_span(),
                event  = "llm_error", resource="llm",
                framework = "crewai",
                model  = kwargs.get("model"),
                error  = str(resp),
                meta   = {"capture_via": "crewai_litellm"},
            ))

    if not any(isinstance(cb, _LiteLLMObs) for cb in (litellm.callbacks or [])):
        litellm.callbacks = list(litellm.callbacks or []) + [_LiteLLMObs()]
