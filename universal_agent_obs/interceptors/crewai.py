"""
interceptors/crewai.py — Patches Crew.__init__ + LiteLLM callbacks.
"""
import time, uuid
from ..core import Span, emit, get_or_new_trace, _current_span, _restore_trace, _set_trace, _set_context, _restore_context


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

        # gather obs_context from any user-provided callbacks (they may be
        # passed as `step_callback`, `task_callback`, or a `callbacks` list)
        obs_ctx = {}
        def _collect_cb(obj):
            try:
                if obj and hasattr(obj, "obs_context"):
                    c = obj.obs_context()
                    if isinstance(c, dict):
                        obs_ctx.update(c)
            except Exception:
                pass

        _collect_cb(user_step)
        _collect_cb(user_task)
        if isinstance(kwargs.get('callbacks'), (list, tuple)):
            for cb in kwargs.get('callbacks'):
                _collect_cb(cb)

        def _step(agent_output):
            task = getattr(agent_output, "task", None)
            task_desc = None
            if task is not None:
                task_desc = str(getattr(task, "description", "") or "") or None
            emit(Span(
                trace_id   = _trace_id,
                event      = "agent_step", resource="agent",
                framework  = "crewai",
                agent_name = str(agent_output.agent) if hasattr(agent_output, "agent") else None,
                tool_name  = getattr(agent_output, "tool", None),
                tool_input = {"input": str(getattr(agent_output, "tool_input", ""))},
                messages   = [{"role": "user", "content": task_desc}] if task_desc else None,
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
        # save collected context onto the Crew instance for use at kickoff
        try:
            self._obs_callback_context = obs_ctx or None
        except Exception:
            pass
        _orig(self, *args, **kwargs)
        self._obs_trace_id = _trace_id

    Crew.__init__ = _patched

    if _orig_kickoff:
        def _patched_kickoff(self, *args, **kwargs):
            trace_id = getattr(self, "_obs_trace_id", None) or str(uuid.uuid4())
            self._obs_trace_id = trace_id
            previous = _set_trace(trace_id, "crewai")
            previous_ctx = None
            try:
                # apply any collected context from callbacks for the duration
                # of the kickoff run
                obs_ctx = (self._obs_callback_context or {}).copy()
                obs_ctx.setdefault("framework", "crewai")
                try:
                    previous_ctx = _set_context(**obs_ctx)
                except Exception:
                    previous_ctx = None
                return _orig_kickoff(self, *args, **kwargs)
            finally:
                if previous_ctx is not None:
                    try:
                        _restore_context(previous_ctx)
                    except Exception:
                        pass
                _restore_trace(previous)

        Crew.kickoff = _patched_kickoff

    # LiteLLM callbacks catch prompts, responses, and token usage for every LLM call
    class _LiteLLMObs:
        def log_pre_api_call(self, model, messages, kwargs):
            pass  # prompts captured in log_success_event via kwargs

        def log_success_event(self, kwargs, resp, start_time, end_time):
            usage = getattr(resp, "usage", None)

            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            if usage is not None:
                prompt_tokens     = getattr(usage, "prompt_tokens",     None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens      = getattr(usage, "total_tokens",      None)
                if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                    total_tokens = prompt_tokens + completion_tokens

            # Normalise raw messages list into [{role, content}] dicts
            raw_messages = kwargs.get("messages") or []
            messages = []
            for m in raw_messages:
                if isinstance(m, dict):
                    messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
                else:
                    messages.append({"role": "user", "content": str(m)})

            # Extract text content from the response (OpenAI-compatible shape)
            response_text = None
            try:
                choices = getattr(resp, "choices", None) or []
                if choices:
                    msg = getattr(choices[0], "message", None)
                    if msg:
                        response_text = getattr(msg, "content", None)
                if not response_text:
                    response_text = getattr(resp, "content", None)
            except Exception:
                pass

            model_name = kwargs.get("model") or getattr(resp, "model", None)

            emit(Span(
                trace_id   = get_or_new_trace(),
                parent_span= _current_span(),
                event      = "llm_end", resource="llm",
                framework  = "crewai",
                model      = model_name,
                messages   = messages or None,
                response   = {"content": response_text} if response_text else None,
                latency_ms = (end_time - start_time).total_seconds() * 1000,
                tokens     = {
                    "prompt":     prompt_tokens,
                    "completion": completion_tokens,
                    "total":      total_tokens,
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
