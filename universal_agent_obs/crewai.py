"""CrewAI helpers for attaching trace attributes to runs.

This mirrors the small helper API used for `google` and `langchain`:
- `TraceContextCallbackHandler` provides an `obs_context()` dict
- `trace_context_callback(...)` returns the handler
- Callables passed into Crew as callbacks may implement `obs_context`
"""

from __future__ import annotations

from typing import Optional, Iterable


class TraceContextCallbackHandler:
    """Carries project, user, tag, and metadata context for Crew runs.

    Instances are callable so they can be passed as `step_callback` /
    `task_callback` to Crew. The interceptor will look for an `obs_context`
    attribute to apply the context at kickoff time.
    """

    def __init__(
        self,
        *,
        user: Optional[dict] = None,
        project_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ):
        self.user = user
        self.project_name = project_name
        self.tags = tags
        self.metadata = metadata

    def obs_context(self) -> dict:
        return {
            key: value
            for key, value in {
                "project_name": self.project_name,
                "user": self.user,
                "tags": self.tags,
                "metadata": self.metadata,
            }.items()
            if value is not None
        }

    def __call__(self, *args, **kwargs):
        # noop — Crew will invoke this as a step/task callback if provided.
        return None


def trace_context_callback(**kwargs) -> TraceContextCallbackHandler:
    return TraceContextCallbackHandler(**kwargs)


def context_from_callbacks(callbacks: Optional[Iterable]) -> dict:
    ctx: dict = {}
    if not callbacks:
        return ctx
    for cb in callbacks:
        try:
            if hasattr(cb, "obs_context"):
                c = cb.obs_context()
                if isinstance(c, dict):
                    ctx.update(c)
        except Exception:
            pass
    return ctx
