"""DSPy helpers for attaching trace context to DSPy calls.

Provides a small `TraceContextCallbackHandler` and `context_from_callbacks`
to mirror the helpers available for other frameworks (openai/crewai).
"""

from __future__ import annotations

from typing import Optional, Iterable


class TraceContextCallbackHandler:
    """Carries project, user, tag, and metadata context for DSPy calls.

    Instances are callable so they can be attached to DSPy objects or passed
    in as callbacks. The interceptor will look for an `obs_context` attribute
    to apply the context when an LM call is made.
    """

    def __init__(
        self,
        *,
        user: Optional[dict] = None,
        project_name: Optional[str] = None,
        tags: Optional[list] = None,
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
        # noop — DSPy may call callbacks; we only expose obs_context
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