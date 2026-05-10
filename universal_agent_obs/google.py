"""Google ADK / Google GenAI helpers for attaching trace context."""

from __future__ import annotations

from typing import Optional


class TraceContextCallbackHandler:
    """Carries project, user, tag, and metadata context for Google traces."""

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


def trace_context_callback(**kwargs) -> TraceContextCallbackHandler:
    return TraceContextCallbackHandler(**kwargs)


def context_from_callbacks(callbacks) -> dict:
    context: dict = {}
    for callback in callbacks or []:
        if hasattr(callback, "obs_context"):
            context.update(callback.obs_context())
    return context
