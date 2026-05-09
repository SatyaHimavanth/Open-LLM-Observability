"""LangChain helpers for attaching trace attributes to runs."""

from __future__ import annotations

from typing import Optional

from .core import PROJECT_NAME, _restore_context, _set_context

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover - only raised without optional extra
    raise ImportError(
        "Install the LangChain optional extra to use TraceContextCallbackHandler: "
        "uv sync --extra langchain"
    ) from exc


class TraceContextCallbackHandler(BaseCallbackHandler):
    """Attach project, user, tags, and metadata to LangChain child spans."""

    def __init__(
        self,
        *,
        user: Optional[dict] = None,
        project_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ):
        self.project_name = project_name or PROJECT_NAME
        self.user = user
        self.tags = tags
        self.metadata = metadata
        self._previous: dict[str, tuple] = {}

    def _start(self, run_id):
        self._previous[str(run_id)] = _set_context(
            project_name=self.project_name,
            user=self.user,
            tags=self.tags,
            metadata=self.metadata,
        )

    def _end(self, run_id):
        _restore_context(self._previous.pop(str(run_id), None))

    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        self._start(run_id)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._end(run_id)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start(run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start(run_id)

    def on_llm_end(self, response, *, run_id, **kwargs):
        self._end(run_id)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end(run_id)

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._start(run_id)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._end(run_id)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._end(run_id)


def trace_context_callback(**kwargs) -> TraceContextCallbackHandler:
    return TraceContextCallbackHandler(**kwargs)
