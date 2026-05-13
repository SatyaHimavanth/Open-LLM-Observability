"""Patch OpenAI SDK calls to accept `callbacks=` and apply obs context."""

from __future__ import annotations

from ..core import (
    _current_framework,
    _restore_context,
    _set_context,
    detect_framework_from_stack,
)
from ..openai import context_from_callbacks


def install():
    try:
        _patch_openai_sdk()
        return True
    except ImportError:
        return False


def _patch_openai_sdk():
    from openai.resources.chat.completions.completions import Completions, AsyncCompletions

    if getattr(Completions.create, "_agent_obs_patched", False):
        return

    _orig_create = Completions.create
    _orig_async_create = AsyncCompletions.create

    def _resolved_framework() -> str:
        # Prefer explicit context first, then stack-based detection for frameworks
        # that may execute in worker threads where contextvars are not propagated.
        return _current_framework() or detect_framework_from_stack() or "openai-sdk"

    def _patched_create(self, *args, callbacks=None, **kwargs):
        ctx = context_from_callbacks(callbacks).copy() if callbacks else {}
        # Preserve already active framework (for example CrewAI), and fall back to
        # stack detection before defaulting to openai-sdk.
        ctx.setdefault("framework", _resolved_framework())
        previous_context = _set_context(**ctx)
        try:
            return _orig_create(self, *args, **kwargs)
        finally:
            if previous_context is not None:
                _restore_context(previous_context)

    async def _patched_async_create(self, *args, callbacks=None, **kwargs):
        ctx = context_from_callbacks(callbacks).copy() if callbacks else {}
        # Preserve already active framework (for example CrewAI), and fall back to
        # stack detection before defaulting to openai-sdk.
        ctx.setdefault("framework", _resolved_framework())
        previous_context = _set_context(**ctx)
        try:
            return await _orig_async_create(self, *args, **kwargs)
        finally:
            if previous_context is not None:
                _restore_context(previous_context)

    _patched_create._agent_obs_patched = True
    _patched_async_create._agent_obs_patched = True
    Completions.create = _patched_create
    AsyncCompletions.create = _patched_async_create