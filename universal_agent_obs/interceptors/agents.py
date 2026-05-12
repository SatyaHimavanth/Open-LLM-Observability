"""
interceptors/autogen.py — Patches ConversableAgent for AutoGen 0.2/0.4.
interceptors/openai_agents.py — Patches Agent with AgentHooks.
Both in one file for convenience.
"""
import time, uuid
from ..core import Span, emit, _restore_context, _restore_trace, _set_context, _set_trace, get_or_new_trace
from ..openai import context_from_callbacks


# ── AutoGen ──────────────────────────────────────────────────────────────────

def install_autogen():
    try:
        _patch_autogen()
        return True
    except ImportError:
        return False


def _patch_autogen():
    from autogen import ConversableAgent

    _orig_init     = ConversableAgent.__init__
    _orig_initiate = ConversableAgent.initiate_chat

    def _patched_init(self, name, *args, **kwargs):
        _orig_init(self, name, *args, **kwargs)
        self._obs_trace_id = None

        def _obs_reply(recipient, messages, sender, config):
            if messages:
                last = messages[-1]
                emit(Span(
                    trace_id   = recipient._obs_trace_id or str(uuid.uuid4()),
                    event      = "agent_message", resource="agent",
                    framework  = "autogen",
                    agent_name = recipient.name,
                    meta       = {
                        "sender":    sender.name if sender else "human",
                        "role":      last.get("role"),
                        "content":   str(last.get("content", ""))[:1000],
                        "msg_count": len(messages),
                        "capture_via": "autogen_register_reply",
                    },
                ))
            return False, None   # pass through to real reply

        self.register_reply(
            trigger   = ConversableAgent,
            reply_func= _obs_reply,
            position  = 0,
            name      = "_universal_obs",
        )

    def _patched_initiate(self, recipient, message, **kwargs):
        tid = str(uuid.uuid4())
        self._obs_trace_id     = tid
        recipient._obs_trace_id = tid
        t0 = time.perf_counter()
        emit(Span(
            trace_id   = tid,
            event      = "conversation_start", resource="agent",
            framework  = "autogen",
            agent_name = self.name,
            meta       = {
                "recipient": recipient.name,
                "message":   str(message)[:500],
                "capture_via": "autogen_initiate_chat",
            },
        ))
        result = _orig_initiate(self, recipient, message, **kwargs)
        emit(Span(
            trace_id   = tid,
            event      = "conversation_end", resource="agent",
            framework  = "autogen",
            agent_name = self.name,
            latency_ms = (time.perf_counter() - t0) * 1000,
            meta       = {"summary": str(getattr(result, "summary", ""))[:500],
                          "capture_via": "autogen_initiate_chat"},
        ))
        return result

    ConversableAgent.__init__    = _patched_init
    ConversableAgent.initiate_chat = _patched_initiate


# ── OpenAI Agents SDK ────────────────────────────────────────────────────────

def install_openai_agents():
    try:
        _patch_openai_agents()
        return True
    except ImportError:
        return False


def _patch_openai_agents():
    from agents import Agent, AgentHooks, RunContextWrapper, Runner

    def _shared_store(ctx: RunContextWrapper):
        store = getattr(ctx, "context", None)
        if store is None:
            state = getattr(ctx, "_obs_state", None)
            if state is None:
                state = {}
                setattr(ctx, "_obs_state", state)
            return state
        return store

    def _store_get(store, key, default=None):
        if isinstance(store, dict):
            return store.get(key, default)
        return getattr(store, key, default)

    def _store_set(store, key, value):
        if isinstance(store, dict):
            store[key] = value
        else:
            setattr(store, key, value)

    def _store_dict(store, key):
        value = _store_get(store, key)
        if value is None:
            value = {}
            _store_set(store, key, value)
        return value

    def _usage_dict(ctx: RunContextWrapper) -> dict:
        usage = getattr(ctx, "usage", None)
        if not usage:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        return {
            "requests": getattr(usage, "requests", 0),
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
            "input_tokens_details": {
                "cached_tokens": getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0),
            },
            "output_tokens_details": {
                "reasoning_tokens": getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0),
            },
        }

    def _model_name(agent: Agent):
        model = getattr(agent, "model", None)
        return getattr(model, "model", None) or str(model) if model else None

    def _model_meta(agent: Agent) -> dict:
        model = getattr(agent, "model", None)
        if model is None:
            return {}
        return {
            "model_class": getattr(model.__class__, "__name__", ""),
            "model_module": getattr(model.__class__, "__module__", ""),
        }

    def _tool_snapshot(agent: Agent) -> list[dict]:
        tools = []
        for tool in getattr(agent, "tools", []) or []:
            tools.append(
                {
                    "name": getattr(tool, "name", None) or getattr(tool, "tool_name", None) or getattr(tool, "__name__", "tool"),
                    "description": getattr(tool, "description", None) or getattr(tool, "tool_description", None) or getattr(tool, "__doc__", None),
                }
            )
        return tools

    def _normalize_input_item(item):
        if item is None:
            return None
        if isinstance(item, dict):
            role = item.get("role") or item.get("type") or "message"
            content = item.get("content")
            if content is None:
                content = item
            return {"role": role, "content": content}
        raw_item = getattr(item, "raw_item", None)
        if isinstance(raw_item, dict):
            role = raw_item.get("role") or raw_item.get("type") or getattr(item, "type", None) or "message"
            content = raw_item.get("content")
            if content is None:
                content = raw_item
            return {"role": role, "content": content}
        role = getattr(item, "role", None) or getattr(item, "type", None) or "message"
        content = getattr(item, "content", None)
        if content is None:
            content = str(item)
        return {"role": role, "content": content}

    def _prompt_messages(system_prompt, input_items) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for item in input_items or []:
            normalized = _normalize_input_item(item)
            if normalized:
                messages.append(normalized)
        return messages

    class _ObsHooks(AgentHooks):
        async def on_start(self, ctx: RunContextWrapper, agent: Agent):
            store = _shared_store(ctx)
            trace_id = _store_get(store, "_obs_trace_id") or get_or_new_trace()
            _store_set(store, "_obs_trace_id", trace_id)
            timers = _store_dict(store, "_obs_timers")
            timers[f"agent:{agent.name}:t0"] = time.perf_counter()
            agent_span_ids = _store_dict(store, "_obs_agent_span_ids")
            agent_span_id = str(uuid.uuid4())
            agent_span_ids[agent.name] = agent_span_id
            emit(Span(
                span_id    = agent_span_id,
                trace_id   = trace_id,
                event      = "agent_start", resource="agent",
                framework  = "openai-agents",
                agent_name = agent.name,
                model      = _model_name(agent),
                messages   = [{"role": "system", "content": getattr(agent, "instructions", "")}],
                meta       = {
                    "capture_via": "openai_agents_hooks",
                    "agents": [{
                        "name": agent.name,
                        "model": _model_name(agent),
                        "instruction": getattr(agent, "instructions", None),
                        "tools": _tool_snapshot(agent),
                    }],
                    **_model_meta(agent),
                },
            ))

        async def on_llm_start(self, ctx: RunContextWrapper, agent: Agent, system_prompt, input_items):
            store = _shared_store(ctx)
            agent_context = _store_dict(store, "_obs_agent_context")
            if agent.name not in agent_context:
                agent_context[agent.name] = {
                    "input": {"messages": _prompt_messages(system_prompt, input_items)},
                    "agents": [{
                        "name": agent.name,
                        "model": _model_name(agent),
                        "instruction": getattr(agent, "instructions", None),
                        "tools": _tool_snapshot(agent),
                    }],
                    "tool_calls": [],
                }

        async def on_end(self, ctx: RunContextWrapper, agent: Agent, output):
            store = _shared_store(ctx)
            trace_id = _store_get(store, "_obs_trace_id")
            timers = _store_dict(store, "_obs_timers")
            agent_span_ids = _store_dict(store, "_obs_agent_span_ids")
            agent_context = _store_dict(store, "_obs_agent_context").get(agent.name, {})
            emit(Span(
                trace_id   = trace_id,
                parent_span= agent_span_ids.get(agent.name),
                event      = "agent_end", resource="agent",
                framework  = "openai-agents",
                agent_name = agent.name,
                latency_ms = (time.perf_counter() -
                              timers.get(f"agent:{agent.name}:t0", time.perf_counter())) * 1000,
                meta       = {
                    "output": str(output)[:500],
                    "usage":  _usage_dict(ctx),
                    "input": agent_context.get("input", {}),
                    "agents": agent_context.get("agents", []),
                    "tool_calls": agent_context.get("tool_calls", []),
                    "capture_via": "openai_agents_hooks",
                },
            ))

        async def on_tool_start(self, ctx: RunContextWrapper, agent: Agent, tool):
            store = _shared_store(ctx)
            trace_id = _store_get(store, "_obs_trace_id") or get_or_new_trace()
            _store_set(store, "_obs_trace_id", trace_id)
            timers = _store_dict(store, "_obs_timers")
            tool_span_ids = _store_dict(store, "_obs_tool_span_ids")
            tool_call_id = getattr(ctx, "tool_call_id", None) or tool.name
            tool_span_id = str(uuid.uuid4())
            timers[f"tool:{tool_call_id}:t0"] = time.perf_counter()
            tool_span_ids[tool_call_id] = tool_span_id
            agent_context = _store_dict(store, "_obs_agent_context")
            current = agent_context.setdefault(agent.name, {"tool_calls": []})
            current.setdefault("tool_calls", []).append(
                {
                    "id": tool_call_id,
                    "name": tool.name,
                    "input": getattr(ctx, "tool_arguments", None),
                }
            )
            emit(Span(
                span_id    = tool_span_id,
                trace_id   = trace_id,
                parent_span= _store_dict(store, "_obs_agent_span_ids").get(agent.name),
                event      = "tool_start", resource="tool",
                framework  = "openai-agents",
                agent_name = agent.name,
                tool_name  = tool.name,
                tool_input = getattr(ctx, "tool_arguments", None) and {"input": str(getattr(ctx, "tool_arguments", ""))} or None,
                meta       = {
                    "capture_via": "openai_agents_hooks",
                    "tool_call_id": tool_call_id,
                },
            ))

        async def on_tool_end(self, ctx: RunContextWrapper, agent: Agent, tool, result: str):
            store = _shared_store(ctx)
            trace_id = _store_get(store, "_obs_trace_id")
            timers = _store_dict(store, "_obs_timers")
            tool_span_ids = _store_dict(store, "_obs_tool_span_ids")
            tool_call_id = getattr(ctx, "tool_call_id", None) or tool.name
            agent_context = _store_dict(store, "_obs_agent_context")
            current = agent_context.setdefault(agent.name, {"tool_calls": []})
            for call in current.get("tool_calls", []):
                if call.get("id") == tool_call_id:
                    call["output"] = str(result)
                    break
            emit(Span(
                trace_id   = trace_id,
                parent_span= tool_span_ids.get(tool_call_id),
                event      = "tool_end", resource="tool",
                framework  = "openai-agents",
                agent_name = agent.name,
                tool_name  = tool.name,
                tool_output= str(result)[:1000],
                latency_ms = (time.perf_counter() -
                              timers.get(f"tool:{tool_call_id}:t0", time.perf_counter())) * 1000,
                meta       = {
                    "capture_via": "openai_agents_hooks",
                    "tool_call_id": tool_call_id,
                },
            ))

        async def on_handoff(self, ctx: RunContextWrapper, agent: Agent, source: Agent):
            store = _shared_store(ctx)
            emit(Span(
                trace_id   = _store_get(store, "_obs_trace_id"),
                event      = "handoff", resource="agent",
                framework  = "openai-agents",
                agent_name = agent.name,
                meta       = {"from": source.name, "to": agent.name,
                              "capture_via": "openai_agents_hooks"},
            ))

    _hooks = _ObsHooks()
    _orig  = Agent.__init__
    _orig_run = Runner.run
    _orig_run_sync = Runner.run_sync
    _orig_run_streamed = Runner.run_streamed

    def _patched(self, *args, **kwargs):
        kwargs["hooks"] = _hooks
        _orig(self, *args, **kwargs)

    Agent.__init__ = _patched

    def _prepare_context(existing_context, trace_id: str):
        if existing_context is None:
            return {
                "_obs_trace_id": trace_id,
                "_obs_timers": {},
                "_obs_agent_span_ids": {},
                "_obs_tool_span_ids": {},
            }
        if isinstance(existing_context, dict):
            existing_context.setdefault("_obs_trace_id", trace_id)
            existing_context.setdefault("_obs_timers", {})
            existing_context.setdefault("_obs_agent_span_ids", {})
            existing_context.setdefault("_obs_tool_span_ids", {})
            return existing_context
        if getattr(existing_context, "_obs_trace_id", None) is None:
            setattr(existing_context, "_obs_trace_id", trace_id)
        if getattr(existing_context, "_obs_timers", None) is None:
            setattr(existing_context, "_obs_timers", {})
        if getattr(existing_context, "_obs_agent_span_ids", None) is None:
            setattr(existing_context, "_obs_agent_span_ids", {})
        if getattr(existing_context, "_obs_tool_span_ids", None) is None:
            setattr(existing_context, "_obs_tool_span_ids", {})
        return existing_context

    async def _patched_run(cls, starting_agent, input, *args, callbacks=None, **kwargs):
        trace_id = str(uuid.uuid4())
        previous_trace = _set_trace(trace_id, None)
        kwargs["context"] = _prepare_context(kwargs.get("context"), trace_id)
        ctx = context_from_callbacks(callbacks).copy() if callbacks else {}
        ctx.setdefault("framework", "openai-agents")
        previous_context = _set_context(**ctx)
        try:
            return await _orig_run.__func__(cls, starting_agent, input, *args, **kwargs)
        finally:
            if previous_context is not None:
                _restore_context(previous_context)
            _restore_trace(previous_trace)

    def _patched_run_sync(cls, starting_agent, input, *args, callbacks=None, **kwargs):
        trace_id = str(uuid.uuid4())
        previous_trace = _set_trace(trace_id, None)
        kwargs["context"] = _prepare_context(kwargs.get("context"), trace_id)
        ctx = context_from_callbacks(callbacks).copy() if callbacks else {}
        ctx.setdefault("framework", "openai-agents")
        previous_context = _set_context(**ctx)
        try:
            return _orig_run_sync.__func__(cls, starting_agent, input, *args, **kwargs)
        finally:
            if previous_context is not None:
                _restore_context(previous_context)
            _restore_trace(previous_trace)

    def _patched_run_streamed(cls, starting_agent, input, *args, callbacks=None, **kwargs):
        trace_id = str(uuid.uuid4())
        previous_trace = _set_trace(trace_id, None)
        kwargs["context"] = _prepare_context(kwargs.get("context"), trace_id)
        ctx = context_from_callbacks(callbacks).copy() if callbacks else {}
        ctx.setdefault("framework", "openai-agents")
        previous_context = _set_context(**ctx)
        try:
            return _orig_run_streamed.__func__(cls, starting_agent, input, *args, **kwargs)
        finally:
            if previous_context is not None:
                _restore_context(previous_context)
            _restore_trace(previous_trace)

    Runner.run = classmethod(_patched_run)
    Runner.run_sync = classmethod(_patched_run_sync)
    Runner.run_streamed = classmethod(_patched_run_streamed)
