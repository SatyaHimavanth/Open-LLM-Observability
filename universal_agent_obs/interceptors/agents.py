"""
interceptors/autogen.py — Patches ConversableAgent for AutoGen 0.2/0.4.
interceptors/openai_agents.py — Patches Agent with AgentHooks.
Both in one file for convenience.
"""
import time, uuid
from ..core import Span, emit


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
    from agents import Agent, AgentHooks, RunContextWrapper

    class _ObsHooks(AgentHooks):
        async def on_start(self, ctx: RunContextWrapper, agent: Agent):
            if "_obs_trace_id" not in ctx.context:
                ctx.context["_obs_trace_id"] = str(uuid.uuid4())
            ctx.context[f"_t0_{agent.name}"] = time.perf_counter()
            emit(Span(
                trace_id   = ctx.context["_obs_trace_id"],
                event      = "agent_start", resource="agent",
                framework  = "openai-agents",
                agent_name = agent.name,
                model      = str(agent.model) if agent.model else None,
                meta       = {"capture_via": "openai_agents_hooks"},
            ))

        async def on_end(self, ctx: RunContextWrapper, agent: Agent, output):
            emit(Span(
                trace_id   = ctx.context.get("_obs_trace_id"),
                event      = "agent_end", resource="agent",
                framework  = "openai-agents",
                agent_name = agent.name,
                latency_ms = (time.perf_counter() -
                              ctx.context.get(f"_t0_{agent.name}", time.perf_counter())) * 1000,
                meta       = {
                    "output": str(output)[:500],
                    "usage":  ctx.usage.model_dump() if hasattr(ctx, "usage") and ctx.usage else {},
                    "capture_via": "openai_agents_hooks",
                },
            ))

        async def on_tool_start(self, ctx: RunContextWrapper, agent: Agent, tool):
            ctx.context[f"_tool_t0_{tool.name}"] = time.perf_counter()
            emit(Span(
                trace_id   = ctx.context.get("_obs_trace_id"),
                event      = "tool_start", resource="tool",
                framework  = "openai-agents",
                agent_name = agent.name,
                tool_name  = tool.name,
                meta       = {"capture_via": "openai_agents_hooks"},
            ))

        async def on_tool_end(self, ctx: RunContextWrapper, agent: Agent, tool, result: str):
            emit(Span(
                trace_id   = ctx.context.get("_obs_trace_id"),
                event      = "tool_end", resource="tool",
                framework  = "openai-agents",
                agent_name = agent.name,
                tool_name  = tool.name,
                tool_output= result[:1000],
                latency_ms = (time.perf_counter() -
                              ctx.context.get(f"_tool_t0_{tool.name}", time.perf_counter())) * 1000,
                meta       = {"capture_via": "openai_agents_hooks"},
            ))

        async def on_handoff(self, ctx: RunContextWrapper, agent: Agent, source: Agent):
            emit(Span(
                trace_id   = ctx.context.get("_obs_trace_id"),
                event      = "handoff", resource="agent",
                framework  = "openai-agents",
                agent_name = agent.name,
                meta       = {"from": source.name, "to": agent.name,
                              "capture_via": "openai_agents_hooks"},
            ))

    _hooks = _ObsHooks()
    _orig  = Agent.__init__

    def _patched(self, *args, **kwargs):
        kwargs["hooks"] = _hooks
        _orig(self, *args, **kwargs)

    Agent.__init__ = _patched
