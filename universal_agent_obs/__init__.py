"""
universal_agent_obs
===================
Zero-code observability for Python agent frameworks.

Usage
-----
Option 1 — import once at the top of your entry point:
    import universal_agent_obs

Option 2 — environment variable only (sitecustomize approach):
    AGENT_OBS=1 AGENT_FRAMEWORK=langchain python your_agent.py

The module auto-detects which frameworks are installed and injects
callbacks/patches without touching your agent code.

Config env vars
---------------
AGENT_OBS=1                    Enable (default: 1)
AGENT_OBS_URL=http://host:4317 Collector server URL (default: localhost:4317)
AGENT_FRAMEWORK=langchain      Framework hint (auto-detected if omitted)
"""

import sys
import importlib
from .core import ENABLED, FRAMEWORK, PROJECT_NAME, SERVER_URL

_installed: list[str] = []


def _try(name: str, fn):
    try:
        if fn():
            _installed.append(name)
    except Exception as e:
        pass   # never crash the agent


def install():
    if not ENABLED:
        return

    # ── Framework-specific callbacks (preferred) ────────────────
    from .interceptors.langchain import install as lc
    _try("langchain", lc)

    from .interceptors.crewai import install as cr
    _try("crewai", cr)

    from .interceptors.agents import install_autogen, install_openai_agents
    _try("autogen", install_autogen)
    _try("openai-agents", install_openai_agents)

    from .interceptors.google_adk import install as gadk
    _try("google-adk", gadk)

    from .interceptors.google_genai import install as ggenai
    _try("google-genai", ggenai)

    from .interceptors.openai_sdk import install as osdk
    _try("openai-sdk", osdk)

    # ── HTTP transport fallback (catches everything else) ───────
    from .interceptors.http import install as http
    _try("http", http)

    if _installed:
        print(
            f"[agent-obs] active on {SERVER_URL} "
            f"| framework={FRAMEWORK} "
            f"| project={PROJECT_NAME} "
            f"| interceptors={','.join(_installed)}"
        )


# Auto-install on import
install()


# ── Public helpers ───────────────────────────────────────────────────────────
from .core import (  # noqa: F401
    Span,
    emit,
    flush,
    get_or_new_trace,
    new_trace,
    set_context,
    trace_context,
)

__all__ = [
    "install",
    "new_trace",
    "get_or_new_trace",
    "Span",
    "emit",
    "flush",
    "set_context",
    "trace_context",
    "SERVER_URL",
    "PROJECT_NAME",
]
__version__ = "0.1.0"
