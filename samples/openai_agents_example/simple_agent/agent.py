from datetime import datetime
from openai import AsyncOpenAI

from dotenv import load_dotenv
load_dotenv()

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
import universal_agent_obs
from universal_agent_obs.openai import TraceContextCallbackHandler


# Disable OpenAI tracing because this example uses local Ollama
set_tracing_disabled(True)


# OpenAI-compatible Ollama client
ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",  # required by OpenAI client, ignored by Ollama
)


@function_tool
def get_current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@function_tool
def get_current_weather(city: str) -> str:
    """Return the current weather of a city"""
    return f"Weather in {city}: Sunny, 72°F"


agent = Agent(
    name="Local Ollama Tool Agent",
    instructions=(
        "You are a helpful local AI agent. "
        "Use tools when the user asks for time or weather."
    ),
    model=OpenAIChatCompletionsModel(
        model="qwen3.5:2b",
        openai_client=ollama_client,
    ),
    tools=[
        get_current_time,
        get_current_weather,
    ],
)

trace_callback = TraceContextCallbackHandler(
    user={
        "id": "demo-user",
        "name": "Demo User",
        "email": "demo.user@example.com",
    },
    tags=["sample", "openai-agents"],
    metadata={"environment": "local"},
)


async def main():
    result = await Runner.run(
        agent,
        "What time is it now? Also what is the weather in Hyderabad?.",
        callbacks=[trace_callback],
    )

    print("\nFinal output:")
    print(result.final_output)
    universal_agent_obs.flush(timeout=5)


