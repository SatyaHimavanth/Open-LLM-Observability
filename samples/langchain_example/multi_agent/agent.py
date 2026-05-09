from dotenv import load_dotenv
load_dotenv()

import universal_agent_obs  # installs tracing interceptors on import
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    TodoListMiddleware
)
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.backends import StoreBackend

from universal_agent_obs.langchain import TraceContextCallbackHandler

from samples.langchain_example.llms import get_chat_model
from samples.langchain_example.simple_agent.tools import TOOLS, get_weather, search
from samples.langchain_example.simple_agent.prompts import SYSTEM_PROMPT


my_backend = StoreBackend(
    namespace=lambda ctx: (ctx.runtime.context.user_id,),
)

middleware=[
        SummarizationMiddleware(
            model=get_chat_model(),
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "search": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
            }
        ),
        TodoListMiddleware(),
        SubAgentMiddleware(
            backend=my_backend,
            subagents=[
                {
                    "name": "weather",
                    "description": "This subagent can get weather in cities.",
                    "system_prompt": "Use the get_weather tool to get the weather in a city.",
                    "tools": [get_weather],
                    "model": get_chat_model(),
                    "middleware": [],
                }
            ],
        )
    ]

agent = create_agent(
    model=get_chat_model(),
    tools=[search],
    system_prompt=SYSTEM_PROMPT,
    middleware=middleware
)


def main():
    trace_user = {
        "name": "Demo User",
        "email": "demo.user@example.com",
        "account": "demo-account",
        "role": "tester",
    }
    trace_config = {
        "tags": ["sample", "langchain", "weather"],
        "metadata": {
            "user": trace_user,
            "environment": "local",
        },
        "callbacks": [
            TraceContextCallbackHandler(
                user=trace_user,
                tags=["sample", "langchain", "weather"],
                metadata={"environment": "local"},
            )
        ],
    }
    message = {
            "messages": [
                {
                    "role": "user", 
                    "content": "What is the current weather in Hyderabad" # and can you find out about the person named satya
                }
            ]
        }
    try:
        for chunk in agent.stream(message, config=trace_config, stream_mode="values"):
            latest_message = chunk["messages"][-1]
            latest_message.pretty_print()
    finally:
        universal_agent_obs.flush(timeout=10)


if __name__ == "__main__":
    main()
