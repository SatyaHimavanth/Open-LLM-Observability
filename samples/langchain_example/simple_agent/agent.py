from dotenv import load_dotenv
load_dotenv()

import universal_agent_obs  # installs tracing interceptors on import
from langchain.agents import create_agent

from universal_agent_obs.langchain import TraceContextCallbackHandler

from samples.langchain_example.llms import get_chat_model
from samples.langchain_example.simple_agent.tools import TOOLS
from samples.langchain_example.simple_agent.prompts import SYSTEM_PROMPT


agent = create_agent(
    get_chat_model(),
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT
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
