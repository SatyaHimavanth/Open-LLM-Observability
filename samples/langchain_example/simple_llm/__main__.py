from dotenv import load_dotenv
load_dotenv()

import universal_agent_obs
from universal_agent_obs.langchain import TraceContextCallbackHandler

from samples.langchain_example.llms import get_chat_model


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
    message = [
                {
                    "role": "user", 
                    "content": "Hi how are you?"
                }
            ]
    
    try:
        llm = get_chat_model()
        response = llm.invoke(message, config=trace_config)
        print(response.content)
    finally:
        universal_agent_obs.flush(timeout=10)
    

if __name__ == "__main__":
    main()