from dotenv import load_dotenv
load_dotenv()

import universal_agent_obs
from openai import OpenAI
from universal_agent_obs.openai import TraceContextCallbackHandler


client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",  # required by OpenAI SDK, ignored by Ollama
)

trace_callback = TraceContextCallbackHandler(
    user={
        "id": "demo-user",
        "name": "Demo User",
        "email": "demo.user@example.com",
    },
    tags=["sample", "openai-sdk"],
    metadata={"environment": "local"},
)

response = client.chat.completions.create(
    model="qwen3.5:2b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi."},
    ],
    callbacks=[trace_callback],
)

print(response.choices[0].message.content)
universal_agent_obs.flush(timeout=5)