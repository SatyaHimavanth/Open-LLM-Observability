from dotenv import load_dotenv
load_dotenv()

import universal_agent_obs
from universal_agent_obs.google import TraceContextCallbackHandler
from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()
trace_callback = TraceContextCallbackHandler(
    user={
        "id": "demo-user",
        "name": "Demo User",
        "email": "demo.user@example.com",
        "account": "demo-account",
        "role": "tester",
    },
    tags=["sample", "google-genai"],
    metadata={"environment": "local"},
)

try:
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents="Hi, How are you?",
        callbacks=[trace_callback],
    )
    print(response.text)
finally:
    universal_agent_obs.flush(timeout=10)
