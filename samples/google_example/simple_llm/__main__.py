from dotenv import load_dotenv
load_dotenv()

import universal_agent_obs
from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

try:
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite", contents="Hi, How are you?"
    )
    print(response.text)
finally:
    universal_agent_obs.flush(timeout=10)