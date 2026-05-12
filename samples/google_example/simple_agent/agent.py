from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    return {"status": "success", "city": city, "time": "10:30 AM"}

def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"


model_name = "ollama_chat/qwen3.5:2b" # "azure/gpt-4.1"
model = LiteLlm(model=model_name)

weather_agent = Agent(
    name="weather_agent_v1",
    model=model,
    description="Provides weather information for specific cities.",
    instruction="You are a helpful weather assistant. "
                "When the user asks for the weather in a specific city, "
                "use the 'get_weather' tool to find the information. "
                "If the tool returns an error, inform the user politely. "
                "If the tool is successful, present the weather report clearly.",
    tools=[get_weather],
)

root_agent = Agent(
    model=model,
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
    tools=[get_current_time],
    sub_agents=[
        weather_agent
    ]
)
