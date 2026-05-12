from crewai import Agent, LLM
from crewai.tools import tool
from datetime import datetime


from dotenv import load_dotenv
load_dotenv()


@tool("weather_tool")
def weather_tool(city: str) -> str:
    """
    Returns fake weather info for a city.
    """
 
    weather_data = {
        "hyderabad": "35°C, Sunny",
        "london": "12°C, Rainy",
        "tokyo": "18°C, Cloudy",
    }
 
    return weather_data.get(
        city.lower(),
        f"No weather data found for {city}"
    )
 
 
@tool("time_tool")
def time_tool(_: str = "") -> str:
    """
    Returns current local time.
    """
 
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 


# 1. Define the local model via Ollama
llm = LLM(
    model="ollama/qwen3.5:2b",
    base_url="http://localhost:11434"
)
# llm = LLM(
#     model="azure/gpt-4.1"
# )

# 2. Define your Agents
weather_agent = Agent(
    role='Whether Expert',
    goal='Provide weather information',
    backstory='You are a weather assistant.',
    tools=[weather_tool],
    llm=llm,
    verbose=True
)

time_agent = Agent(
    role='Time Expert',
    goal='Provide current time',
    backstory='You know time accurately.',
    tools=[time_tool],
    llm=llm,
    verbose=True
)

router_agent = Agent(
    role="Router",
    goal="""
    Decide which specialist should answer:
    - weather questions -> Weather Expert
    - time questions -> Time Expert
    """,
    backstory="You route user queries.",
    llm=llm,
    verbose=True,
)