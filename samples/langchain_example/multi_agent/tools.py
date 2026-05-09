
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}. Satya is GenAI developer"


def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"


TOOLS = [search, get_weather]