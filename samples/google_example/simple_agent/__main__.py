import asyncio

from dotenv import load_dotenv
load_dotenv()


import universal_agent_obs
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from universal_agent_obs.google import TraceContextCallbackHandler

from samples.google_example.simple_agent.agent import root_agent


trace_callback = TraceContextCallbackHandler(
    user={
        "id": "demo-user",
        "name": "Demo User",
        "email": "demo.user@example.com",
        "account": "demo-account",
        "role": "tester",
    },
    tags=["sample", "google-adk"],
    metadata={"environment": "local"},
)


async def main():
    app_name = "sample_agent"
    user_id = "demo-user"
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=app_name, user_id=user_id)
    runner = Runner(agent=root_agent, app_name=app_name, session_service=session_service)
    message = types.Content(
        role="user",
        parts=[
            types.Part(
                text="What is the weather in Hyderabad and what is the current time there?"
            )
        ],
    )

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=message,
            callbacks=[trace_callback],
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "text", None):
                        print(part.text)
    except Exception as e:
        print(e)
    finally:
        universal_agent_obs.flush(timeout=10)


if __name__ == "__main__":
    asyncio.run(main())
