from crewai import Task, Crew, Process

from samples.crewai_example.simple_agent.agent import (
    weather_agent,
    time_agent,
)

import universal_agent_obs
from universal_agent_obs.crewai import TraceContextCallbackHandler

# Best-effort flush of observability spans
try:

    task_weather = Task(
        description="Analyze weather for Tokyo.",
        expected_output="A weather report for Tokyo.",
        agent=weather_agent
    )

    task_time = Task(
        description="Determine local time.",
        expected_output="Current time.",
        agent=time_agent
    )

    # Create trace callback handler and attach as step/task callbacks
    trace_callback = TraceContextCallbackHandler(
        user={
            "id": "demo-user",
            "name": "Demo User",
            "email": "demo.user@example.com",
        },
        tags=["sample", "crewai"],
        metadata={"environment": "local"},
    )

    project_crew = Crew(
        agents=[weather_agent, time_agent],
        tasks=[task_weather, task_time],
        process=Process.sequential, # The crew manages the order
        step_callback=trace_callback,
        task_callback=trace_callback,
        verbose=True
    )

    result = project_crew.kickoff()
    print(result)
finally:
    universal_agent_obs.flush(timeout=5)

