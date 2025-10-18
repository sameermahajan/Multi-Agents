# background_crew.py
import asyncio
from crewai import Crew, Agent, LLM, Task

TIMEOUT_SECONDS = 30  # adjust as needed

client = LLM(model="ollama/llama3.1", base_url="http://localhost:11434")

def create_background_crew():
    background_agent = Agent(
        role="Background Analyzer",
        goal="Perform slow background processing and produce insights.",
        backstory=(
            "You analyze data in the background and produce insights to help the chatbot improve its responses."
        ),
        llm=client,
    )

    background_task = Task(
        description="Generate background insights asynchronously (e.g. summarize news, user data, etc.)",
        expected_output="Text summary or insight.",
        agent=background_agent,
    )

    return Crew(agents=[background_agent], tasks=[background_task])

async def run_background_analysis():
    crew = create_background_crew()

    try:
        # Run the crew in a thread with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(crew.kickoff()),  # CrewAI runs synchronously, so move it to a thread
            timeout=TIMEOUT_SECONDS
        )
        return result

    except asyncio.TimeoutError:
        return f"⚠️ Background analysis timed out after {TIMEOUT_SECONDS} seconds."
    except Exception as e:
        # Catch unexpected runtime errors and report them safely
        return f"❌ Background analysis failed due to error: {str(e)}"