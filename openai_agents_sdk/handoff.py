from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_tracing_disabled
)

from openai import AsyncOpenAI
import asyncio

# Disable tracing
set_tracing_disabled(True)

# Connect to local Ollama server
client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"   # required but can be any string
)

# Model configuration
model = OpenAIChatCompletionsModel(
    model="llama3.1",
    openai_client=client
)

# Spanish agent
spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    model=model
)

# French agent
french_agent = Agent(
    name="French agent",
    instructions="You only speak French.",
    model=model
)

# English agent
english_agent = Agent(
    name="English agent",
    instructions="You only speak English.",
    model=model
)

# Hindi agent
hindi_agent = Agent(
    name="Hindi agent",
    instructions="You only speak Hindi.",
    model=model
)

# Router agent
triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "Detect the language of the user request "
        "and handoff to the correct language agent."
    ),
    handoffs=[
        spanish_agent,
        french_agent,
        english_agent,
        hindi_agent
    ],
    model=model
)

# Main loop
async def main():

    while True:

        user_input = input("> ").strip()

        if user_input.lower() == "quit":
            break

        result = await Runner.run(
            triage_agent,
            input=user_input
        )

        print("\n")
        print(result.final_output)
        print("\n")

# Run app
if __name__ == "__main__":
    asyncio.run(main())