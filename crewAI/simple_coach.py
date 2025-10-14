from crewai import Agent, Crew, Task, LLM

# ---- Base model (Local Ollama) ----
ollama_model=LLM(model="ollama/llama3.1", base_url="http://localhost:11434")

# ---- Main Conversational Agent ----
main_agent = Agent(
    name="Main Conversational Agent",
    role="Converses with user naturally.",
    goal="Provide friendly, context-aware replies.",
    backstory="You're a helpful assistant that tailors responses based on user profile and conversation history.",
    llm=ollama_model,
)

# ---- Tasks ----
main_task = Task(
    description="Generate user response based on conversation context.",
    expected_output="Conversational response text.",
    agent=main_agent,
)

# ---- Crew ----
crew = Crew(
    name="Simple Chatbot Crew",
    agents=[main_agent],
    tasks=[main_task],
    process="sequential",
)

# ---- Run a sample conversation ----
result = crew.kickoff(inputs={"user_message": "Hey, what should I learn next in Python?"})

print(result)
