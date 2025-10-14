from crewai import Agent, Crew, Task, LLM

# ---- Base model (Local Ollama) ----
ollama_model=LLM(model="ollama/llama3.1", base_url="http://localhost:11434")

# ---- Main Conversational Agent ----
main_agent = Agent(
    name="Main Conversational Agent",
    role="Converses with user naturally using context and profile.",
    goal="Provide friendly, context-aware replies.",
    backstory="You're a helpful assistant that tailors responses based on user profile and conversation history.",
    llm=ollama_model,
)

# ---- Evaluator 1: Personalization ----
personalization_evaluator = Agent(
    name="Personalization Evaluator",
    role="Evaluates if the main agent's response reflects user's known profile.",
    goal="Ensure conversation includes personalization and user context.",
    backstory="You assess if the assistant's replies are personalized based on user data and past interactions.",
    llm=ollama_model,
)

# ---- Evaluator 2: Personality Adherence ----
personality_evaluator = Agent(
    name="Personality Evaluator",
    role="Checks if tone and style match user's defined personality.",
    goal="Ensure conversation reflects user's personality traits and tone.",
    backstory="You evaluate if the assistant's replies align with the user's expressed personality.",
    llm=ollama_model,
)

# ---- Tasks ----
main_task = Task(
    description="Generate user response based on conversation context.",
    expected_output="Conversational response text.",
    agent=main_agent,
)

personalization_task = Task(
    description="Evaluate the main agent's output for personalization accuracy.",
    expected_output="Feedback on personalization (score and suggestions).",
    agent=personalization_evaluator,
)

personality_task = Task(
    description="Evaluate if the main agent's output fits user's personality.",
    expected_output="Feedback on tone/style adherence.",
    agent=personality_evaluator,
)

# ---- Crew ----
crew = Crew(
    name="Hierarchical Chatbot Crew",
    agents=[main_agent, personalization_evaluator, personality_evaluator],
    tasks=[main_task, personalization_task, personality_task],
    process="hierarchical",  # ensures top-down task flow
    manager_llm=ollama_model
)

# ---- Run a sample conversation ----
result = crew.kickoff(inputs={
    "user_message": "Hey, what should I learn next in Python?",
    "user_profile": {
        "name": "Sameer",
        "interests": ["AI", "automation", "teaching"],
        "tone": "curious and practical"
    }
})

print(result)
