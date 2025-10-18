# chatbot_crew.py
from crewai import Crew, Agent, LLM, Task

client = LLM(model="ollama/llama3.1", base_url="http://localhost:11434")

def create_chatbot_crew():
    chatbot_agent = Agent(
        role="Chatbot Assistant",
        goal="Converse with user and incorporate background insights when available.",
        backstory=(
            "You are a friendly and intelligent assistant. "
            "You continue chatting normally while waiting for background insights. "
            "If new insights appear, include them naturally in your next reply."
        ),
        llm=client,
    )

    chatbot_task = Task(
        description="Talk with user and optionally include any insights when available.",
        expected_output="Conversational, contextual response.",
        agent=chatbot_agent,
    )

    return Crew(agents=[chatbot_agent], tasks=[chatbot_task])
