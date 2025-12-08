# minimal_rag_crew_ollama.py

import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import RagTool
from crewai_tools.tools.rag import RagToolConfig, ProviderSpec
# -------------------------------------------------
# 1. Use Ollama as the LLM
# -------------------------------------------------
# No API key required for Ollama.
# Just ensure Ollama is running locally:
#   ollama serve
# And pull a model before running:
#   ollama pull llama3

llm = LLM(
    model="ollama/llama3",     # Use any installed model
    base_url="http://localhost:11434"   # Ollama default endpoint
    # api_key="not-needed"
)

embedding_model: ProviderSpec = {
    "provider": "ollama",
    "config": {
        "model_name": "nomic-embed-text",  # or another embedding-capable model
        "url": "http://localhost:11434/api/embeddings",
    },
}

rag_config: RagToolConfig = {
    "embedding_model": embedding_model
    # you could also configure "vectordb" here if needed
}

# -------------------------------------------------
# 2. Create the RAG tool and index local docs
# -------------------------------------------------
rag_tool = RagTool(
    name="LocalDocsRAG",
    description="Answers questions using local knowledge documents.",
    config=rag_config,    # <-- use config, NOT embeddings=
    summarize=True,
    verbose=True
)

# Your documents folder (supports PDFs, text, markdown, etc.)
rag_tool.add(
    data_type="directory",
    path="./data"
)

question = "What does our refund policy say about cancellations?"

# rag_raw_answer = rag_tool.run({"query": question})
rag_raw_answer = rag_tool.run(question)

print(rag_raw_answer)

# -------------------------------------------------
# 3. Define a single RAG-enabled agent
# -------------------------------------------------
rag_agent = Agent(
    role="Documentation Assistant",
    goal="Use provided documents to answer accurately.",
    backstory="You specialize in reading the company's documents.",
    tools=[],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# -------------------------------------------------
# 4. Define a single task
# -------------------------------------------------

# rag_task = Task(
#    description=(
#        "Using the RAG tool, answer the question: "
#        f"'{question}'. Only use the documents. If not found, say so."
#    ),
#    agent=rag_agent,
#    expected_output="A short, accurate summary of the answer from the docs."
#)

rag_task = Task(
    description=(
        "You are given an answer produced by a retrieval system from our policy documents.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved answer:\n{rag_raw_answer}\n\n"
        "Rewrite this answer in 3–5 clear sentences, keeping only what is supported "
        "by the retrieved text. If the retrieved answer is empty or says it couldn't "
        "find anything, say that the policy is not available in the documents."
    ),
    agent=rag_agent,
    expected_output="A short, clear summary of the cancellation rules from the refund policy.",
)

# -------------------------------------------------
# 5. Crew kickoff
# -------------------------------------------------
crew = Crew(
    agents=[rag_agent],
    tasks=[rag_task],
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n=== FINAL ANSWER ===\n")
    print(result)
