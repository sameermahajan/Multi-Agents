import os
import time
from typing import TypedDict, List, Any

# --------------------------
# LangChain / LangGraph imports
# --------------------------

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# --------------------------
# 1. LLM (Local Ollama via OpenAI-compatible API)
# --------------------------

llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",    # Ollama local server
    api_key="ollama",                        # dummy key
    model="llama3.1"                         # or any installed Ollama model
)


# --------------------------
# 2. Tools: Tavily, Math, Wait
# --------------------------

# Set Tavily API Key
os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"

# Real web search tool
search_tool = TavilySearchResults(max_results=3)


@tool
def math(expr: str):
    """Evaluate a Python math expression."""
    try:
        return eval(expr)
    except Exception as e:
        return f"Math error: {e}"


@tool
def wait(seconds: int):
    """Pause for N seconds."""
    time.sleep(seconds)
    return f"Waited {seconds} seconds"


TOOLS = [search_tool, math, wait]


# --------------------------
# 3. LangGraph State Structure
# --------------------------

class AgentState(TypedDict):
    messages: List[Any]


# --------------------------
# 4. Agent Node (LLM decides next action)
# --------------------------

def agent_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


# --------------------------
# 5. Tool Execution Node
# --------------------------

tool_node = ToolNode(TOOLS)


# --------------------------
# 6. Router: Should we call a tool or terminate?
# --------------------------

def router(state: AgentState):
    last = state["messages"][-1]

    # If LLM wants to call a tool (OpenAI tool_calls format)
    if last.additional_kwargs.get("tool_calls"):
        return "tool"

    # Otherwise: terminate agent
    return "end"


# --------------------------
# 7. Build the LangGraph
# --------------------------

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("agent")

# LLM → tool
graph.add_conditional_edges(
    "agent",
    router,            # routing function
    {
        "tool": "tool",
        "end": END,
    }
)

# Tool → LLM
graph.add_edge("tool", "agent")

app = graph.compile()

# --------------------------
# 8. Runner Function
# --------------------------

def run_agent(query: str):
    state = {"messages": [{"role": "user", "content": query}]}
    final_state = None

    print("\n========== AGENT START ==========\n")

    for update in app.stream(state):
        print(update)
        final_state = update

    print("\n========== FINAL ANSWER ==========\n")
    last_msg = final_state["agent"]["messages"][-1]
    print(last_msg.content)


# --------------------------
# 9. Example Runs – uncomment to test
# --------------------------

if __name__ == "__main__":

    print("\n\n### EXAMPLE 1 — Simple Search ###\n")
    run_agent("What is FastAPI?")

    print("\n\n### EXAMPLE 2 — Multi-step with math + wait ###\n")
    run_agent("Compute 12 * 8, then wait 2 seconds, then respond with the result.")

    print("\n\n### EXAMPLE 3 — Multi-search + summary ###\n")
    run_agent("Search 'Python language' and 'RAG technique', then summarize both.")
