# Have a folder with
#   langgraph.json
#   all .py specified in above file
# run
#   langgraph dev
# run this program
#    python api.py

from langgraph_sdk import get_client

# This is the URL of the local development server
URL = "http://127.0.0.1:2024"
client = get_client(url=URL)

import asyncio
from langchain_core.messages import HumanMessage

async def main():
    # Search all hosted graphs
    assistants = await client.assistants.search()
    print(assistants)  # List all graphs
    thread = await client.threads.create()

    # Input
    input = {"messages": [HumanMessage(content="Multiply 3 by 2.")]}

    # Stream
    async for chunk in client.runs.stream(
        thread['thread_id'],
        "agent",
        input=input,
        stream_mode="values",
    ):
        if chunk.data and chunk.event != "metadata":
            print(chunk.data['messages'][-1])

asyncio.run(main()) # Run the async function

