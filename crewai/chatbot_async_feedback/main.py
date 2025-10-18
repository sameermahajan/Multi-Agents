# main.py
import asyncio
from chatbot_crew import create_chatbot_crew
from background_crew import run_background_analysis

async def background_loop(latest_result_ref):
    """
    Continuously rerun background crew, update latest_result_ref in place.
    Restarts automatically after each completion or timeout.
    """
    while True:
        result = await run_background_analysis()
        latest_result_ref["result"] = result
        await asyncio.sleep(1)  # small delay between cycles

async def main():
    chatbot_crew = create_chatbot_crew()

    latest_result_ref = {"result": None}  # shared state between loops

    # Start background crew loop
    asyncio.create_task(background_loop(latest_result_ref))

    background_result = None
    print("🤖 Chatbot started. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Get latest background result (if available)
        background_result = latest_result_ref["result"]

        # Combine background insights if available
        if background_result:
            print(f"\n📥 Background insights available: {background_result}\n")
            chatbot_input = (
                f"User said: {user_input}\n\n"
                f"Background insights: {background_result}\n\n"
                "Now respond accordingly."
            )
        else:
            print("\n(No background insights available yet.)\n")
            chatbot_input = f"User said: {user_input}"

        # Run chatbot task synchronously
        response = chatbot_crew.kickoff(inputs={"input": chatbot_input})
        print(f"Chatbot: {response}\n")

asyncio.run(main())
