from crewai import Agent, Task, Crew

# Reporter Agent
reporter = Agent(
    role="Reporter",
    goal="Write a short news report about AI in schools",
    backstory="You are a technology journalist.",
    verbose=True
)

# Editor Agent
editor = Agent(
    role="Editor",
    goal="Improve the news article",
    backstory="You improve clarity and grammar.",
    verbose=True
)

# Headline Agent
headline_writer = Agent(
    role="Headline Writer",
    goal="Create a catchy headline",
    backstory="You write exciting newspaper headlines.",
    verbose=True
)

# Tasks
task1 = Task(
    description="Write a 100-word article about AI helping students learn.",
    agent=reporter
)

task2 = Task(
    description="Edit the article to make it clearer and more engaging.",
    agent=editor
)

task3 = Task(
    description="Create an interesting headline for the article.",
    agent=headline_writer
)

# Crew
crew = Crew(
    agents=[reporter, editor, headline_writer],
    tasks=[task1, task2, task3]
)

# Run
result = crew.kickoff()

print("\n===== FINAL RESULT =====\n")
print(result)
