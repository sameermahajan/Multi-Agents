# Pre requisites

python

virtual environement (recommended)

# Setup

pip install crewai openai

# Run

python newsroom.py

# Modifications

## Change Topic

e.e. AI in schools to climate change

## Add a new agent

e.g.

fact_checker = Agent(
    role="Fact Checker",
    goal="Verify the article facts",
    backstory="You identify doubtful claims.",
    verbose=True
)

add a task

add to crew

## Make Funny Agents
### sarcastic editor
### dramatic reporter
### meme writer

## Reduce or increase word limit



