import asyncio

async def agent_talk(agents, question, options, selections, run_model, max_rounds=3):
    history = []

    for i in range(1, max_rounds + 1):
        round_answers = {}

        for agent_id in agents:
            if history:
                prior_opinions = "\n".join(
                    f"Agent {a}: {resp}"
                    for a, resp in history[-1].items()
                    if a != agent_id
                )
                context = (
                    "Here are answers from other respondents:\n"
                    f"{prior_opinions}\n\n"
                    "You may revise or reaffirm your opinion."
                )
            else:
                context = ""

            prompt = f"""
You are answering a subjective public opinion question.

Question:
{question}

Answer options:
{options}

Global response distributions by country:
{selections}

{context}

Task:
Give your own considered answer and briefly explain your reasoning.
"""
            response = run_model(prompt).strip()
            round_answers[agent_id] = response

        history.append(round_answers)

    return history