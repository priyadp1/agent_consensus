import asyncio

async def agent_talk(agents, question, options, selections, run_model, max_rounds=3):
    history = []

    option_block = "\n".join(
        [f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)]
    )

    for _ in range(max_rounds):
        round_answers = {}

        for agent_id in agents:
            if history:
                prior = "\n".join(
                    f"Agent {a}: {resp}"
                    for a, resp in history[-1].items()
                    if a != agent_id
                )
                context = f"""
Here are answers from other respondents:
{prior}

You may revise or reaffirm your answer.
"""
            else:
                context = ""

            prompt = f"""
You are answering a subjective public opinion survey question.

Question:
{question}

Answer options:
{option_block}

{context}

Instructions:
- Choose ONE option.
- Respond ONLY in the format:
  Final answer: <LETTER>
- Do NOT add explanations or extra text.

Final answer:
"""

            response = run_model(prompt).strip()
            response = " ".join(response.split())
            round_answers[agent_id] = response

        history.append(round_answers)

    return history