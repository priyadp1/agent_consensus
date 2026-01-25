async def agent_talk(agents, agent_runners, question, options, selections, max_rounds=3):
    history = []

    letters = [chr(65 + i) for i in range(len(options))]
    allowed = ", ".join(letters)

    option_block = "\n".join(
        f"({letters[i]}) {opt}" for i, opt in enumerate(options)
    )

    for _ in range(max_rounds):
        round_answers = {}

        for agent_id in agents:
            if history:
                prior = "\n".join(
                    f"{a}: {resp}"
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
- Choose exactly ONE option by its letter ({allowed})
- Explain your reasoning briefly
- End your response with a final line in this format:

ANSWER: <LETTER>

where <LETTER> is one of: {allowed}

Please ensure the final answer line appears exactly as shown, with no additional text after it.
"""

            runner = agent_runners[agent_id]
            response = runner(prompt)

            if not isinstance(response, str):
                response = ""
            else:
                response = response.strip()

            round_answers[agent_id] = response

        history.append(round_answers)

    return history