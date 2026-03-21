import asyncio


async def agent_talk(agents, agent_runners, question, options, selections=None, max_rounds=3, history=None, show_agent_names=True):
    # selections is unused but kept for API compatibility
    # history: pass an existing list of round dicts to resume a previous conversation
    # show_agent_names: if True, peers are identified by their agent_id; if False, by "Respondent N"
    if history is None:
        history = []

    # Map option indices to letters (A, B, C, ...)
    letters = [chr(65 + i) for i in range(len(options))]
    allowed = ", ".join(letters)

    # Format the answer options as a labeled block for the prompt
    option_block = "\n".join(
        f"({letters[i]}) {opt}" for i, opt in enumerate(options)
    )

    # Resume from where the previous history left off rather than restarting at round 0
    start_round = len(history)

    for _ in range(start_round, max_rounds):
        round_answers = {}

        def build_prompt_for_agent(agent_id):
            # If prior rounds exist, inject other agents' most recent answers as context
            if history:
                other_responses = [
                    (a, resp) for a, resp in history[-1].items()
                    if a != agent_id  # Exclude the current agent's own prior answer
                ]
                if show_agent_names:
                    # Label each response with the actual agent/model name
                    prior = "\n".join(f"{a}: {resp}" for a, resp in other_responses)
                else:
                    # Anonymize: replace names with generic "Respondent N" labels
                    prior = "\n".join(
                        f"Respondent {i + 1}: {resp}"
                        for i, (_, resp) in enumerate(other_responses)
                    )
                context = f"""
Here are answers from other respondents:
{prior}

You may revise or reaffirm your answer.
"""
            else:
                # First round: no prior context
                context = ""

            return f"""
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

        async def call_agent(agent_id):
            prompt = build_prompt_for_agent(agent_id)
            runner = agent_runners[agent_id]
            response = await asyncio.to_thread(runner, prompt)
            if not isinstance(response, str):
                response = ""
            else:
                response = response.strip()
            return agent_id, response

        # Run all agents in parallel within each round
        results = await asyncio.gather(*[call_agent(agent_id) for agent_id in agents])
        round_answers = {agent_id: response for agent_id, response in results}

        # Record all agents' answers for this round
        history.append(round_answers)

    return history
