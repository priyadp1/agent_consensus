import json
import os
import argparse
import asyncio
import re

from model import run_model, MODEL_NAME
from multiagent import agent_talk
from filter_questions import valid_question

DATA_PATH = "data/jsonl/train.jsonl"


def stream_jsonL(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def build_prompt(example):
    options = example["options"]
    option_block = "\n".join(
        [f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)]
    )

    return f"""
You are answering a subjective public opinion survey question.

Question:
{example["question"]}

Answer options:
{option_block}

Instructions:
- Choose ONE option.
- Respond ONLY in the format:
  Final answer: <LETTER>
- Do NOT add explanations or extra text.

Final answer:
"""


ANSWER_RE = re.compile(r"Final answer:\s*\(?([A-Z])\)?")


def parse_answer(text, num_options):
    match = ANSWER_RE.search(text)
    if not match:
        return "INVALID"

    letter = match.group(1)
    idx = ord(letter) - ord("A")

    if 0 <= idx < num_options:
        return letter

    return "INVALID"


def single_agent(limit):
    results_dir = f"results/{MODEL_NAME}/single_agent_{limit}"
    os.makedirs(results_dir, exist_ok=True)

    used = 0

    for example in stream_jsonL(DATA_PATH):
        if not valid_question(example):
            continue

        prompt = build_prompt(example)
        raw = run_model(prompt)
        answer = parse_answer(raw, len(example["options"]))

        output = {
            "question": example["question"],
            "options": example["options"],
            "answer": answer,
            "raw_output": raw.strip()
        }

        with open(os.path.join(results_dir, f"q_{used}.json"), "w") as f:
            json.dump(output, f, indent=2)

        used += 1
        if used >= limit:
            break


async def multi_agent(num_agents, limit, max_rounds):
    results_dir = f"results/{MODEL_NAME}/agents_{num_agents}_questions_{limit}"
    os.makedirs(results_dir, exist_ok=True)

    agents = list(range(num_agents))
    used = 0

    for example in stream_jsonL(DATA_PATH):
        if not valid_question(example):
            continue

        history = await agent_talk(
            agents=agents,
            question=example["question"],
            options=example["options"],
            selections=example["selections"],
            run_model=run_model,
            max_rounds=max_rounds
        )

        parsed_rounds = []

        for round_data in history:
            parsed = {}
            for agent_id, raw in round_data.items():
                parsed[agent_id] = {
                    "answer": parse_answer(raw, len(example["options"])),
                    "raw_output": raw.strip()
                }
            parsed_rounds.append(parsed)

        output = {
            "question": example["question"],
            "options": example["options"],
            "rounds": parsed_rounds
        }

        with open(os.path.join(results_dir, f"q_{used}.json"), "w") as f:
            json.dump(output, f, indent=2)

        used += 1
        if used >= limit:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=1)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)

    args = parser.parse_args()

    if args.agents == 1:
        single_agent(args.limit)
    else:
        asyncio.run(multi_agent(args.agents, args.limit, args.rounds))