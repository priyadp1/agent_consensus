import json
import os
import argparse
import asyncio
import re

from model import run_model 
from multiagent import agent_talk
from filter_questions import valid_question

AGENT_MODELS = {
    "Agent 1": "gpt-4.1-nano",
    "Agent 2": "gpt-4.1-mini",
    "Agent 3": "gpt-4.1"
}

DATA_PATH = "data/jsonl/train.jsonl"


def stream_jsonL(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def get_completed(results_dir):
    completed = set()
    if not os.path.exists(results_dir):
        return completed
    for fname in os.listdir(results_dir):
        if fname.startswith("q_") and fname.endswith(".json"):
            try:
                completed.add(int(fname[2:-5]))
            except ValueError:
                pass
    return completed


def build_prompt(example):
    options = example["options"]
    letters = [chr(65 + i) for i in range(len(options))]
    option_block = "\n".join(f"({letters[i]}) {opt}" for i, opt in enumerate(options))
    allowed = ", ".join(letters)
    return f"""
You are answering a subjective public opinion survey question.

Question:
{example["question"]}

Answer options:
{option_block}

INSTRUCTIONS (MUST FOLLOW EXACTLY):
- Choose exactly ONE option by its letter ({allowed}).
- Explain your reasoning.
- The FINAL LINE of your response MUST be exactly in this format:

ANSWER: <LETTER>

where <LETTER> is one of: {allowed}

RULES:
- Do NOT write "Final answer".
- Do NOT include option text.
- Do NOT include parentheses.
- Do NOT include any text after the ANSWER line.
- Any response not following this format will be marked INVALID.
"""


def build_answer_regex(num_options):
    letters = "".join(chr(65 + i) for i in range(num_options))
    return re.compile(rf"ANSWER:\s*([{letters}])\s*$")


def parse_answer(text, num_options):
    if not isinstance(text, str):
        return "INVALID"
    text = text.strip()
    if not text:
        return "INVALID"
    match = build_answer_regex(num_options).search(text)
    if not match:
        return "INVALID"
    return match.group(1)


def single_agent(limit):
    results_dir = f"results/gpt-4.1-nano/single_agent_{limit}"
    os.makedirs(results_dir, exist_ok=True)

    completed = get_completed(results_dir)
    used = 0

    for example in stream_jsonL(DATA_PATH):
        if not valid_question(example):
            continue

        if used in completed:
            print(f"[SKIP] Single-agent question {used}")
            used += 1
            continue

        prompt = build_prompt(example)
        raw = run_model(prompt)

        if not isinstance(raw, str):
            raw = ""

        answer = parse_answer(raw, len(example["options"]))

        output = {
            "question": example["question"],
            "options": example["options"],
            "answer": answer,
            "raw_output": raw.strip(),
            "model_failed": raw == ""
        }

        output_path = os.path.join(results_dir, f"q_{used}.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[SAVED] Single-agent convo {used} -> {output_path}")

        used += 1
        if used >= limit:
            break


async def multi_agent(num_agents, limit, max_rounds):
    results_dir = f"results/mixed_models/agents_{num_agents}_questions_{limit}"
    os.makedirs(results_dir, exist_ok=True)

    completed = get_completed(results_dir)
    agents = list(AGENT_MODELS.keys())
    agent_runners = {
        agent_id: (lambda p, m=model: run_model(p, model_name=m))
        for agent_id, model in AGENT_MODELS.items()
        }
    used = 0

    for example in stream_jsonL(DATA_PATH):
        if not valid_question(example):
            continue

        if used in completed:
            print(f"[SKIP] Multi-agent question {used}")
            used += 1
            continue

        history = await agent_talk(
            agents=agents,
            agent_runners= agent_runners,
            question=example["question"],
            options=example["options"],
            selections=example["selections"],
            max_rounds=max_rounds
        )

        parsed_rounds = []

        for round_data in history:
            parsed = {}
            for agent_id, raw in round_data.items():
                if not isinstance(raw, str):
                    raw = ""
                parsed[agent_id] = {
                    "answer": parse_answer(raw, len(example["options"])),
                    "raw_output": raw.strip(),
                    "model_failed": raw == ""
                }
            parsed_rounds.append(parsed)

        output = {
            "question": example["question"],
            "options": example["options"],
            "agent_models" : AGENT_MODELS,
            "rounds": parsed_rounds
        }

        output_path = os.path.join(results_dir, f"q_{used}.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(
            f"[SAVED] Multi-agent convo {used} "
            f"(agents={num_agents}, rounds={max_rounds}) -> {output_path}"
        )

        used += 1
        if used >= limit:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)

    args = parser.parse_args()

    if args.agents == 1:
        single_agent(args.limit)
    else:
        asyncio.run(multi_agent(args.agents, args.limit, args.rounds))