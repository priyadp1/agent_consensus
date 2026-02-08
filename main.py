import json
import os
import argparse
import asyncio
import re
import yaml

from model import run_model
from multiagent import agent_talk
from filter_questions_globalqa import valid_question


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
    option_block = "\n".join(
        f"({letters[i]}) {opt}" for i, opt in enumerate(options)
    )
    allowed = ", ".join(letters)

    return f"""
You are answering a subjective public opinion survey question.

Question:
{example["question"]}

Answer options:
{option_block}

Instructions:
- Choose exactly ONE option by its letter ({allowed})
- Explain your reasoning briefly
- End your response with a final line in this format:

ANSWER: <LETTER>

where <LETTER> is one of: {allowed}

Please ensure the final answer line appears exactly as shown, with no additional text after it.
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


def single_agent(config):
    data_path = config["data"]["path"]
    limit = config["defaults"]["limit"]
    results_root = config["experiment"]["results_root"]

    results_dir = os.path.join(results_root, f"single_agent_{limit}")
    os.makedirs(results_dir, exist_ok=True)

    completed = get_completed(results_dir)
    used = 0

    for example in stream_jsonL(data_path):
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

        output = {
            "question": example["question"],
            "options": example["options"],
            "answer": parse_answer(raw, len(example["options"])),
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


async def multi_agent(config):
    agent_models = config["agents"]
    data_path = config["data"]["path"]
    limit = config["defaults"]["limit"]
    max_rounds = config["defaults"]["max_rounds"]
    results_root = config["experiment"]["results_root"]

    agents = list(agent_models.values())
    num_agents = len(agents)

    results_dir = os.path.join(
        results_root,
        f"agents_{num_agents}_questions_{limit}"
    )
    os.makedirs(results_dir, exist_ok=True)

    completed = get_completed(results_dir)

    agent_runners = {
        model_name: (lambda p, m=model_name: run_model(p, model_name=m))
        for model_name in agent_models.values()
    }

    used = 0

    for example in stream_jsonL(data_path):
        if not valid_question(example):
            continue

        if used in completed:
            print(f"[SKIP] Multi-agent question {used}")
            used += 1
            continue

        history = await agent_talk(
            agents=agents,
            agent_runners=agent_runners,
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
            "agent_models": agent_models,
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
    config_dir = "configs"
    config_files = sorted(
        f for f in os.listdir(config_dir)
        if f.endswith(".yaml") or f.endswith(".yml")
    )

    if not config_files:
        raise RuntimeError("No config files found")

    for cfg in config_files:
        config_path = os.path.join(config_dir, cfg)
        config = load_config(config_path)
        num_agents = len(config["agents"])

        print(f"\nRunning experiment: {config['experiment']['name']}")
        print(f"Agents: {num_agents}")
        print(f"Data: {config['data']['path']}")
        print(f"Results: {config['experiment']['results_root']}")

        if num_agents == 1:
            single_agent(config)
        else:
            asyncio.run(multi_agent(config))