import json
import os
import sys
import argparse
import asyncio
import re
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import run_model
from agents.multiagent import agent_talk
from preprocessing.filter_questions import valid_question

# Maximum number of questions processed concurrently
CONCURRENCY = 3


def load_config(path):
    # Load a YAML config file and return it as a dict
    with open(path, "r") as f:
        return yaml.safe_load(f)


def stream_jsonL(path):
    # Lazily yield one parsed JSON object per line from a JSONL file
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def get_completed(results_dir):
    # Return the set of question indices already saved as q_<idx>.json in results_dir
    completed = set()
    if not os.path.exists(results_dir):
        return completed

    for fname in os.listdir(results_dir):
        if fname.startswith("q_") and fname.endswith(".json"):
            try:
                # Extract the integer index from the filename
                completed.add(int(fname[2:-5]))
            except ValueError:
                pass

    return completed


def build_prompt(example):
    # Format a survey question and its options into a structured prompt string
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
    # Build a regex that matches "ANSWER: <X>" where X is a valid option letter
    letters = "".join(chr(65 + i) for i in range(num_options))
    return re.compile(rf"ANSWER:\s*([{letters}])\s*$")


def parse_answer(text, num_options):
    # Extract the chosen letter from a model response; returns "INVALID" if not found
    if not isinstance(text, str):
        return "INVALID"

    text = text.strip()
    if not text:
        return "INVALID"

    match = build_answer_regex(num_options).search(text)
    if not match:
        return "INVALID"

    return match.group(1)


async def single_agent(config):
    # Run a single-agent experiment: one model answers each question independently
    data_path = config["data"]["path"]
    options_key = config["data"].get("options_key", "options")
    limit = config["defaults"]["limit"]
    results_root = config["experiment"]["results_root"]

    results_dir = os.path.join(results_root, f"single_agent_{limit}")
    os.makedirs(results_dir, exist_ok=True)

    completed = get_completed(results_dir)

    # Collect all valid questions up to limit, preserving their index
    all_valid = []
    used = 0
    for example in stream_jsonL(data_path):
        if not valid_question(example, key=options_key):
            continue
        all_valid.append((used, example))
        used += 1
        if used >= limit:
            break

    # Print skips upfront, then process pending questions in parallel
    pending = []
    for qid, example in all_valid:
        if qid in completed:
            print(f"[SKIP] Single-agent question {qid}")
        else:
            pending.append((qid, example))

    sem = asyncio.Semaphore(CONCURRENCY)

    async def process_one(qid, example):
        async with sem:
            prompt = build_prompt(example)
            raw = await asyncio.to_thread(run_model, prompt)

            if not isinstance(raw, str):
                raw = ""

            output = {
                "question": example["question"],
                "options": example["options"],
                "answer": parse_answer(raw, len(example["options"])),
                "raw_output": raw.strip(),
                "model_failed": raw == ""
            }

            output_path = os.path.join(results_dir, f"q_{qid}.json")
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

            print(f"[SAVED] Single-agent convo {qid} -> {output_path}")

    await asyncio.gather(*[process_one(qid, ex) for qid, ex in pending])


async def multi_agent(config):
    # Run both the named and anonymous variants of the multi-agent experiment.
    # Results are saved to separate subdirectories (named/ and anonymous/) so
    # both can coexist without any manual config changes.
    agent_models = config["agents"]
    data_path = config["data"]["path"]
    options_key = config["data"].get("options_key", "options")
    limit = config["defaults"]["limit"]
    max_rounds = config["defaults"]["max_rounds"]
    results_root = config["experiment"]["results_root"]

    # Use model names as agent identifiers
    agents = list(agent_models.values())

    # Create a callable runner for each agent model, capturing the model name via default arg
    agent_runners = {
        model_name: (lambda p, m=model_name: run_model(p, model_name=m))
        for model_name in agent_models.values()
    }

    # Run once with agents seeing real model names, once with anonymized "Respondent N" labels
    for show_agent_names in [True, False]:
        # variant label is used as a subdirectory name: results_root/named/ or results_root/anonymous/
        variant = "named" if show_agent_names else "anonymous"
        results_dir = os.path.join(results_root, variant)
        os.makedirs(results_dir, exist_ok=True)

        completed = get_completed(results_dir)

        # Collect all valid questions up to limit, preserving their index
        all_valid = []
        used = 0
        for example in stream_jsonL(data_path):
            if not valid_question(example, key=options_key):
                continue
            all_valid.append((used, example))
            used += 1
            if used >= limit:
                break

        print(f"\n  [{variant.upper()}] Starting variant...")

        # Print skips upfront, then process pending questions in parallel
        pending = []
        for qid, example in all_valid:
            if qid in completed:
                print(f"[SKIP] Multi-agent ({variant}) question {qid}")
            else:
                pending.append((qid, example))

        sem = asyncio.Semaphore(CONCURRENCY)

        async def process_one(qid, example, show_agent_names=show_agent_names, results_dir=results_dir):
            async with sem:
                history = await agent_talk(
                    agents=agents,
                    agent_runners=agent_runners,
                    question=example["question"],
                    options=example["options"],
                    selections=example.get("selections"),
                    max_rounds=max_rounds,
                    show_agent_names=show_agent_names
                )

                # Parse each agent's raw response per round into a structured format
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

                output_path = os.path.join(results_dir, f"q_{qid}.json")
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=2)

                print(
                    f"[SAVED] Multi-agent ({variant}) convo {qid} "
                    f"(agents={len(agents)}, rounds={max_rounds}) -> {output_path}"
                )

        await asyncio.gather(*[process_one(qid, ex) for qid, ex in pending])


if __name__ == "__main__":
    config_dir = "baseline_configs_20"
    # Collect all config files whose names end with "fam.yaml" or "fam.yml"
    config_files = sorted(
        f for f in os.listdir(config_dir)
        if f.endswith("fam.yaml") or f.endswith("fam.yml")
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

        # Dispatch to single- or multi-agent runner based on agent count
        if num_agents == 1:
            asyncio.run(single_agent(config))
        else:
            asyncio.run(multi_agent(config))
