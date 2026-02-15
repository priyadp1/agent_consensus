import json
import os
import asyncio
import re

from model import run_model
from multiagent_rotate import agent_talk
from main import load_config, get_completed, parse_answer


def rank_models(config_path):
    if config_path == "configs/gpt-4.1-fam-rotate.yaml":
        biggest = "gpt-4.1"
        middle = "gpt-4.1-mini"
        smallest = "gpt-4.1-nano"
    elif config_path == "configs/llama-fam-rotate.yaml":
        biggest = "Meta-Llama-3.1-405B-Instruct"
        middle = "Llama-3.3-70B-Instruct"
        smallest = "Meta-Llama-3.1-8B-Instruct"
    elif config_path == "configs/random_models-rotate.yaml":
        biggest = "grok-3"
        middle = "DeepSeek-R1"
        smallest = "Llama-3.3-70B-Instruct"
    else:
        raise ValueError(f"Unknown config path: {config_path}")
    return biggest, middle, smallest


def rotate_R1_answers(R1, biggest_agent, middle_agent, smallest_agent, num_options):
    ori_biggest = R1[biggest_agent]["raw_output"]
    ori_middle = R1[middle_agent]["raw_output"]
    ori_smallest = R1[smallest_agent]["raw_output"]

    biggest_answer = parse_answer(ori_biggest, num_options)
    middle_answer = parse_answer(ori_middle, num_options)
    smallest_answer = parse_answer(ori_smallest, num_options)

    if "INVALID" in {biggest_answer, middle_answer, smallest_answer}:
        return None

    if len({biggest_answer, middle_answer, smallest_answer}) == 1:
        return None

    def swap_answers(raw, new_answer):
        return re.sub(r"ANSWER:\s*[A-Z]", f"ANSWER: {new_answer}", raw)

    R1_rotated = json.loads(json.dumps(R1))

    R1_rotated[middle_agent]["raw_output"] = swap_answers(ori_middle, smallest_answer)
    R1_rotated[middle_agent]["answer"] = smallest_answer

    R1_rotated[biggest_agent]["raw_output"] = swap_answers(ori_biggest, middle_answer)
    R1_rotated[biggest_agent]["answer"] = middle_answer

    R1_rotated[smallest_agent]["raw_output"] = swap_answers(ori_smallest, biggest_answer)
    R1_rotated[smallest_agent]["answer"] = biggest_answer

    return R1_rotated


def parse_round(round_data, num_options):
    parsed = {}
    for agent_id, raw in round_data.items():
        if not isinstance(raw, str) or not raw.strip():
            parsed[agent_id] = {
                "answer": "INVALID",
                "raw_output": "",
                "model_failed": True
            }
        else:
            parsed[agent_id] = {
                "answer": parse_answer(raw, num_options),
                "raw_output": raw.strip(),
                "model_failed": False
            }
    return parsed


async def run_rotated_experiment(config_path, old_results_dir):
    config = load_config(config_path)
    biggest_model, middle_model, smallest_model = rank_models(config_path)

    agent_to_model = config["agents"]
    model_to_agent = {
        model_name: agent_label
        for agent_label, model_name in agent_to_model.items()
    }

    agents = list(agent_to_model.keys())

    agent_runners = {}
    for agent_label, model_name in agent_to_model.items():
        agent_runners[agent_label] = (
            lambda p, m=model_name: run_model(p, model_name=m)
        )

    new_dir = old_results_dir + "_rotated"
    os.makedirs(new_dir, exist_ok=True)

    completed = get_completed(new_dir)

    files = sorted(
        (
            f for f in os.listdir(old_results_dir)
            if f.startswith("q_") and f.endswith(".json")
        ),
        key=lambda x: int(x[2:-5])
    )

    for fname in files:
        qid = int(fname[2:-5])

        if qid in completed:
            print(f"[SKIP] Rotated question {qid}")
            continue

        with open(os.path.join(old_results_dir, fname)) as f:
            old = json.load(f)

        R1 = old["rounds"][0]
        num_options = len(old["options"])

        biggest_agent = model_to_agent[biggest_model]
        middle_agent = model_to_agent[middle_model]
        smallest_agent = model_to_agent[smallest_model]

        R1_rotated = rotate_R1_answers(
            R1,
            biggest_agent,
            middle_agent,
            smallest_agent,
            num_options
        )

        rotation_applied = R1_rotated is not None
        if not rotation_applied:
            R1_rotated = R1

        history = [{
        agent: R1_rotated[agent]["raw_output"]
        for agent in agents
                    }]


        new_rounds = await agent_talk(
            agents=agents,
            agent_runners=agent_runners,
            question=old["question"],
            options=old["options"],
            selections=None,
            max_rounds=3,
            history=history
        )

        parsed_rounds = [
            parse_round(r, num_options) for r in new_rounds
        ]

        output = {
            "question": old["question"],
            "options": old["options"],
            "agent_models": agent_to_model,
            "rounds": parsed_rounds,
            "rotation": rotation_applied
        }

        with open(os.path.join(new_dir, f"q_{qid}.json"), "w") as f:
            json.dump(output, f, indent=2)

        print(f"[SAVED] Rotated convo {qid} -> {new_dir}")


if __name__ == "__main__":
    config_dir = "configs"

    config_files = sorted(
        f for f in os.listdir(config_dir)
        if f.endswith(".yaml") or f.endswith(".yml")
    )

    if not config_files:
        raise RuntimeError("No config files found in configs/")

    for cfg in config_files:
        if "rotate" not in cfg:
            continue

        config_path = os.path.join(config_dir, cfg)
        config = load_config(config_path)
        results_root = config["experiment"]["results_root"]

        asyncio.run(
            run_rotated_experiment(config_path, old_results_dir=results_root)
        )