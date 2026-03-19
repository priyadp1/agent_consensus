import json
import os
import sys
import asyncio
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import run_model
from agents.multiagent import agent_talk
from main import load_config, get_completed, parse_answer


def rank_models(agent_to_model):
    # Return (biggest, middle, smallest) model names by detecting the model family.
    models = set(agent_to_model.values())

    gpt41_family = {"gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"}
    llama_family = {"Meta-Llama-3.1-405B-Instruct", "Llama-3.3-70B-Instruct", "Meta-Llama-3.1-8B-Instruct"}
    random_family = {"grok-3", "DeepSeek-R1", "Llama-3.3-70B-Instruct"}

    if models == gpt41_family:
        return "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"
    elif models == llama_family:
        return "Meta-Llama-3.1-405B-Instruct", "Llama-3.3-70B-Instruct", "Meta-Llama-3.1-8B-Instruct"
    elif models == random_family:
        return "grok-3", "DeepSeek-R1", "Llama-3.3-70B-Instruct"
    else:
        raise ValueError(f"Unknown model family: {models}")


def rotate_R1_answers(R1, biggest_key, middle_key, smallest_key, num_options):
    # Cyclically rotate round-1 answers among the three agents:
    # biggest gets middle's answer, middle gets smallest's, smallest gets biggest's.
    # Returns None if any answer is INVALID or all three already agree.
    ori_biggest  = R1[biggest_key]["raw_output"]
    ori_middle   = R1[middle_key]["raw_output"]
    ori_smallest = R1[smallest_key]["raw_output"]

    biggest_answer  = parse_answer(ori_biggest,  num_options)
    middle_answer   = parse_answer(ori_middle,   num_options)
    smallest_answer = parse_answer(ori_smallest, num_options)

    if "INVALID" in {biggest_answer, middle_answer, smallest_answer}:
        return None

    if len({biggest_answer, middle_answer, smallest_answer}) == 1:
        return None

    def swap_answers(raw, new_answer):
        return re.sub(r"ANSWER:\s*[A-Z]", f"ANSWER: {new_answer}", raw)

    R1_rotated = json.loads(json.dumps(R1))
    R1_rotated[middle_key]["raw_output"]   = swap_answers(ori_middle,   smallest_answer)
    R1_rotated[middle_key]["answer"]       = smallest_answer
    R1_rotated[biggest_key]["raw_output"]  = swap_answers(ori_biggest,  middle_answer)
    R1_rotated[biggest_key]["answer"]      = middle_answer
    R1_rotated[smallest_key]["raw_output"] = swap_answers(ori_smallest, biggest_answer)
    R1_rotated[smallest_key]["answer"]     = biggest_answer

    return R1_rotated


def parse_round(round_data, num_options):
    # Convert raw agent responses into structured {answer, raw_output, model_failed} dicts.
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


async def run_rotated_experiment(config_path, results_root):
    # Run both named (see-names) and anonymous (no-see-names) rotated variants.
    # Named: agents and result keys are model name strings (results in named_rotated/).
    # Anonymous: agents and result keys are agent label keys like "agent_1" (results in anonymous_rotated/).
    config = load_config(config_path)
    agent_to_model = config["agents"]  # {"agent_1": "gpt-4.1", ...}

    biggest_model, middle_model, smallest_model = rank_models(agent_to_model)
    model_to_agent = {v: k for k, v in agent_to_model.items()}

    for use_agent_labels in [False, True]:
        variant = "anonymous" if use_agent_labels else "named"
        old_results_dir = os.path.join(results_root, variant)

        if use_agent_labels:
            # Anonymous: results keyed by agent label; map model names back to labels for rotation lookup
            agents = list(agent_to_model.keys())
            agent_runners = {
                label: (lambda p, m=model: run_model(p, model_name=m))
                for label, model in agent_to_model.items()
            }
            biggest_key  = model_to_agent[biggest_model]
            middle_key   = model_to_agent[middle_model]
            smallest_key = model_to_agent[smallest_model]
        else:
            # Named: results keyed by model name directly
            agents = list(agent_to_model.values())
            agent_runners = {
                model: (lambda p, m=model: run_model(p, model_name=m))
                for model in agents
            }
            biggest_key  = biggest_model
            middle_key   = middle_model
            smallest_key = smallest_model

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
                print(f"[SKIP] {variant} rotated question {qid}")
                continue

            with open(os.path.join(old_results_dir, fname)) as f:
                old = json.load(f)

            R1 = old["rounds"][0]
            num_options = len(old["options"])

            R1_rotated = rotate_R1_answers(
                R1, biggest_key, middle_key, smallest_key, num_options
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

            parsed_rounds = [parse_round(r, num_options) for r in new_rounds]

            output = {
                "question": old["question"],
                "options": old["options"],
                "agent_models": agent_to_model,
                "rounds": parsed_rounds,
                "rotation": rotation_applied
            }

            with open(os.path.join(new_dir, f"q_{qid}.json"), "w") as f:
                json.dump(output, f, indent=2)

            print(f"[SAVED] {variant} rotated convo {qid} -> {new_dir}")


if __name__ == "__main__":
    config_dir = "rotate_configs"
    config_files = sorted(
        f for f in os.listdir(config_dir)
        if f.endswith(".yaml") or f.endswith(".yml")
    )

    if not config_files:
        raise RuntimeError("No config files found in configs/")

    for cfg in config_files:
        config_path = os.path.join(config_dir, cfg)
        config = load_config(config_path)
        results_root = config["experiment"]["results_root"]

        asyncio.run(run_rotated_experiment(config_path, results_root=results_root))
