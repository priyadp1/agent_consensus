import json
import os
from collections import defaultdict

RESULTS_DIR = "results/HLE/20_rounds/gpt-4.1-fam/agents_3_questions_2089/named"
DATA_PATH = "data/jsonl/hle/test_processed.jsonl"

def load_results(results_dir):
    files = sorted(
        f for f in os.listdir(results_dir)
        if f.startswith("q_") and f.endswith(".json")
    )
    data = []
    for fname in files:
        idx = int(fname[2:-5])
        with open(os.path.join(results_dir, fname)) as fh:
            item = json.load(fh)
            item["_question_idx"] = idx
            data.append(item)
    return data

def load_data(DATA_PATH):
    data = []
    with open(DATA_PATH) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def detect_num_rounds(data):
    # Find the max number of rounds seen across all questions
    return max((len(item.get("rounds", [])) for item in data), default=0)

def compute_accuracy_across_rounds(data, ground_truth, num_rounds):
    #For each round, compute per-model accuracy against the ground truth answer.
    #ground_truth is a list of dicts loaded from test_processed.jsonl.
    #Each result item has _question_idx linking it to the correct ground truth entry.
    model_correct = defaultdict(lambda: [0] * num_rounds)
    model_total   = defaultdict(lambda: [0] * num_rounds)

    for item in data:
        idx = item.get("_question_idx")
        if idx is None or idx >= len(ground_truth):
            continue
        correct_answer = ground_truth[idx]["answer"] 

        rounds = item.get("rounds", [])
        for i in range(min(len(rounds), num_rounds)):
            for model, agent_data in rounds[i].items():
                ans = agent_data.get("answer")
                if ans in (None, "INVALID") or agent_data.get("model_failed"):
                    continue
                model_total[model][i] += 1
                if ans == correct_answer:
                    model_correct[model][i] += 1

    return model_correct, model_total


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = load_results(RESULTS_DIR)
    ground_truth = load_data(DATA_PATH)
    num_rounds = detect_num_rounds(data)

    model_correct, model_total = compute_accuracy_across_rounds(data, ground_truth, num_rounds)

    print(f"Rounds: {num_rounds}  |  Questions: {len(data)}\n")
    for model in sorted(model_total.keys()):
        print(f"{model}")
        for i in range(num_rounds):
            total = model_total[model][i]
            correct = model_correct[model][i]
            pct = correct / total * 100 if total else 0
            print(f"  Round {i+1}: {correct}/{total} ({pct:.1f}%)")