import json
import os
from collections import defaultdict

RESULTS_DIR = "results/gpt-4.1-family/agents_3_questions_2089"

MODEL_ORDER = {
    "gpt-4.1-nano": 0,
    "gpt-4.1-mini": 1,
    "gpt-4.1": 2,
}

def load_results(results_dir):
    files = sorted(
        f for f in os.listdir(results_dir)
        if f.startswith("q_") and f.endswith(".json")
    )
    data = []
    for f in files:
        with open(os.path.join(results_dir, f)) as fh:
            data.append(json.load(fh))
    return data

def get_answer(round_data, agent):
    return round_data.get(agent, {}).get("answer", "INVALID")

def analyze(data):
    initial_disagreements = defaultdict(int)
    small_to_large = defaultdict(int)
    large_to_small = defaultdict(int)

    for item in data:
        rounds = item["rounds"]
        agents = list(item["agent_models"].keys())

        if len(rounds) < 2:
            continue

        round1 = rounds[0]

        
        for a in agents:
            for b in agents:
                if a == b:
                    continue

                model_a = item["agent_models"][a]
                model_b = item["agent_models"][b]

                if MODEL_ORDER[model_a] >= MODEL_ORDER[model_b]:
                    continue 

                ans_a_r1 = get_answer(round1, a)
                ans_b_r1 = get_answer(round1, b)

                if ans_a_r1 == "INVALID" or ans_b_r1 == "INVALID":
                    continue

                if ans_a_r1 == ans_b_r1:
                    continue

                key = f"{model_a} -> {model_b}"
                initial_disagreements[key] += 1

                for r in rounds[1:]:
                    ans_a = get_answer(r, a)
                    ans_b = get_answer(r, b)

                    if ans_a == ans_b_r1:
                        small_to_large[key] += 1
                        break

                    if ans_b == ans_a_r1:
                        large_to_small[key] += 1
                        break

    return initial_disagreements, small_to_large, large_to_small

def print_results(initial, s2l, l2s):
    print("\n=== ANALYSIS of GPT-4.1 Family ===\n")
    for key in sorted(initial.keys()):
        total = initial[key]
        s = s2l[key]
        l = l2s[key]

        print(f"{key}")
        print(f"  Initial disagreements: {total}")
        print(f"  Small -> Large: {s} ({s/total:.2%})")
        print(f"  Large -> Small: {l} ({l/total:.2%})")
        print("-" * 40)

if __name__ == "__main__":
    data = load_results(RESULTS_DIR)
    initial, s2l, l2s = analyze(data)
    print_results(initial, s2l, l2s)