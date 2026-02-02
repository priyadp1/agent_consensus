import json
import os
from collections import defaultdict

RESULTS_DIR = "results/random_models/agents_3_questions_2089"

MODEL_ORDER = {
    "Llama-3.3-70B-Instruct": 0,
    "DeepSeek-R1": 1,
    "grok-3": 2,
}

OUTPUT_JSON = "analysis_outputs/random_models/random_models_directional.json"


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
    round1_agreements = defaultdict(int)
    round1_disagreements = defaultdict(int)

    models = list(MODEL_ORDER.keys())

    for m1 in models:
        for m2 in models:
            if m1 == m2:
                continue
            if MODEL_ORDER[m1] < MODEL_ORDER[m2]:
                key = f"{m1} -> {m2}"
                initial_disagreements[key] = 0
                small_to_large[key] = 0
                large_to_small[key] = 0
                round1_agreements[key] = 0
                round1_disagreements[key] = 0

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

                model_a = item["agent_models"].get(a)
                model_b = item["agent_models"].get(b)

                if model_a not in MODEL_ORDER or model_b not in MODEL_ORDER:
                    continue

                if MODEL_ORDER[model_a] >= MODEL_ORDER[model_b]:
                    continue

                ans_a_r1 = get_answer(round1, a)
                ans_b_r1 = get_answer(round1, b)

                if ans_a_r1 == "INVALID" or ans_b_r1 == "INVALID":
                    continue

                key = f"{model_a} -> {model_b}"

                if ans_a_r1 == ans_b_r1:
                    round1_agreements[key] += 1
                    continue
                else:
                    round1_disagreements[key] += 1
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

    return (
        initial_disagreements,
        small_to_large,
        large_to_small,
        round1_agreements,
        round1_disagreements,
    )


def save_results_json(initial, s2l, l2s, agree, disagree):
    results = {}

    for key in sorted(initial.keys()):
        total = initial[key]
        s = s2l[key]
        l = l2s[key]
        a = agree[key]
        d = disagree[key]

        results[key] = {
            "round1_agreements": a,
            "round1_disagreements": d,
            "initial_disagreements": total,
            "small_to_large": s,
            "large_to_small": l,
            "small_to_large_pct": round((s / total) * 100, 2) if total > 0 else None,
            "large_to_small_pct": round((l / total) * 100, 2) if total > 0 else None,
        }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {OUTPUT_JSON}")


def print_results(initial, s2l, l2s, agree, disagree):
    print("\n=== ANALYSIS OF RANDOM MODELS (DeepSeek / Grok / Llama) ===\n")

    for key in sorted(initial.keys()):
        total = initial[key]
        s = s2l[key]
        l = l2s[key]
        a = agree[key]
        d = disagree[key]

        print(f"{key}")
        print(f"  Round-1 agreements: {a}")
        print(f"  Round-1 disagreements: {d}")

        if total == 0:
            print("  Deference: N/A (no initial disagreement)")
        else:
            print(f"  Small -> Large: {s} ({s/total:.2%})")
            print(f"  Large -> Small: {l} ({l/total:.2%})")

        print("-" * 50)


if __name__ == "__main__":
    data = load_results(RESULTS_DIR)
    initial, s2l, l2s, agree, disagree = analyze(data)

    print_results(initial, s2l, l2s, agree, disagree)
    save_results_json(initial, s2l, l2s, agree, disagree)
