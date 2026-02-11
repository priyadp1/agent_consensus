import json
import os
from collections import defaultdict

ORIGINAL_RESULTS_DIR = (
    "results/GlobalOpinionsQA/agent_names/random_models/"
    "agents_3_questions_2556"
)

ROTATED_RESULTS_DIR = (
    "results/GlobalOpinionsQA/agent_names/random_models/"
    "agents_3_questions_2556_rotated"
)

OUTPUT_BASE = "analysis_outputs/GlobalOpinionsQA/agent_names/random_models"

ROUND_JSON_NORMAL = os.path.join(
    OUTPUT_BASE, "per_round_disagreement_NORMAL.json"
)

ROUND_JSON_ROTATED = os.path.join(
    OUTPUT_BASE, "per_round_disagreement_ROTATED.json"
)

DIRECTIONAL_JSON_NORMAL = os.path.join(
    OUTPUT_BASE, "random_models_NORMAL_directional.json"
)

DIRECTIONAL_JSON_ROTATED = os.path.join(
    OUTPUT_BASE, "random_models_ROTATED_directional.json"
)

MODEL_ORDER = {
    "Llama-3.3-70B-Instruct": 0,
    "DeepSeek-R1": 1,
    "grok-3": 2,
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


def get_answer(round_data, model):
    entry = round_data.get(model)
    if not entry:
        return "INVALID"
    if entry.get("model_failed"):
        return "INVALID"
    return entry.get("answer", "INVALID")


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def analyze_per_round(data):
    per_round = defaultdict(lambda: {"disagreements": 0, "total_pairs": 0})

    for item in data:
        rounds = item["rounds"]
        models = list(item["agent_models"].values())

        for r_idx, r in enumerate(rounds):
            for i, m1 in enumerate(models):
                for m2 in models[i + 1:]:
                    per_round[r_idx]["total_pairs"] += 1

                    a1 = get_answer(r, m1)
                    a2 = get_answer(r, m2)

                    if a1 == "INVALID" or a2 == "INVALID":
                        continue

                    if a1 != a2:
                        per_round[r_idx]["disagreements"] += 1

    return per_round


def print_per_round(label, per_round):
    print(f"\n=== ROUND-AGNOSTIC DISAGREEMENT ({label}) ===\n")
    for r in sorted(per_round.keys()):
        d = per_round[r]["disagreements"]
        t = per_round[r]["total_pairs"]
        pct = d / t if t > 0 else 0
        print(f"Round {r+1}: {d}/{t} disagreements ({pct:.2%})")


def save_per_round(per_round, path):
    out = {}
    for r in sorted(per_round.keys()):
        d = per_round[r]["disagreements"]
        t = per_round[r]["total_pairs"]
        out[f"round_{r+1}"] = {
            "disagreements": d,
            "total_pairs": t,
            "percentage": round(d / t * 100, 2) if t > 0 else None,
        }

    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved → {path}")


def analyze_conditioned(original_data, comparison_data):
    initial = defaultdict(int)
    s2l = defaultdict(int)
    l2s = defaultdict(int)

    for orig, comp in zip(original_data, comparison_data):
        models = list(orig["agent_models"].values())

        r1_orig = orig["rounds"][0]
        r_rest = comp["rounds"][1:]

        for m_small in models:
            for m_large in models:
                if m_small == m_large:
                    continue

                if MODEL_ORDER[m_small] >= MODEL_ORDER[m_large]:
                    continue

                a1 = get_answer(r1_orig, m_small)
                b1 = get_answer(r1_orig, m_large)

                if a1 == "INVALID" or b1 == "INVALID":
                    continue
                if a1 == b1:
                    continue

                key = f"{m_small} -> {m_large}"
                initial[key] += 1

                for r in r_rest:
                    if get_answer(r, m_small) == b1:
                        s2l[key] += 1
                        break
                    if get_answer(r, m_large) == a1:
                        l2s[key] += 1
                        break

    return initial, s2l, l2s


def print_directional(label, initial, s2l, l2s):
    print(f"\n=== CAUSAL DIRECTIONAL DEFERENCE ({label}) ===\n")
    for key in sorted(initial.keys()):
        total = initial[key]
        s = s2l[key]
        l = l2s[key]
        print(key)
        print(f"  Initial disagreements: {total}")
        print(f"  Small → Large: {s} ({s/total:.2%})")
        print(f"  Large → Small: {l} ({l/total:.2%})")
        print("-" * 40)


def save_directional(initial, s2l, l2s, path):
    out = {}
    for key in sorted(initial.keys()):
        total = initial[key]
        s = s2l[key]
        l = l2s[key]

        out[key] = {
            "initial_disagreements": total,
            "small_to_large": s,
            "large_to_small": l,
            "small_to_large_pct": round(s / total * 100, 2),
            "large_to_small_pct": round(l / total * 100, 2),
        }

    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved → {path}")


if __name__ == "__main__":
    print("\n=== RANDOM MODELS: NORMAL vs ROTATED ANALYSIS ===\n")

    orig_data = load_results(ORIGINAL_RESULTS_DIR)
    rot_data = load_results(ROTATED_RESULTS_DIR)

    print(f"Loaded {len(orig_data)} normal questions")
    print(f"Loaded {len(rot_data)} rotated questions")

    per_round_norm = analyze_per_round(orig_data)
    print_per_round("NORMAL", per_round_norm)
    save_per_round(per_round_norm, ROUND_JSON_NORMAL)

    per_round_rot = analyze_per_round(rot_data)
    print_per_round("ROTATED", per_round_rot)
    save_per_round(per_round_rot, ROUND_JSON_ROTATED)

    init_norm, s2l_norm, l2s_norm = analyze_conditioned(orig_data, orig_data)
    print_directional("NORMAL", init_norm, s2l_norm, l2s_norm)
    save_directional(init_norm, s2l_norm, l2s_norm, DIRECTIONAL_JSON_NORMAL)

    init_rot, s2l_rot, l2s_rot = analyze_conditioned(orig_data, rot_data)
    print_directional("ROTATED", init_rot, s2l_rot, l2s_rot)
    save_directional(init_rot, s2l_rot, l2s_rot, DIRECTIONAL_JSON_ROTATED)
