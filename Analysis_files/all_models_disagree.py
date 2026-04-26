import json
import os
from collections import defaultdict
from datetime import datetime, timezone

RESULTS_ROOT = "results"
OUTPUT_ROOT = "new_analysis_outputs"

MODEL_ORDER = {
    # Random models 
    "Llama-3.3-70B-Instruct": 0,
    "DeepSeek-R1": 1,
    "grok-3": 2,
    # GPT-4.1 family 
    "gpt-4.1-nano": 0,
    "gpt-4.1-mini": 1,
    "gpt-4.1": 2,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_leaf_dirs(root):
    #Walk root and yield every directory that contains q_*.json files.
    for dirpath, dirnames, filenames in os.walk(root):
        if any(f.startswith("q_") and f.endswith(".json") for f in filenames):
            yield dirpath


def results_to_output(results_dir):
    #Convert a results/ path to the mirrored new_analysis_outputs/ path.
    rel = os.path.relpath(results_dir, RESULTS_ROOT)
    return os.path.join(OUTPUT_ROOT, rel)


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


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


# ── Question-level metrics ────────────────────────────────────────────────────

def round_disagree(data, round_idx):
    rounds = data.get("rounds", [])
    if len(rounds) <= round_idx:
        return None
    answers = []
    for agent_data in rounds[round_idx].values():
        ans = agent_data.get("answer")
        if ans in (None, "INVALID"):
            return None
        answers.append(ans)
    if len(answers) < 2:
        return False
    return len(set(answers)) > 1


def compute_metrics(results_dir, output_dir):
    files = [f for f in os.listdir(results_dir) if f.startswith("q_") and f.endswith(".json")]
    if not files:
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(results_dir, files[0])) as f:
        sample = json.load(f)
    num_rounds = len(sample.get("rounds", []))

    total_valid = [0] * num_rounds
    disagreements = [0] * num_rounds

    for fname in files:
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        for i in range(num_rounds):
            result = round_disagree(data, i)
            if result is None:
                continue
            total_valid[i] += 1
            if result:
                disagreements[i] += 1

    percentages = [
        (100 * disagreements[i] / total_valid[i]) if total_valid[i] > 0 else 0
        for i in range(num_rounds)
    ]

    metrics = {
        "model": results_dir,
        "rounds": {
            f"round_{i + 1}": {
                "valid_questions": total_valid[i],
                "disagreement_count": disagreements[i],
                "disagreement_percentage": round(percentages[i], 2),
            }
            for i in range(num_rounds)
        }
    }

    out_path = os.path.join(output_dir, "interagent_disagree.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  [metrics]     → {out_path}")
    for i in range(num_rounds):
        print(f"Round {i+1}: {disagreements[i]}/{total_valid[i]} ({percentages[i]:.2f}%)")


# ── Pair-level disagreement ───────────────────────────────────────────────────

def build_model_to_label(agent_models):
    return {model: label for label, model in agent_models.items()}


def round_key_for_model(round_data, model, model_to_label):
    if model in round_data:
        return model
    label = model_to_label.get(model)
    if label and label in round_data:
        return label
    return model


def pair_disagree(data):
    counts = defaultdict(lambda: defaultdict(lambda: {"disagreements": 0, "total": 0}))
    for item in data:
        rounds = item["rounds"]
        agent_models = item.get("agent_models", {})
        agent_items = list(agent_models.items())  # [(label, model), ...]
        for r_idx, r in enumerate(rounds):
            for i, (label1, m1) in enumerate(agent_items):
                for label2, m2 in agent_items[i + 1:]:
                    key = f"{m1} vs {m2}"
                    # Prefer model name as round key (fam format); fall back to agent
                    # label (null format, where all models share the same name and
                    # round data is keyed by "Agent 1" / "Agent 2" / ... instead)
                    k1 = m1 if m1 in r else label1
                    k2 = m2 if m2 in r else label2
                    a1 = get_answer(r, k1)
                    a2 = get_answer(r, k2)
                    if a1 == "INVALID" or a2 == "INVALID":
                        continue
                    counts[key][r_idx]["total"] += 1
                    if a1 != a2:
                        counts[key][r_idx]["disagreements"] += 1
    return counts


def save_per_round(counts, path):
    out = {}
    for pair in sorted(counts.keys()):
        out[pair] = {}
        for r_idx in sorted(counts[pair].keys()):
            d = counts[pair][r_idx]["disagreements"]
            t = counts[pair][r_idx]["total"]
            out[pair][f"round_{r_idx+1}"] = {
                "disagreements": d,
                "total": t,
                "percentage": round(d / t * 100, 2) if t > 0 else None,
            }
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f" [pair-level]  -> {path}")


# ── Directional deference ─────────────────────────────────────────────────────

def analyze_conditioned(original_data):
    # Counts are accumulated over all round-to-round transitions.
    # For each transition (n-1 -> n), the denominator is pairs that disagreed
    # in round n-1; deference is switching to the *other model's round-(n-1)* answer.
    initial = defaultdict(int)
    s2l = defaultdict(int)
    l2s = defaultdict(int)

    for item in original_data:
        agent_models = item.get("agent_models", {})
        agent_items = list(agent_models.items())  # [(label, model), ...]
        rounds = item["rounds"]

        for i, (label_small, m_small) in enumerate(agent_items):
            for label_large, m_large in agent_items[i + 1:]:
                if MODEL_ORDER.get(m_small, -1) >= MODEL_ORDER.get(m_large, -1):
                    continue

                key = f"{m_small} -> {m_large}"

                for n in range(1, len(rounds)):
                    r_prev = rounds[n - 1]
                    r_curr = rounds[n]

                    k_small_prev = m_small if m_small in r_prev else label_small
                    k_large_prev = m_large if m_large in r_prev else label_large
                    a_prev = get_answer(r_prev, k_small_prev)
                    b_prev = get_answer(r_prev, k_large_prev)

                    # Only count transitions where the pair disagreed in round n-1
                    if a_prev == "INVALID" or b_prev == "INVALID" or a_prev == b_prev:
                        continue

                    initial[key] += 1

                    k_small_curr = m_small if m_small in r_curr else label_small
                    k_large_curr = m_large if m_large in r_curr else label_large
                    a_curr = get_answer(r_curr, k_small_curr)
                    b_curr = get_answer(r_curr, k_large_curr)

                    if a_curr == "INVALID" or b_curr == "INVALID":
                        continue

                    # Small deferred to large: adopted large's round-(n-1) answer
                    if a_curr == b_prev:
                        s2l[key] += 1
                    # Large deferred to small: adopted small's round-(n-1) answer
                    if b_curr == a_prev:
                        l2s[key] += 1

    return initial, s2l, l2s


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
    print(f"[directional]  -> {path}")


def model_deference_each_round(data):
    # Same logic as analyze_conditioned but keyed by round transition index.
    # Returns: {pair_key: {transition_idx: {"initial": int, "s2l": int, "l2s": int}}}
    counts = defaultdict(lambda: defaultdict(lambda: {"initial": 0, "s2l": 0, "l2s": 0}))

    for item in data:
        agent_models = item.get("agent_models", {})
        agent_items = list(agent_models.items())  # [(label, model), ...]
        rounds = item["rounds"]

        for i, (label_small, m_small) in enumerate(agent_items):
            for label_large, m_large in agent_items[i + 1:]:
                if MODEL_ORDER.get(m_small, -1) >= MODEL_ORDER.get(m_large, -1):
                    continue

                key = f"{m_small} -> {m_large}"

                for n in range(1, len(rounds)):
                    r_prev = rounds[n - 1]
                    r_curr = rounds[n]

                    k_small_prev = m_small if m_small in r_prev else label_small
                    k_large_prev = m_large if m_large in r_prev else label_large
                    a_prev = get_answer(r_prev, k_small_prev)
                    b_prev = get_answer(r_prev, k_large_prev)

                    if a_prev == "INVALID" or b_prev == "INVALID" or a_prev == b_prev:
                        continue

                    counts[key][n - 1]["initial"] += 1

                    k_small_curr = m_small if m_small in r_curr else label_small
                    k_large_curr = m_large if m_large in r_curr else label_large
                    a_curr = get_answer(r_curr, k_small_curr)
                    b_curr = get_answer(r_curr, k_large_curr)

                    if a_curr == "INVALID" or b_curr == "INVALID":
                        continue

                    if a_curr == b_prev:
                        counts[key][n - 1]["s2l"] += 1
                    if b_curr == a_prev:
                        counts[key][n - 1]["l2s"] += 1

    return counts


def save_deference_each_round(counts, path):
    out = {}
    for key in sorted(counts.keys()):
        out[key] = {}
        for t_idx in sorted(counts[key].keys()):
            d = counts[key][t_idx]
            total = d["initial"]
            s = d["s2l"]
            l = d["l2s"]
            out[key][f"round_{t_idx + 1}_to_{t_idx + 2}"] = {
                "initial_disagreements": total,
                "small_to_large": s,
                "large_to_small": l,
                "small_to_large_pct": round(s / total * 100, 2) if total > 0 else None,
                "large_to_small_pct": round(l / total * 100, 2) if total > 0 else None,
            }
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[deference/round] -> {path}")




# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    leaf_dirs = sorted(find_leaf_dirs(RESULTS_ROOT))

    if not leaf_dirs:
        print(f"No result directories found under '{RESULTS_ROOT}'.")
    else:
        print(f"Found {len(leaf_dirs)} result director{'y' if len(leaf_dirs) == 1 else 'ies'}.\n")

    for results_dir in leaf_dirs:
        output_dir = results_to_output(results_dir)
        print(f"Processing: {results_dir}")

        data = load_results(results_dir)
        if not data:
            print("  [SKIP] No question files found.\n")
            continue

        print(f"  Loaded {len(data)} questions")

        compute_metrics(results_dir, output_dir)

        counts = pair_disagree(data)
        save_per_round(counts, os.path.join(output_dir, "per_round_disagreement.json"))

        initial, s2l, l2s = analyze_conditioned(data)
        save_directional(initial, s2l, l2s, os.path.join(output_dir, "directional.json"))

        round_counts = model_deference_each_round(data)
        save_deference_each_round(round_counts, os.path.join(output_dir, "deference_each_round.json"))
