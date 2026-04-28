import json
import os
from collections import defaultdict

RESULTS_ROOT = "results/HLE"
OUTPUT_ROOT = "new_analysis_outputs/HLE"
DATA_PATH = "data/jsonl/hle/test_processed.jsonl"
MODEL_ORDER = {
    # Random models (current: Phi < Llama < Mistral)
    "Phi-4": 0,
    "Llama-4-Maverick-17B-128E-Instruct-FP8": 1,
    "Mistral-Large-3": 2,
    #Mistral Models
    "Ministral-3B": 0,
    "mistral-medium-2505": 1,
    "Mistral-Large-3": 2,
    # GPT-4.1 family
    "gpt-4.1-nano": 0,
    "gpt-4.1-mini": 1,
    "gpt-4.1": 2,
}


def find_leaf_dirs(root):
    # Walk root and yield every directory that contains q_*.json files.
    for dirpath, dirnames, filenames in os.walk(root):
        if any(f.startswith("q_") and f.endswith(".json") for f in filenames):
            yield dirpath


def results_to_output(results_dir):
    # Convert a results/HLE/ path to the mirrored new_analysis_outputs/HLE/ path.
    rel = os.path.relpath(results_dir, RESULTS_ROOT)
    return os.path.join(OUTPUT_ROOT, rel)


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


def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_answer(round_data, model):
    entry = round_data.get(model)
    if not entry:
        return "INVALID"
    if entry.get("model_failed"):
        return "INVALID"
    return entry.get("answer", "INVALID")


def detect_num_rounds(data):
    # Find the max number of rounds seen across all questions
    return max((len(item.get("rounds", [])) for item in data), default=0)


def compute_accuracy_across_rounds(data, ground_truth, num_rounds):
    # For each round, a question counts as correct only if ALL models got the right answer.
    # A question is valid if all models returned non-INVALID answers.
    total_valid = [0] * num_rounds
    correct     = [0] * num_rounds

    for item in data:
        idx = item.get("_question_idx")
        if idx is None or idx >= len(ground_truth):
            continue
        correct_answer = ground_truth[idx]["answer"]

        rounds = item.get("rounds", [])
        for i in range(min(len(rounds), num_rounds)):
            answers = []
            for agent_data in rounds[i].values():
                ans = agent_data.get("answer")
                if ans in (None, "INVALID") or agent_data.get("model_failed"):
                    answers = None
                    break
                answers.append(ans)
            if answers is None:
                continue
            total_valid[i] += 1
            if all(ans == correct_answer for ans in answers):
                correct[i] += 1

    return correct, total_valid

def build_model_to_label(agent_models):
    """Map model names to their agent labels, e.g. 'gpt-4.1-nano' -> 'Agent 1'."""
    return {model: label for label, model in agent_models.items()}


def round_key_for_model(round_data, model, model_to_label):
    """Return the key to use for looking up a model in round_data.
    Handles both formats: rounds keyed by model name or by agent label."""
    if model in round_data:
        return model
    label = model_to_label.get(model)
    if label and label in round_data:
        return label
    return model  # fallback, will result in INVALID via get_answer


def compute_pairwise_accuracy(data, ground_truth, num_rounds):
    """
    For each round and each model pair, count questions where both models got the correct answer.
    Returns dict[pair_key] -> (total_valid_list, both_correct_list) per round.
    """
    pairwise_totals = defaultdict(lambda: [0] * num_rounds)
    pairwise_correct = defaultdict(lambda: [0] * num_rounds)

    for item in data:
        rounds = item.get("rounds", [])
        agent_models = item.get("agent_models", {})
        model_to_label = build_model_to_label(agent_models)
        models = list(agent_models.values())
        correct_answer = ground_truth[item["_question_idx"]]["answer"]

        for i in range(min(len(rounds), num_rounds)):
            for j, m1 in enumerate(models):
                for m2 in models[j + 1:]:
                    k1 = round_key_for_model(rounds[i], m1, model_to_label)
                    k2 = round_key_for_model(rounds[i], m2, model_to_label)
                    a1 = get_answer(rounds[i], k1)
                    a2 = get_answer(rounds[i], k2)
                    if a1 == "INVALID" or a2 == "INVALID":
                        continue

                    key = f"{m1} vs {m2}"
                    pairwise_totals[key][i] += 1
                    if a1 == correct_answer and a2 == correct_answer:
                        pairwise_correct[key][i] += 1

    return pairwise_totals, pairwise_correct


# ── Directional deference ─────────────────────────────────────────────────────

def compute_deference_accuracy(data, ground_truth, num_rounds):
    """
    For each consecutive round transition (n-1 -> n), among pairs that disagreed in
    round n-1, track who deferred to whom in round n:
      small -> large : smaller model adopted the larger model's round-(n-1) answer
      large -> small : larger model adopted the smaller model's round-(n-1) answer
        correct -> wrong : deferring model had the correct answer in round n-1
        wrong -> correct : deferring model had the wrong answer in round n-1
    """
    initial     = defaultdict(int)  # disagreement transitions per pair
    s2l         = defaultdict(int)  # small deferred to large
    l2s         = defaultdict(int)  # large deferred to small
    s2l_w2c     = defaultdict(int)  # small deferred to large, large was correct (wrong->correct)
    s2l_c2w     = defaultdict(int)  # small deferred to large, small was correct (correct->wrong)
    l2s_w2c     = defaultdict(int)  # large deferred to small, small was correct (wrong->correct)
    l2s_c2w     = defaultdict(int)  # large deferred to small, large was correct (correct->wrong)

    for item in data:
        correct_answer = ground_truth[item["_question_idx"]]["answer"]
        rounds = item.get("rounds", [])
        if len(rounds) < 2:
            continue

        agent_models = item.get("agent_models", {})
        agent_items = list(agent_models.items())  # [(label, model), ...]

        for i, (label_a, m_a) in enumerate(agent_items):
            for label_b, m_b in agent_items[i + 1:]:
                if MODEL_ORDER.get(m_a, -1) == -1 or MODEL_ORDER.get(m_b, -1) == -1:
                    continue
                ord_a = MODEL_ORDER[m_a]
                ord_b = MODEL_ORDER[m_b]
                if ord_a == ord_b:
                    continue  # same tier, no meaningful smaller/larger distinction
                # Ensure label_small/m_small always refers to the actually smaller model
                if ord_a < ord_b:
                    label_small, m_small, label_large, m_large = label_a, m_a, label_b, m_b
                else:
                    label_small, m_small, label_large, m_large = label_b, m_b, label_a, m_a

                key = f"{m_small} ({label_small}) -> {m_large} ({label_large})" if m_small == m_large else f"{m_small} -> {m_large}"

                for n in range(1, min(len(rounds), num_rounds)):
                    r_prev = rounds[n - 1]
                    r_curr = rounds[n]

                    k_small_prev = m_small if m_small in r_prev else label_small
                    k_large_prev = m_large if m_large in r_prev else label_large
                    a_small = get_answer(r_prev, k_small_prev)
                    a_large = get_answer(r_prev, k_large_prev)

                    if a_small == "INVALID" or a_large == "INVALID" or a_small == a_large:
                        continue  # no valid disagreement in round n-1

                    initial[key] += 1

                    k_small_curr = m_small if m_small in r_curr else label_small
                    k_large_curr = m_large if m_large in r_curr else label_large
                    cur_small = get_answer(r_curr, k_small_curr)
                    cur_large = get_answer(r_curr, k_large_curr)

                    if cur_small == "INVALID" or cur_large == "INVALID":
                        continue

                    if cur_small == a_large:
                        s2l[key] += 1
                        if a_large == correct_answer:
                            s2l_w2c[key] += 1
                        elif a_small == correct_answer:
                            s2l_c2w[key] += 1

                    if cur_large == a_small:
                        l2s[key] += 1
                        if a_small == correct_answer:
                            l2s_w2c[key] += 1
                        elif a_large == correct_answer:
                            l2s_c2w[key] += 1

    return initial, s2l, l2s, s2l_w2c, s2l_c2w, l2s_w2c, l2s_c2w


def process_dir(results_dir, ground_truth):
    data = load_results(results_dir)
    if not data:
        print("  [SKIP] No question files found.\n")
        return

    print(f"  Loaded {len(data)} questions")
    num_rounds = detect_num_rounds(data)
    output_dir = results_to_output(results_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── Accuracy across rounds ─────────────────────────────────────────────────
    correct, total_valid = compute_accuracy_across_rounds(data, ground_truth, num_rounds)

    print("  === Accuracy Across Rounds ===")
    metrics = {"rounds": num_rounds, "questions": len(data), "rounds_data": {}}
    for i in range(num_rounds):
        total = total_valid[i]
        corr = correct[i]
        pct = corr / total * 100 if total else 0
        print(f"  Round {i+1}: {corr}/{total} ({pct:.1f}%)")
        metrics["rounds_data"][f"round_{i+1}"] = {
            "correct": corr,
            "total": total,
            "accuracy_pct": round(pct, 1),
        }

    out_path = os.path.join(output_dir, "hle_accuracy.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [saved] -> {out_path}")

    # ── Pairwise accuracy ──────────────────────────────────────────────────────
    pairwise_totals, pairwise_correct = compute_pairwise_accuracy(data, ground_truth, num_rounds)

    print("\n  === Pairwise Accuracy (on disagreements) ===")
    pairwise_metrics = {}
    for pair in sorted(pairwise_totals.keys()):
        print(f"  {pair}")
        pairwise_metrics[pair] = {}
        for i in range(num_rounds):
            total = pairwise_totals[pair][i]
            correct = pairwise_correct[pair][i]
            pct = correct / total * 100 if total else 0
            print(f"    Round {i+1}: {correct}/{total} ({pct:.1f}%)")
            pairwise_metrics[pair][f"round_{i+1}"] = {
                "correct": correct,
                "total": total,
                "accuracy_pct": round(pct, 1),
            }

    out_path = os.path.join(output_dir, "hle_pairwise_accuracy.json")
    with open(out_path, "w") as f:
        json.dump(pairwise_metrics, f, indent=2)
    print(f"  [saved] -> {out_path}")

    # ── Deference accuracy ─────────────────────────────────────────────────────
    initial, s2l, l2s, s2l_w2c, s2l_c2w, l2s_w2c, l2s_c2w = compute_deference_accuracy(data, ground_truth, num_rounds)

    print("\n  === Deference Accuracy ===")
    deference_metrics = {}
    for key in sorted(initial.keys()):
        total = initial[key]
        print(f"  {key}  (R1 disagreements: {total})")
        deference_metrics[key] = {"initial_disagreements": total}
        if total:
            s = s2l[key]; l = l2s[key]
            print(f"    small -> large   : {s}/{total} ({s/total*100:.1f}%)")
            print(f"      wrong -> correct : {s2l_w2c[key]}/{s} ({s2l_w2c[key]/s*100:.1f}%)" if s else f"      wrong -> correct : 0/0")
            print(f"      correct -> wrong : {s2l_c2w[key]}/{s} ({s2l_c2w[key]/s*100:.1f}%)" if s else f"      correct -> wrong : 0/0")
            print(f"    large -> small   : {l}/{total} ({l/total*100:.1f}%)")
            print(f"      wrong -> correct : {l2s_w2c[key]}/{l} ({l2s_w2c[key]/l*100:.1f}%)" if l else f"      wrong -> correct : 0/0")
            print(f"      correct -> wrong : {l2s_c2w[key]}/{l} ({l2s_c2w[key]/l*100:.1f}%)" if l else f"      correct -> wrong : 0/0")
            deference_metrics[key].update({
                "small_to_large": s, "s2l_pct": round(s/total*100, 1),
                "s2l_wrong_to_correct": s2l_w2c[key], "s2l_w2c_pct": round(s2l_w2c[key]/s*100, 1) if s else 0,
                "s2l_correct_to_wrong": s2l_c2w[key], "s2l_c2w_pct": round(s2l_c2w[key]/s*100, 1) if s else 0,
                "large_to_small": l, "l2s_pct": round(l/total*100, 1),
                "l2s_wrong_to_correct": l2s_w2c[key], "l2s_w2c_pct": round(l2s_w2c[key]/l*100, 1) if l else 0,
                "l2s_correct_to_wrong": l2s_c2w[key], "l2s_c2w_pct": round(l2s_c2w[key]/l*100, 1) if l else 0,
            })

    out_path = os.path.join(output_dir, "hle_deference_accuracy.json")
    with open(out_path, "w") as f:
        json.dump(deference_metrics, f, indent=2)
    print(f"  [saved] -> {out_path}\n")


if __name__ == "__main__":
    leaf_dirs = sorted(find_leaf_dirs(RESULTS_ROOT))

    if not leaf_dirs:
        print(f"No result directories found under '{RESULTS_ROOT}'.")
    else:
        print(f"Found {len(leaf_dirs)} result director{'y' if len(leaf_dirs) == 1 else 'ies'}.\n")

    ground_truth = load_data(DATA_PATH)

    for results_dir in leaf_dirs:
        print(f"Processing: {results_dir}")
        process_dir(results_dir, ground_truth)