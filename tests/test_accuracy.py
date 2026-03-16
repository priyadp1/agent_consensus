import json
import os
from collections import defaultdict

RESULTS_DIR = "results/HLE/20_rounds/gpt-4.1-fam/agents_3_questions_591/named"
DATA_PATH = "data/jsonl/hle/test_processed.jsonl"
MODEL_ORDER = {
    "gpt-4.1-nano": 0,
    "gpt-4.1-mini": 1,
    "gpt-4.1":      2,
}

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
    Among pairs that disagreed in round 1, track who deferred to whom in later rounds:
      small -> large : smaller model adopted the larger model's answer
      large -> small : larger model adopted the smaller model's answer
      correct -> wrong : model with correct answer in R1 switched to wrong answer
      wrong -> correct : model with wrong answer in R1 switched to correct answer
    """
    initial     = defaultdict(int)  # how many R1 disagreements per pair
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
        model_to_label = build_model_to_label(agent_models)
        models = list(agent_models.values())
        r1     = rounds[0]
        r_rest = rounds[1:num_rounds]

        for m_small in models:
            for m_large in models:
                if m_small == m_large:
                    continue
                if m_small not in MODEL_ORDER or m_large not in MODEL_ORDER:
                    continue
                if MODEL_ORDER[m_small] >= MODEL_ORDER[m_large]:
                    continue  # only consider (smaller, larger) pairs

                k_small = round_key_for_model(r1, m_small, model_to_label)
                k_large = round_key_for_model(r1, m_large, model_to_label)
                a_small = get_answer(r1, k_small)
                a_large = get_answer(r1, k_large)

                if a_small == "INVALID" or a_large == "INVALID":
                    continue
                if a_small == a_large:
                    continue  # no disagreement to track

                key = f"{m_small} -> {m_large}"
                initial[key] += 1

                for r in r_rest:
                    rk_small = round_key_for_model(r, m_small, model_to_label)
                    rk_large = round_key_for_model(r, m_large, model_to_label)
                    cur_small = get_answer(r, rk_small)
                    cur_large = get_answer(r, rk_large)
                    if cur_small == a_large:
                        s2l[key] += 1
                        if a_large == correct_answer:
                            s2l_w2c[key] += 1
                        elif a_small == correct_answer:
                            s2l_c2w[key] += 1
                        break
                    if cur_large == a_small:
                        l2s[key] += 1
                        if a_small == correct_answer:
                            l2s_w2c[key] += 1
                        elif a_large == correct_answer:
                            l2s_c2w[key] += 1
                        break

    return initial, s2l, l2s, s2l_w2c, s2l_c2w, l2s_w2c, l2s_c2w




if __name__ == "__main__":
    data = load_results(RESULTS_DIR)
    ground_truth = load_data(DATA_PATH)
    num_rounds = detect_num_rounds(data)

    print(f"Rounds: {num_rounds}  |  Questions: {len(data)}\n")

    # ── Accuracy across rounds ─────────────────────────────────────────────────
    correct, total_valid = compute_accuracy_across_rounds(data, ground_truth, num_rounds)

    print("=== Accuracy Across Rounds ===")
    for i in range(num_rounds):
        total = total_valid[i]
        corr = correct[i]
        pct = corr / total * 100 if total else 0
        print(f"  Round {i+1}: {corr}/{total} ({pct:.1f}%)")

    # ── Pairwise accuracy ──────────────────────────────────────────────────────
    pairwise_totals, pairwise_correct = compute_pairwise_accuracy(data, ground_truth, num_rounds)

    print("\n=== Pairwise Accuracy (on disagreements) ===")
    for pair in sorted(pairwise_totals.keys()):
        print(f"{pair}")
        for i in range(num_rounds):
            total = pairwise_totals[pair][i]
            correct = pairwise_correct[pair][i]
            pct = correct / total * 100 if total else 0
            print(f"  Round {i+1}: {correct}/{total} ({pct:.1f}%)")

    # ── Deference accuracy ─────────────────────────────────────────────────────
    initial, s2l, l2s, s2l_w2c, s2l_c2w, l2s_w2c, l2s_c2w = compute_deference_accuracy(data, ground_truth, num_rounds)

    print("\n=== Deference Accuracy ===")
    for key in sorted(initial.keys()):
        total = initial[key]
        print(f"{key}  (R1 disagreements: {total})")
        if total:
            s = s2l[key]; l = l2s[key]
            print(f"  small -> large   : {s}/{total} ({s/total*100:.1f}%)")
            print(f"    wrong -> correct : {s2l_w2c[key]}/{s} ({s2l_w2c[key]/s*100:.1f}%)" if s else f"    wrong -> correct : 0/0")
            print(f"    correct -> wrong : {s2l_c2w[key]}/{s} ({s2l_c2w[key]/s*100:.1f}%)" if s else f"    correct -> wrong : 0/0")
            print(f"  large -> small   : {l}/{total} ({l/total*100:.1f}%)")
            print(f"    wrong -> correct : {l2s_w2c[key]}/{l} ({l2s_w2c[key]/l*100:.1f}%)" if l else f"    wrong -> correct : 0/0")
            print(f"    correct -> wrong : {l2s_c2w[key]}/{l} ({l2s_c2w[key]/l*100:.1f}%)" if l else f"    correct -> wrong : 0/0")