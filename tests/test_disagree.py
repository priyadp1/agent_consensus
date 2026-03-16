import json
import os
from collections import defaultdict

RESULTS_DIR = "results/HLE/20_rounds/gpt-4.1-fam/agents_3_questions_591/named"
MODEL_ORDER = {
    "gpt-4.1-nano": 0,
    "gpt-4.1-mini": 1,
    "gpt-4.1":      2,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(results_dir):
    files = sorted(
        f for f in os.listdir(results_dir)
        if f.startswith("q_") and f.endswith(".json")
    )
    data = []
    for fname in files:
        with open(os.path.join(results_dir, fname)) as fh:
            data.append(json.load(fh))
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


# ── Question-level disagreement (per round) ───────────────────────────────────

def compute_true_disagreement(data, num_rounds):
    """
    For each round, count how many questions had at least one pair of agents disagree.
    """
    total_valid  = [0] * num_rounds
    disagreements = [0] * num_rounds

    for item in data:
        rounds = item.get("rounds", [])
        for i in range(min(len(rounds), num_rounds)):
            answers = []
            valid = True
            for agent_data in rounds[i].values():
                ans = agent_data.get("answer")
                if ans in (None, "INVALID"):
                    valid = False
                    break
                answers.append(ans)

            if not valid or len(answers) < 2:
                continue

            total_valid[i] += 1
            if len(set(answers)) > 1:
                disagreements[i] += 1

    return total_valid, disagreements


# ── Pair-level disagreement (per round) ───────────────────────────────────────

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


def compute_pairwise_disagreement(data, num_rounds):
    """
    For each round, count disagreements across all agent pairs.
    Returns dict[pair_key] -> (total_valid_list, disagreement_list) per round.
    """
    pair_totals = defaultdict(lambda: [0] * num_rounds)
    pair_disagree = defaultdict(lambda: [0] * num_rounds)

    for item in data:
        rounds = item.get("rounds", [])
        agent_models = item.get("agent_models", {})
        model_to_label = build_model_to_label(agent_models)
        models = list(agent_models.values())

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
                    pair_totals[key][i] += 1
                    if a1 != a2:
                        pair_disagree[key][i] += 1

    return pair_totals, pair_disagree


# ── Directional deference ─────────────────────────────────────────────────────

def compute_deference(data, num_rounds):
    """
    Among pairs that disagreed in round 1, track who deferred to whom in later rounds:
      small → large : smaller model adopted the larger model's answer
      large → small : larger model adopted the smaller model's answer
    """
    initial = defaultdict(int)  # how many R1 disagreements per pair
    s2l     = defaultdict(int)  # small deferred to large
    l2s     = defaultdict(int)  # large deferred to small

    for item in data:
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
                        break
                    if cur_large == a_small:
                        l2s[key] += 1
                        break

    return initial, s2l, l2s


# ── Printers ──────────────────────────────────────────────────────────────────

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)



def print_question_level(total_valid, disagreements, num_rounds):
    print_header("QUESTION-LEVEL DISAGREEMENT (≥1 pair disagreed)")
    for i in range(num_rounds):
        tv = total_valid[i]
        d  = disagreements[i]
        pct = (100 * d / tv) if tv > 0 else 0.0
        print(f"  Round {i+1}: {d}/{tv} questions disagreed ({pct:.1f}%)")


def print_pair_level(pair_totals, pair_disagree, num_rounds):
    print_header("PAIR-LEVEL DISAGREEMENT (per model pair)")
    for key in sorted(pair_totals.keys()):
        print(f"\n  {key}")
        for i in range(num_rounds):
            tv = pair_totals[key][i]
            d = pair_disagree[key][i]
            pct = (100 * d / tv) if tv > 0 else 0.0
            print(f"Round {i+1}: {d}/{tv} disagreed ({pct:.1f}%)")



def print_deference(initial, s2l, l2s):
    print_header("DIRECTIONAL DEFERENCE")
    if not initial:
        print("  No R1 disagreements found yet.")
        return

    for key in sorted(initial.keys()):
        total = initial[key]
        s     = s2l[key]
        l     = l2s[key]
        neither = total - s - l
        print(f"\n  {key}")
        print(f"    Small -> Large    : {s:4d}  ({100*s/total:.1f}%)")
        print(f"    Large -> Small    : {l:4d}  ({100*l/total:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        raise SystemExit(1)

    data = load_results(RESULTS_DIR)
    num_rounds = detect_num_rounds(data)

    print(f"\nResults dir : {RESULTS_DIR}")
    print(f"Questions   : {len(data)} loaded so far")
    print(f"Max rounds  : {num_rounds} seen in data")

    if not data:
        print("No results yet — run the experiment first.")
        raise SystemExit(0)

    total_valid, disagreements = compute_true_disagreement(data, num_rounds)
    print_question_level(total_valid, disagreements, num_rounds)

    pair_totals, pair_disagree = compute_pairwise_disagreement(data, num_rounds)
    print_pair_level(pair_totals, pair_disagree, num_rounds)

    initial, s2l, l2s = compute_deference(data, num_rounds)
    print_deference(initial, s2l, l2s)

    print()
