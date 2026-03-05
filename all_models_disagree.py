import json
import os
from collections import defaultdict
from datetime import datetime, timezone

# ── Results input directories ─────────────────────────────────────────────────

ORIGINAL_RANDOM_RESULTS_DIR_SEE_NAMES = (
    "results/GlobalOpinionsQA/agent_names/random_models/"
    "agents_3_questions_2556"
)
ROTATED_RANDOM_RESULTS_DIR_SEE_NAMES = (
    "results/GlobalOpinionsQA/agent_names/random_models/"
    "agents_3_questions_2556_rotated"
)
ORIGINAL_GPT41_RESULTS_DIR_SEE_NAMES = (
    "results/GlobalOpinionsQA/agent_names/gpt-4.1-fam/"
    "agents_3_questions_2556"
)
ROTATED_GPT41_RESULTS_DIR_SEE_NAMES = (
    "results/GlobalOpinionsQA/agent_names/gpt-4.1-fam/"
    "agents_3_questions_2556_rotated"
)

ORIGINAL_RANDOM_RESULTS_DIR_NO_SEE_NAMES = (
    "results/GlobalOpinionsQA/random_models/"
    "agents_3_questions_2089"
)
ROTATED_RANDOM_RESULTS_DIR_NO_SEE_NAMES = (
    "results/GlobalOpinionsQA/gpt-4.1-family/"
    "agents_3_questions_2089_rotated"
)

INDEP_PRMPT_GPT41_RESULTS_DIR_SEE_NAMES = (
    "results/GlobalOpinionsQA/agent_names/gpt-4.1-fam/"
    "agents_3_questions_2556/critical_independent/named"
)
INDEP_PRMPT_GPT41_RESULTS_DIR_ANON = (
    "results/GlobalOpinionsQA/agent_names/gpt-4.1-fam/"
    "agents_3_questions_2556/critical_independent/anonymous"
)
ADVERSARIAL_GPT41_RESULTS_DIR_ANON = (
    "results/GlobalOpinionsQA/agent_names/gpt-4.1-fam/"
    "agents_3_questions_2556/adversarial/anonymous"
)
ADVERSARIAL_GPT41_RESULTS_DIR_NAMED = (
    "results/GlobalOpinionsQA/agent_names/gpt-4.1-fam/"
    "agents_3_questions_2556/adversarial/named"
)

ORIGINAL_GPT41_RESULTS_DIR_NO_SEE_NAMES = (
    "results/GlobalOpinionsQA/gpt-4.1-family/"
    "agents_3_questions_2089_rotated"
)
ROTATED_GPT41_RESULTS_DIR_NO_SEE_NAMES = (
    "results/GlobalOpinionsQA/gpt-4.1-family/"
    "agents_3_questions_2089_rotated"
)
WITHOUT_REVISE_PROMPT_GPT_41_ADVERSARIAL_NAMED = (
    "results/GlobalOpinionsQA/agent_names/gpt-4.1-fam/"
    "without_revise/agents_3_questions_2556/adversarial/named"
)
WITHOUT_REVISE_PROMP_GPT_41_ADVERSARIAL_ANON = (
    "results/GlobalOpinionsQA/agent_names/gpt-4.1-fam/"
    "without_revise/agents_3_questions_2556/adversarial/anonymous"
)
WITHOUT_REVISE_PROMPT_GPT_41_CRITICAL_INDEPENDENT_NAMED = (
    "results/GlobalOpinionsQA/agent_names/gpt-4.1-fam/"
    "without_revise/agents_3_questions_2556/critical_independent/named"
)
WITHOUT_REVISE_PROMPT_GPT_41_CRITICAL_INDEPENDENT_ANON = (
    "results/GlobalOpinionsQA/agent_names/gpt-4.1-fam/"
    "without_revise/agents_3_questions_2556/critical_independent/anonymous"
)
TWENTY_ROUND_GPT41_RESULTS_DIR_SEE_NAMES = (
    "results/GlobalOpinionsQA/agent_names/20_rounds/gpt-4.1-fam/"
    "agents_3_questions_2556/named"
)
TWENTY_ROUND_GPT41_RESULTS_DIR_NO_SEE_NAMES = (
    "results/GlobalOpinionsQA/agent_names/20_rounds/gpt-4.1-fam/"
    "agents_3_questions_2556/anonymous"
)


# ── Output paths: Random models, see names ────────────────────────────────────
OUTPUT_BASE_RANDOM_SEE = "analysis_outputs/GlobalOpinionsQA/agent_names/random_models"

ROUND_JSON_RANDOM_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_RANDOM_SEE, "per_round_disagreement_NORMAL.json"
)
ROUND_JSON_RANDOM_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_RANDOM_SEE, "per_round_disagreement_ROTATED.json"
)
DIRECTIONAL_JSON_RANDOM_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_RANDOM_SEE, "random_models_NORMAL_directional.json"
)
DIRECTIONAL_JSON_RANDOM_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_RANDOM_SEE, "random_models_ROTATED_directional.json"
)

# ── Output paths: GPT-4.1, see names ─────────────────────────────────────────
OUTPUT_BASE_GPT41_SEE = "analysis_outputs/GlobalOpinionsQA/agent_names/gpt-4.1-fam"

ROUND_JSON_GPT41_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "per_round_disagreement_NORMAL.json"
)
ROUND_JSON_GPT41_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "per_round_disagreement_ROTATED.json"
)
DIRECTIONAL_JSON_GPT41_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "gpt4.1_family_NORMAL_directional.json"
)
DIRECTIONAL_JSON_GPT41_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "gpt4.1_family_ROTATED_directional.json"
)

ROUND_JSON_GPT41_SEE_INDEP = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "per_round_disagreement_INDEP.json"
)
DIRECTIONAL_JSON_GPT41_SEE_INDEP = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "gpt4.1_family_INDEP_directional.json"
)

ROUND_JSON_GPT41_SEE_INDEP_ANON = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "per_round_disagreement_INDEP_ANON.json"
)
DIRECTIONAL_JSON_GPT41_SEE_INDEP_ANON = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "gpt4.1_family_INDEP_ANON_directional.json"
)

ROUND_JSON_GPT41_SEE_ADVERSARIAL_ANON = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "per_round_disagreement_ADVERSARIAL_ANON.json"
)
DIRECTIONAL_JSON_GPT41_SEE_ADVERSARIAL_ANON = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "gpt4.1_family_ADVERSARIAL_ANON_directional.json"
)

ROUND_JSON_GPT41_SEE_ADVERSARIAL_NAMED = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "per_round_disagreement_ADVERSARIAL_NAMED.json"
)
DIRECTIONAL_JSON_GPT41_SEE_ADVERSARIAL_NAMED = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "gpt4.1_family_ADVERSARIAL_NAMED_directional.json"
)

# ── Output paths: GPT-4.1, see names, without_revise ─────────────────────────
OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE = "analysis_outputs/GlobalOpinionsQA/agent_names/gpt-4.1-fam/without_revise"

ROUND_JSON_WITHOUT_REVISE_GPT41_ADV_NAMED = os.path.join(
    OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE, "per_round_disagreement_ADVERSARIAL_NAMED.json"
)
DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_ADV_NAMED = os.path.join(
    OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE, "gpt4.1_family_ADVERSARIAL_NAMED_directional.json"
)

ROUND_JSON_WITHOUT_REVISE_GPT41_ADV_ANON = os.path.join(
    OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE, "per_round_disagreement_ADVERSARIAL_ANON.json"
)
DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_ADV_ANON = os.path.join(
    OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE, "gpt4.1_family_ADVERSARIAL_ANON_directional.json"
)

ROUND_JSON_WITHOUT_REVISE_GPT41_CRIT_INDEP_NAMED = os.path.join(
    OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE, "per_round_disagreement_CRITICAL_INDEPENDENT_NAMED.json"
)
DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_CRIT_INDEP_NAMED = os.path.join(
    OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE, "gpt4.1_family_CRITICAL_INDEPENDENT_NAMED_directional.json"
)

ROUND_JSON_WITHOUT_REVISE_GPT41_CRIT_INDEP_ANON = os.path.join(
    OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE, "per_round_disagreement_CRITICAL_INDEPENDENT_ANON.json"
)
DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_CRIT_INDEP_ANON = os.path.join(
    OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE, "gpt4.1_family_CRITICAL_INDEPENDENT_ANON_directional.json"
)

# ── Output paths: GPT-4.1, 20 rounds, see names ───────────────────────────────
OUTPUT_BASE_GPT41_TWENTY_ROUND_SEE = "analysis_outputs/GlobalOpinionsQA/agent_names/20_rounds/gpt-4.1-fam"

ROUND_JSON_TWENTY_ROUND_GPT41_SEE = os.path.join(
    OUTPUT_BASE_GPT41_TWENTY_ROUND_SEE, "per_round_disagreement.json"
)
DIRECTIONAL_JSON_TWENTY_ROUND_GPT41_SEE = os.path.join(
    OUTPUT_BASE_GPT41_TWENTY_ROUND_SEE, "gpt4.1_family_directional.json"
)

# ── Output paths: GPT-4.1, 20 rounds, no see names ───────────────────────────
OUTPUT_BASE_GPT41_TWENTY_ROUND_NO_SEE = "analysis_outputs/GlobalOpinionsQA/gpt-4.1-family/20_rounds"

ROUND_JSON_TWENTY_ROUND_GPT41_NO_SEE = os.path.join(
    OUTPUT_BASE_GPT41_TWENTY_ROUND_NO_SEE, "per_round_disagreement.json"
)
DIRECTIONAL_JSON_TWENTY_ROUND_GPT41_NO_SEE = os.path.join(
    OUTPUT_BASE_GPT41_TWENTY_ROUND_NO_SEE, "gpt4.1_family_directional.json"
)

# ── Output paths: Random models, no see names ─────────────────────────────────
OUTPUT_BASE_RANDOM_NO_SEE = "analysis_outputs/GlobalOpinionsQA/random_models"

ROUND_JSON_RANDOM_NO_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_RANDOM_NO_SEE, "per_round_disagreement_NORMAL.json"
)
ROUND_JSON_RANDOM_NO_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_RANDOM_NO_SEE, "per_round_disagreement_ROTATED.json"
)
DIRECTIONAL_JSON_RANDOM_NO_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_RANDOM_NO_SEE, "random_models_NORMAL_directional.json"
)
DIRECTIONAL_JSON_RANDOM_NO_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_RANDOM_NO_SEE, "random_models_ROTATED_directional.json"
)

# ── Output paths: GPT-4.1, no see names ──────────────────────────────────────
OUTPUT_BASE_GPT41_NO_SEE = "analysis_outputs/GlobalOpinionsQA/gpt-4.1-family"

ROUND_JSON_GPT41_NO_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_GPT41_NO_SEE, "per_round_disagreement_NORMAL.json"
)
ROUND_JSON_GPT41_NO_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_GPT41_NO_SEE, "per_round_disagreement_ROTATED.json"
)
DIRECTIONAL_JSON_GPT41_NO_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_GPT41_NO_SEE, "gpt4.1_family_NORMAL_directional.json"
)
DIRECTIONAL_JSON_GPT41_NO_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_GPT41_NO_SEE, "gpt4.1_family_ROTATED_directional.json"
)

MODEL_ORDER = {
    # Random models (smallest → largest)
    "Llama-3.3-70B-Instruct": 0,
    "DeepSeek-R1": 1,
    "grok-3": 2,
    # GPT-4.1 family (smallest → largest)
    "gpt-4.1-nano": 0,
    "gpt-4.1-mini": 1,
    "gpt-4.1": 2,
}

# ── question-level metrics (from check.py) ────────────────────────────────────

def round_disagree(data, round_idx):
    # Returns True if any two agents disagreed on this question in the given round,
    # False if all agreed, None if the round is missing or any answer is invalid.
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


def compute_metrics(results_dir, metrics_dir):
    # Read all q_*.json files in results_dir, count per-round disagreements at
    # the question level, and write interagent_disagree.json to metrics_dir.
    if not os.path.exists(results_dir):
        print(f"[SKIP] Results dir not found: {results_dir}")
        return

    files = [f for f in os.listdir(results_dir) if f.startswith("q_") and f.endswith(".json")]
    if not files:
        print(f"[SKIP] No question files found in: {results_dir}")
        return

    os.makedirs(metrics_dir, exist_ok=True)

    # Detect number of rounds from the first file
    with open(os.path.join(results_dir, files[0]), "r") as f:
        sample = json.load(f)
    num_rounds = len(sample.get("rounds", []))

    total_valid = [0] * num_rounds
    disagreements = [0] * num_rounds

    for fname in files:
        with open(os.path.join(results_dir, fname), "r") as f:
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
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = os.path.join(metrics_dir, "interagent_disagree.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[METRICS] {results_dir}")
    for i in range(num_rounds):
        print(f"  Round {i+1}: {disagreements[i]}/{total_valid[i]} ({percentages[i]:.2f}%)")
    print(f"  Saved → {out_path}")


# ── Shared helpers ────────────────────────────────────────────────────────────

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


# ── Pair-level disagreement ───────────────────────────────────────────────────
# Metric: for each round, count the number of (model_A, model_B) pairs that
# gave different answers, aggregated across all questions. This captures how
# often any two specific models disagree, independent of the third agent.

def pair_disagree(data):
    # Returns {pair_key: {r_idx: {"disagreements": N, "total": N}}}
    counts = defaultdict(lambda: defaultdict(lambda: {"disagreements": 0, "total": 0}))

    for item in data:
        rounds = item["rounds"]
        models = list(item["agent_models"].values())

        for r_idx, r in enumerate(rounds):
            for i, m1 in enumerate(models):
                for m2 in models[i + 1:]:
                    key = f"{m1} vs {m2}"
                    a1 = get_answer(r, m1)
                    a2 = get_answer(r, m2)

                    if a1 == "INVALID" or a2 == "INVALID":
                        continue

                    counts[key][r_idx]["total"] += 1
                    if a1 != a2:
                        counts[key][r_idx]["disagreements"] += 1

    return counts


def print_per_round(label, counts):
    print(f"\n=== PAIR-LEVEL DISAGREEMENT ({label}) ===\n")
    for pair in sorted(counts.keys()):
        print(f"  {pair}")
        for r_idx in sorted(counts[pair].keys()):
            d = counts[pair][r_idx]["disagreements"]
            t = counts[pair][r_idx]["total"]
            pct = d / t if t > 0 else 0
            print(f"    Round {r_idx+1}: {d}/{t} disagreed ({pct:.2%})")


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

    print(f"Saved → {path}")


def analyze_conditioned(original_data, comparison_data):
    # Given pairs of agents who disagreed in round 1, count who deferred to whom
    # in subsequent rounds: small→large means the lower-ranked model changed.
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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: compute question-level metrics (check.py logic) ───────────────
    # Each tuple is (results_dir, metrics_dir). metrics_dir mirrors results_dir
    # but under metrics/ instead of results/.
    print("\n=== COMPUTING QUESTION-LEVEL METRICS ===\n")

    metrics_runs = [
        (ORIGINAL_RANDOM_RESULTS_DIR_SEE_NAMES,
         "metrics/GlobalOpinionsQA/agent_names/random_models/agents_3_questions_2556"),
        (ROTATED_RANDOM_RESULTS_DIR_SEE_NAMES,
         "metrics/GlobalOpinionsQA/agent_names/random_models/agents_3_questions_2556_rotated"),
        (ORIGINAL_GPT41_RESULTS_DIR_SEE_NAMES,
         "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/agents_3_questions_2556"),
        (ROTATED_GPT41_RESULTS_DIR_SEE_NAMES,
         "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/agents_3_questions_2556_rotated"),
        (ORIGINAL_RANDOM_RESULTS_DIR_NO_SEE_NAMES,
         "metrics/GlobalOpinionsQA/random_models/agents_3_questions_2556"),
        (ROTATED_RANDOM_RESULTS_DIR_NO_SEE_NAMES,
         "metrics/GlobalOpinionsQA/random_models/agents_3_questions_2089_rotated"),
        (ORIGINAL_GPT41_RESULTS_DIR_NO_SEE_NAMES,
         "metrics/GlobalOpinionsQA/gpt-4.1-family/agents_3_questions_2089"),
        (ROTATED_GPT41_RESULTS_DIR_NO_SEE_NAMES,
         "metrics/GlobalOpinionsQA/gpt-4.1-family/agents_3_questions_2089_rotated"),
         (INDEP_PRMPT_GPT41_RESULTS_DIR_SEE_NAMES,
          "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/critical_independent_named"
          ),
         (INDEP_PRMPT_GPT41_RESULTS_DIR_ANON,
          "metrics/GlobalOpinionsQA/gpt-4.1-family/critical_independent_anonymous"
          ),
         (ADVERSARIAL_GPT41_RESULTS_DIR_ANON,
          "metrics/GlobalOpinionsQA/gpt-4.1-family/adversarial_anonymous"
          ),
         (ADVERSARIAL_GPT41_RESULTS_DIR_NAMED,
          "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/adversarial_named"
          ),
         (WITHOUT_REVISE_PROMPT_GPT_41_ADVERSARIAL_NAMED,
          "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/without_revise/adversarial_named"
          ),
         (WITHOUT_REVISE_PROMP_GPT_41_ADVERSARIAL_ANON,
          "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/without_revise/adversarial_anonymous"
          ),
         (WITHOUT_REVISE_PROMPT_GPT_41_CRITICAL_INDEPENDENT_NAMED,
          "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/without_revise/critical_independent_named"
          ),
         (WITHOUT_REVISE_PROMPT_GPT_41_CRITICAL_INDEPENDENT_ANON,
          "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/without_revise/critical_independent_anonymous"
          ),
         (TWENTY_ROUND_GPT41_RESULTS_DIR_SEE_NAMES,
          "metrics/GlobalOpinionsQA/agent_names/20_rounds/gpt-4.1-fam/named"
          ),
         (TWENTY_ROUND_GPT41_RESULTS_DIR_NO_SEE_NAMES,
          "metrics/GlobalOpinionsQA/gpt-4.1-family/20_rounds/anonymous"
          ),
    ]

    for results_dir, metrics_dir in metrics_runs:
        compute_metrics(results_dir, metrics_dir)

    # ── Step 2: pair-level disagreement + directional deference analysis ───────

    # ── Random models: see names ──────────────────────────────────────────────
    print("\n=== RANDOM MODELS (SEE NAMES): NORMAL vs ROTATED ANALYSIS ===\n")

    rand_see_orig = load_results(ORIGINAL_RANDOM_RESULTS_DIR_SEE_NAMES)
    rand_see_rot  = load_results(ROTATED_RANDOM_RESULTS_DIR_SEE_NAMES)

    print(f"Loaded {len(rand_see_orig)} normal questions")
    print(f"Loaded {len(rand_see_rot)} rotated questions")

    per_round_norm = pair_disagree(rand_see_orig)
    print_per_round("NORMAL", per_round_norm)
    save_per_round(per_round_norm, ROUND_JSON_RANDOM_SEE_NORMAL)

    per_round_rot = pair_disagree(rand_see_rot)
    print_per_round("ROTATED", per_round_rot)
    save_per_round(per_round_rot, ROUND_JSON_RANDOM_SEE_ROTATED)

    init_norm, s2l_norm, l2s_norm = analyze_conditioned(rand_see_orig, rand_see_orig)
    print_directional("NORMAL", init_norm, s2l_norm, l2s_norm)
    save_directional(init_norm, s2l_norm, l2s_norm, DIRECTIONAL_JSON_RANDOM_SEE_NORMAL)

    init_rot, s2l_rot, l2s_rot = analyze_conditioned(rand_see_orig, rand_see_rot)
    print_directional("ROTATED", init_rot, s2l_rot, l2s_rot)
    save_directional(init_rot, s2l_rot, l2s_rot, DIRECTIONAL_JSON_RANDOM_SEE_ROTATED)

    # ── GPT-4.1: see names ────────────────────────────────────────────────────
    print("\n=== GPT-4.1 FAMILY (SEE NAMES): NORMAL vs ROTATED ANALYSIS ===\n")

    gpt_see_orig = load_results(ORIGINAL_GPT41_RESULTS_DIR_SEE_NAMES)
    gpt_see_rot  = load_results(ROTATED_GPT41_RESULTS_DIR_SEE_NAMES)

    print(f"Loaded {len(gpt_see_orig)} normal questions")
    print(f"Loaded {len(gpt_see_rot)} rotated questions")

    per_round_norm = pair_disagree(gpt_see_orig)
    print_per_round("NORMAL", per_round_norm)
    save_per_round(per_round_norm, ROUND_JSON_GPT41_SEE_NORMAL)

    per_round_rot = pair_disagree(gpt_see_rot)
    print_per_round("ROTATED", per_round_rot)
    save_per_round(per_round_rot, ROUND_JSON_GPT41_SEE_ROTATED)

    init_norm, s2l_norm, l2s_norm = analyze_conditioned(gpt_see_orig, gpt_see_orig)
    print_directional("NORMAL", init_norm, s2l_norm, l2s_norm)
    save_directional(init_norm, s2l_norm, l2s_norm, DIRECTIONAL_JSON_GPT41_SEE_NORMAL)

    init_rot, s2l_rot, l2s_rot = analyze_conditioned(gpt_see_orig, gpt_see_rot)
    print_directional("ROTATED", init_rot, s2l_rot, l2s_rot)
    save_directional(init_rot, s2l_rot, l2s_rot, DIRECTIONAL_JSON_GPT41_SEE_ROTATED)

    # ── Random models: no see names ───────────────────────────────────────────
    print("\n=== RANDOM MODELS (NO SEE NAMES): NORMAL vs ROTATED ANALYSIS ===\n")

    rand_no_orig = load_results(ORIGINAL_RANDOM_RESULTS_DIR_NO_SEE_NAMES)
    rand_no_rot  = load_results(ROTATED_RANDOM_RESULTS_DIR_NO_SEE_NAMES)

    print(f"Loaded {len(rand_no_orig)} normal questions")
    print(f"Loaded {len(rand_no_rot)} rotated questions")

    per_round_norm = pair_disagree(rand_no_orig)
    print_per_round("NORMAL", per_round_norm)
    save_per_round(per_round_norm, ROUND_JSON_RANDOM_NO_SEE_NORMAL)

    per_round_rot = pair_disagree(rand_no_rot)
    print_per_round("ROTATED", per_round_rot)
    save_per_round(per_round_rot, ROUND_JSON_RANDOM_NO_SEE_ROTATED)

    init_norm, s2l_norm, l2s_norm = analyze_conditioned(rand_no_orig, rand_no_orig)
    print_directional("NORMAL", init_norm, s2l_norm, l2s_norm)
    save_directional(init_norm, s2l_norm, l2s_norm, DIRECTIONAL_JSON_RANDOM_NO_SEE_NORMAL)

    init_rot, s2l_rot, l2s_rot = analyze_conditioned(rand_no_orig, rand_no_rot)
    print_directional("ROTATED", init_rot, s2l_rot, l2s_rot)
    save_directional(init_rot, s2l_rot, l2s_rot, DIRECTIONAL_JSON_RANDOM_NO_SEE_ROTATED)

    # ── GPT-4.1: no see names ─────────────────────────────────────────────────
    print("\n=== GPT-4.1 FAMILY (NO SEE NAMES): NORMAL vs ROTATED ANALYSIS ===\n")

    gpt_no_orig = load_results(ORIGINAL_GPT41_RESULTS_DIR_NO_SEE_NAMES)
    gpt_no_rot  = load_results(ROTATED_GPT41_RESULTS_DIR_NO_SEE_NAMES)

    print(f"Loaded {len(gpt_no_orig)} normal questions")
    print(f"Loaded {len(gpt_no_rot)} rotated questions")

    per_round_norm = pair_disagree(gpt_no_orig)
    print_per_round("NORMAL", per_round_norm)
    save_per_round(per_round_norm, ROUND_JSON_GPT41_NO_SEE_NORMAL)

    per_round_rot = pair_disagree(gpt_no_rot)
    print_per_round("ROTATED", per_round_rot)
    save_per_round(per_round_rot, ROUND_JSON_GPT41_NO_SEE_ROTATED)

    init_norm, s2l_norm, l2s_norm = analyze_conditioned(gpt_no_orig, gpt_no_orig)
    print_directional("NORMAL", init_norm, s2l_norm, l2s_norm)
    save_directional(init_norm, s2l_norm, l2s_norm, DIRECTIONAL_JSON_GPT41_NO_SEE_NORMAL)

    init_rot, s2l_rot, l2s_rot = analyze_conditioned(gpt_no_orig, gpt_no_rot)
    print_directional("ROTATED", init_rot, s2l_rot, l2s_rot)
    save_directional(init_rot, s2l_rot, l2s_rot, DIRECTIONAL_JSON_GPT41_NO_SEE_ROTATED)

    # ── GPT-4.1: see names, independent prompt ────────────────────────────────
    print("\n=== GPT-4.1 FAMILY (SEE NAMES): INDEPENDENT PROMPT ANALYSIS ===\n")

    gpt_see_indep = load_results(INDEP_PRMPT_GPT41_RESULTS_DIR_SEE_NAMES)

    print(f"Loaded {len(gpt_see_indep)} independent questions")

    per_round_indep = pair_disagree(gpt_see_indep)
    print_per_round("INDEP", per_round_indep)
    save_per_round(per_round_indep, ROUND_JSON_GPT41_SEE_INDEP)

    init_indep, s2l_indep, l2s_indep = analyze_conditioned(gpt_see_indep, gpt_see_indep)
    print_directional("INDEP", init_indep, s2l_indep, l2s_indep)
    save_directional(init_indep, s2l_indep, l2s_indep, DIRECTIONAL_JSON_GPT41_SEE_INDEP)

    # ── GPT-4.1: critical_independent/anonymous ──────────────────────────────
    print("\n=== GPT-4.1 FAMILY: INDEPENDENT PROMPT (ANONYMOUS) ANALYSIS ===\n")

    gpt_indep_anon = load_results(INDEP_PRMPT_GPT41_RESULTS_DIR_ANON)

    print(f"Loaded {len(gpt_indep_anon)} independent anonymous questions")

    per_round_indep_anon = pair_disagree(gpt_indep_anon)
    print_per_round("INDEP_ANON", per_round_indep_anon)
    save_per_round(per_round_indep_anon, ROUND_JSON_GPT41_SEE_INDEP_ANON)

    init_indep_anon, s2l_indep_anon, l2s_indep_anon = analyze_conditioned(gpt_indep_anon, gpt_indep_anon)
    print_directional("INDEP_ANON", init_indep_anon, s2l_indep_anon, l2s_indep_anon)
    save_directional(init_indep_anon, s2l_indep_anon, l2s_indep_anon, DIRECTIONAL_JSON_GPT41_SEE_INDEP_ANON)

    # ── GPT-4.1: adversarial/anonymous ───────────────────────────────────────
    print("\n=== GPT-4.1 FAMILY: ADVERSARIAL (ANONYMOUS) ANALYSIS ===\n")

    gpt_adv_anon = load_results(ADVERSARIAL_GPT41_RESULTS_DIR_ANON)

    print(f"Loaded {len(gpt_adv_anon)} adversarial anonymous questions")

    per_round_adv_anon = pair_disagree(gpt_adv_anon)
    print_per_round("ADVERSARIAL_ANON", per_round_adv_anon)
    save_per_round(per_round_adv_anon, ROUND_JSON_GPT41_SEE_ADVERSARIAL_ANON)

    init_adv_anon, s2l_adv_anon, l2s_adv_anon = analyze_conditioned(gpt_adv_anon, gpt_adv_anon)
    print_directional("ADVERSARIAL_ANON", init_adv_anon, s2l_adv_anon, l2s_adv_anon)
    save_directional(init_adv_anon, s2l_adv_anon, l2s_adv_anon, DIRECTIONAL_JSON_GPT41_SEE_ADVERSARIAL_ANON)

    # ── GPT-4.1: adversarial/named ───────────────────────────────────────────
    print("\n=== GPT-4.1 FAMILY: ADVERSARIAL (NAMED) ANALYSIS ===\n")

    gpt_adv_named = load_results(ADVERSARIAL_GPT41_RESULTS_DIR_NAMED)

    print(f"Loaded {len(gpt_adv_named)} adversarial named questions")

    per_round_adv_named = pair_disagree(gpt_adv_named)
    print_per_round("ADVERSARIAL_NAMED", per_round_adv_named)
    save_per_round(per_round_adv_named, ROUND_JSON_GPT41_SEE_ADVERSARIAL_NAMED)

    init_adv_named, s2l_adv_named, l2s_adv_named = analyze_conditioned(gpt_adv_named, gpt_adv_named)
    print_directional("ADVERSARIAL_NAMED", init_adv_named, s2l_adv_named, l2s_adv_named)
    save_directional(init_adv_named, s2l_adv_named, l2s_adv_named, DIRECTIONAL_JSON_GPT41_SEE_ADVERSARIAL_NAMED)

    # ── GPT-4.1: without_revise/adversarial/named ────────────────────────────
    print("\n=== GPT-4.1 FAMILY: WITHOUT REVISE — ADVERSARIAL (NAMED) ANALYSIS ===\n")

    gpt_wo_adv_named = load_results(WITHOUT_REVISE_PROMPT_GPT_41_ADVERSARIAL_NAMED)

    print(f"Loaded {len(gpt_wo_adv_named)} without-revise adversarial named questions")

    per_round_wo_adv_named = pair_disagree(gpt_wo_adv_named)
    print_per_round("WITHOUT_REVISE_ADV_NAMED", per_round_wo_adv_named)
    save_per_round(per_round_wo_adv_named, ROUND_JSON_WITHOUT_REVISE_GPT41_ADV_NAMED)

    init_wo_adv_named, s2l_wo_adv_named, l2s_wo_adv_named = analyze_conditioned(gpt_wo_adv_named, gpt_wo_adv_named)
    print_directional("WITHOUT_REVISE_ADV_NAMED", init_wo_adv_named, s2l_wo_adv_named, l2s_wo_adv_named)
    save_directional(init_wo_adv_named, s2l_wo_adv_named, l2s_wo_adv_named, DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_ADV_NAMED)

    # ── GPT-4.1: without_revise/adversarial/anonymous ────────────────────────
    print("\n=== GPT-4.1 FAMILY: WITHOUT REVISE — ADVERSARIAL (ANONYMOUS) ANALYSIS ===\n")

    gpt_wo_adv_anon = load_results(WITHOUT_REVISE_PROMP_GPT_41_ADVERSARIAL_ANON)

    print(f"Loaded {len(gpt_wo_adv_anon)} without-revise adversarial anonymous questions")

    per_round_wo_adv_anon = pair_disagree(gpt_wo_adv_anon)
    print_per_round("WITHOUT_REVISE_ADV_ANON", per_round_wo_adv_anon)
    save_per_round(per_round_wo_adv_anon, ROUND_JSON_WITHOUT_REVISE_GPT41_ADV_ANON)

    init_wo_adv_anon, s2l_wo_adv_anon, l2s_wo_adv_anon = analyze_conditioned(gpt_wo_adv_anon, gpt_wo_adv_anon)
    print_directional("WITHOUT_REVISE_ADV_ANON", init_wo_adv_anon, s2l_wo_adv_anon, l2s_wo_adv_anon)
    save_directional(init_wo_adv_anon, s2l_wo_adv_anon, l2s_wo_adv_anon, DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_ADV_ANON)

    # ── GPT-4.1: without_revise/critical_independent/named ───────────────────
    print("\n=== GPT-4.1 FAMILY: WITHOUT REVISE — CRITICAL INDEPENDENT (NAMED) ANALYSIS ===\n")

    gpt_wo_ci_named = load_results(WITHOUT_REVISE_PROMPT_GPT_41_CRITICAL_INDEPENDENT_NAMED)

    print(f"Loaded {len(gpt_wo_ci_named)} without-revise critical-independent named questions")

    per_round_wo_ci_named = pair_disagree(gpt_wo_ci_named)
    print_per_round("WITHOUT_REVISE_CRIT_INDEP_NAMED", per_round_wo_ci_named)
    save_per_round(per_round_wo_ci_named, ROUND_JSON_WITHOUT_REVISE_GPT41_CRIT_INDEP_NAMED)

    init_wo_ci_named, s2l_wo_ci_named, l2s_wo_ci_named = analyze_conditioned(gpt_wo_ci_named, gpt_wo_ci_named)
    print_directional("WITHOUT_REVISE_CRIT_INDEP_NAMED", init_wo_ci_named, s2l_wo_ci_named, l2s_wo_ci_named)
    save_directional(init_wo_ci_named, s2l_wo_ci_named, l2s_wo_ci_named, DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_CRIT_INDEP_NAMED)

    # ── GPT-4.1: without_revise/critical_independent/anonymous ───────────────
    print("\n=== GPT-4.1 FAMILY: WITHOUT REVISE — CRITICAL INDEPENDENT (ANONYMOUS) ANALYSIS ===\n")

    gpt_wo_ci_anon = load_results(WITHOUT_REVISE_PROMPT_GPT_41_CRITICAL_INDEPENDENT_ANON)

    print(f"Loaded {len(gpt_wo_ci_anon)} without-revise critical-independent anonymous questions")

    per_round_wo_ci_anon = pair_disagree(gpt_wo_ci_anon)
    print_per_round("WITHOUT_REVISE_CRIT_INDEP_ANON", per_round_wo_ci_anon)
    save_per_round(per_round_wo_ci_anon, ROUND_JSON_WITHOUT_REVISE_GPT41_CRIT_INDEP_ANON)

    init_wo_ci_anon, s2l_wo_ci_anon, l2s_wo_ci_anon = analyze_conditioned(gpt_wo_ci_anon, gpt_wo_ci_anon)
    print_directional("WITHOUT_REVISE_CRIT_INDEP_ANON", init_wo_ci_anon, s2l_wo_ci_anon, l2s_wo_ci_anon)
    save_directional(init_wo_ci_anon, s2l_wo_ci_anon, l2s_wo_ci_anon, DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_CRIT_INDEP_ANON)

    # ── GPT-4.1: 20 rounds, see names ────────────────────────────────────────
    print("\n=== GPT-4.1 FAMILY: 20 ROUNDS (SEE NAMES) ANALYSIS ===\n")

    gpt_twenty_see = load_results(TWENTY_ROUND_GPT41_RESULTS_DIR_SEE_NAMES)

    print(f"Loaded {len(gpt_twenty_see)} 20-round see-names questions")

    per_round_twenty_see = pair_disagree(gpt_twenty_see)
    print_per_round("TWENTY_ROUND_SEE", per_round_twenty_see)
    save_per_round(per_round_twenty_see, ROUND_JSON_TWENTY_ROUND_GPT41_SEE)

    init_twenty_see, s2l_twenty_see, l2s_twenty_see = analyze_conditioned(gpt_twenty_see, gpt_twenty_see)
    print_directional("TWENTY_ROUND_SEE", init_twenty_see, s2l_twenty_see, l2s_twenty_see)
    save_directional(init_twenty_see, s2l_twenty_see, l2s_twenty_see, DIRECTIONAL_JSON_TWENTY_ROUND_GPT41_SEE)

    # ── GPT-4.1: 20 rounds, no see names ─────────────────────────────────────
    print("\n=== GPT-4.1 FAMILY: 20 ROUNDS (NO SEE NAMES) ANALYSIS ===\n")

    gpt_twenty_no_see = load_results(TWENTY_ROUND_GPT41_RESULTS_DIR_NO_SEE_NAMES)

    print(f"Loaded {len(gpt_twenty_no_see)} 20-round no-see-names questions")

    per_round_twenty_no_see = pair_disagree(gpt_twenty_no_see)
    print_per_round("TWENTY_ROUND_NO_SEE", per_round_twenty_no_see)
    save_per_round(per_round_twenty_no_see, ROUND_JSON_TWENTY_ROUND_GPT41_NO_SEE)

    init_twenty_no_see, s2l_twenty_no_see, l2s_twenty_no_see = analyze_conditioned(gpt_twenty_no_see, gpt_twenty_no_see)
    print_directional("TWENTY_ROUND_NO_SEE", init_twenty_no_see, s2l_twenty_no_see, l2s_twenty_no_see)
    save_directional(init_twenty_no_see, s2l_twenty_no_see, l2s_twenty_no_see, DIRECTIONAL_JSON_TWENTY_ROUND_GPT41_NO_SEE)
