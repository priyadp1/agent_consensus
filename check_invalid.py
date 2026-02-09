import json
import os
from collections import defaultdict


def disagreement_excluding_invalid(round_data):
    """
    round_data: dict[agent] -> {answer, raw_output, model_failed}
    Returns True if there is disagreement among VALID answers only.
    """
    valid_answers = {
        info["answer"]
        for info in round_data.values()
        if info["answer"] != "INVALID"
    }

    return len(valid_answers) > 1


def analyze_directory(results_dir):
    total_questions = 0
    disagreement_counts = defaultdict(int)

    # ===== DEBUG COUNTERS =====
    invalid_present = defaultdict(int)
    invalid_only_disagreements = defaultdict(int)
    # =========================

    files = sorted(
        f for f in os.listdir(results_dir)
        if f.startswith("q_") and f.endswith(".json")
    )

    for fname in files:
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)

        total_questions += 1

        for i, round_data in enumerate(data["rounds"], start=1):
            all_answers = {info["answer"] for info in round_data.values()}
            valid_answers = {
                info["answer"]
                for info in round_data.values()
                if info["answer"] != "INVALID"
            }

            # ----- DEBUG LOGIC -----
            if "INVALID" in all_answers:
                invalid_present[i] += 1

            if len(all_answers) > 1 and len(valid_answers) <= 1:
                invalid_only_disagreements[i] += 1
            # -----------------------

            if len(valid_answers) > 1:
                disagreement_counts[i] += 1

    print(f"\nResults (excluding INVALID answers)")
    print(f"Total questions analyzed: {total_questions}")
    for r in sorted(disagreement_counts):
        pct = 100 * disagreement_counts[r] / total_questions
        print(
            f"Questions with disagreement in round {r}: "
            f"{disagreement_counts[r]} ({pct:.2f}%)"
        )

    # ===== DEBUG OUTPUT =====
    print("\nDebug: INVALID impact analysis")
    for r in sorted(set(invalid_present) | set(invalid_only_disagreements)):
        print(
            f"Round {r}: "
            f"INVALID present in {invalid_present[r]} questions, "
            f"INVALID-only disagreements: {invalid_only_disagreements[r]}"
        )
    # =======================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        required=False,
        default="results/GlobalOpinionsQA/agent_names/gpt-4.1-fam/agents_3_questions_2556",
        help="Path to experiment results directory"
    )
    args = parser.parse_args()

    analyze_directory(args.results_dir)
