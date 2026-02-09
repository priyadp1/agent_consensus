import os
import json
import argparse
from datetime import datetime


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


def main():
    parser = argparse.ArgumentParser(
        description="Compute inter-agent disagreement statistics"
    )
    parser.add_argument(
        "--folder",
        default="GlobalOpinionsQA/agent_names/gpt-4.1-fam",
        required=False
    )
    parser.add_argument(
        "--results-subdir",
        default="agents_3_questions_2556_rotated",
        required=False
    )

    args = parser.parse_args()
    results_dir = os.path.join(
        "results",
        args.folder.strip("/"),
        args.results_subdir.strip("/")
    )

    metrics_dir = os.path.join(
        "metrics",
        args.folder.strip("/"),
        args.results_subdir.strip("/")
    )

    os.makedirs(metrics_dir, exist_ok=True)

    # For logging / metadata only (NOT a path)
    model_name = f"{args.folder}/{args.results_subdir}"

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    total_valid = [0, 0, 0]
    disagreements = [0, 0, 0]

    for fname in os.listdir(results_dir):
        if not fname.startswith("q_") or not fname.endswith(".json"):
            continue

        with open(os.path.join(results_dir, fname), "r") as f:
            data = json.load(f)

        for i in range(3):
            result = round_disagree(data, i)
            if result is None:
                continue
            total_valid[i] += 1
            if result:
                disagreements[i] += 1

    percentages = [
        (100 * disagreements[i] / total_valid[i]) if total_valid[i] > 0 else 0
        for i in range(3)
    ]

    metrics = {
        "model": model_name,
        "rounds": {
            "round_1": {
                "valid_questions": total_valid[0],
                "disagreement_count": disagreements[0],
                "disagreement_percentage": round(percentages[0], 2),
            },
            "round_2": {
                "valid_questions": total_valid[1],
                "disagreement_count": disagreements[1],
                "disagreement_percentage": round(percentages[1], 2),
            },
            "round_3": {
                "valid_questions": total_valid[2],
                "disagreement_count": disagreements[2],
                "disagreement_percentage": round(percentages[2], 2),
            },
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    out_path = os.path.join(metrics_dir, "interagent_disagree.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results for model: {model_name}")
    for i in range(3):
        print(f"Round {i+1}:")
        print(f"  Valid questions: {total_valid[i]}")
        print(f"  Disagreements: {disagreements[i]}")
        print(f"  Percentage: {percentages[i]:.2f}%")

    print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()
