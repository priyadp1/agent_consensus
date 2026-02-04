import os
import json
import argparse
from datetime import datetime


def round_disagree(data, round_idx):
    rounds = data.get("rounds", [])
    if len(rounds) <= round_idx:
        return False

    answers = [
        agent_data.get("answer")
        for agent_data in rounds[round_idx].values()
        if agent_data.get("answer") not in (None, "INVALID")
    ]

    if len(answers) < 2:
        return False

    return len(set(answers)) > 1


def main():
    parser = argparse.ArgumentParser(
        description="Compute inter-agent disagreement statistics"
    )
    parser.add_argument(
        "--folder",
        default="OpinionsQA/train/gpt-4.1-family",
        required=False,
        help="Folder name (e.g., DeepSeek-R1, Llama-3.3-70B-Instruct)"
    )
    parser.add_argument(
        "--results-subdir",
        default="agents_3_questions_2556",
        help="Subdirectory under results/<folder>/ (default: agents_3_questions_2556)"
    )

    args = parser.parse_args()
    model_name = args.folder
    results_dir = os.path.join("results", model_name, args.results_subdir)
    metrics_dir = os.path.join("metrics", model_name)
    os.makedirs(metrics_dir, exist_ok=True)

    total = 0
    disagreements = [0, 0, 0]

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for fname in os.listdir(results_dir):
        if not fname.startswith("q_") or not fname.endswith(".json"):
            continue

        with open(os.path.join(results_dir, fname), "r") as f:
            data = json.load(f)

        total += 1
        for i in range(3):
            if round_disagree(data, i):
                disagreements[i] += 1

    if total == 0:
        print("No result files found.")
        return

    percentages = [100 * d / total for d in disagreements]

    metrics = {
        "model": model_name,
        "total_questions": total,
        "rounds": {
            "round_1": {
                "disagreement_count": disagreements[0],
                "disagreement_percentage": round(percentages[0], 2),
            },
            "round_2": {
                "disagreement_count": disagreements[1],
                "disagreement_percentage": round(percentages[1], 2),
            },
            "round_3": {
                "disagreement_count": disagreements[2],
                "disagreement_percentage": round(percentages[2], 2),
            },
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    out_path = os.path.join(metrics_dir, "interagent_disagree.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Console output
    print(f"Results for model: {model_name}")
    print(f"Total questions analyzed: {total}")
    for i in range(3):
        print(f"Questions with disagreement in round {i+1}: {disagreements[i]}")
        print(f"Percentage with round-{i+1} disagreement: {percentages[i]:.2f}%")

    print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()
