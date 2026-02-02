import argparse
import json
import os
import matplotlib.pyplot as plt


def load_model_metrics(model_name, metrics_root):
    path = os.path.join(metrics_root, model_name, "interagent_disagree.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found for model '{model_name}': {path}")

    with open(path, "r") as f:
        data = json.load(f)

    rounds = data["rounds"]
    return [
        rounds["round_1"]["disagreement_percentage"],
        rounds["round_2"]["disagreement_percentage"],
        rounds["round_3"]["disagreement_percentage"],
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Plot inter-agent disagreement across deliberation rounds"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names (must exist under metrics/<model>/)"
    )
    parser.add_argument(
        "--metrics-root",
        default="metrics",
        help="Root directory containing model metrics (default: metrics/)"
    )
    parser.add_argument(
        "--out-dir",
        default="figures",
        help="Directory to save the figure (default: figures/)"
    )
    parser.add_argument(
        "--out-name",
        default="disagreement_across_rounds.png",
        help="Output filename"
    )
    parser.add_argument(
        "--title",
        default="Disagreement Across Deliberation Rounds",
        help="Plot title"
    )

    args = parser.parse_args()

    rounds = [1, 2, 3]
    plt.figure(figsize=(8, 5))

    for model in args.models:
        values = load_model_metrics(model, args.metrics_root)
        plt.plot(rounds, values, marker="o", label=model)

    plt.xticks(rounds)
    plt.xlabel("Deliberation Round")
    plt.ylabel("Questions with ≥ 1 Disagreement (%)")
    plt.title(args.title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
