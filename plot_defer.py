import json
import os
import matplotlib.pyplot as plt
import numpy as np


RANDOM_JSON = "analysis_outputs/random_models/random_models_directional.json"
GPT41_JSON = "analysis_outputs/OpinionsQA/test/gpt-4.1-fam/gpt4.1_family_directional.json"

FIG_DIR = "figures/directional_metrics"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_directional_bars(data, title, filename):
    labels = list(data.keys())
    small_to_large = [data[k]["small_to_large_pct"] for k in labels]
    large_to_small = [data[k]["large_to_small_pct"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, small_to_large, width, label="Small → Large")
    plt.bar(x + width / 2, large_to_small, width, label="Large → Small")

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Percentage of Disagreements (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


def plot_asymmetry_ratios(data, title, filename):
    labels = list(data.keys())
    ratios = []

    for k in labels:
        s = data[k]["small_to_large_pct"]
        l = data[k]["large_to_small_pct"]
        ratios.append(s / l if l > 0 else 0)

    plt.figure(figsize=(10, 4))
    plt.bar(labels, ratios)
    plt.axhline(1.0)
    plt.ylabel("Asymmetry Ratio (Small → Large / Large → Small)")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


if __name__ == "__main__":
    random_data = load_json(RANDOM_JSON)
    gpt41_data = load_json(GPT41_JSON)

    # Plot 1: Directional disagreement
    plot_directional_bars(
        random_data,
        "Direction of Disagreement (Random Models)",
        "directional_bars_random_models.png",
    )

    plot_directional_bars(
        gpt41_data,
        "Direction of Disagreement (GPT-4.1 Family on OpinionsQA Test)",
        "directional_bars_gpt41_family_opinionsqa.png",
    )

    # Plot 2: Asymmetry ratios
    plot_asymmetry_ratios(
        random_data,
        "Asymmetric Alignment Strength (Frontier Models)",
        "asymmetry_ratios_frontier_models.png",
    )

    plot_asymmetry_ratios(
        gpt41_data,
        "Asymmetric Alignment Strength (GPT-4.1 Family on OpinionsQA Test)",
        "asymmetry_ratios_gpt41_family_opinionsqa.png",
    )

    print(f"Figures saved to: {FIG_DIR}")