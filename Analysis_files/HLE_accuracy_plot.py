import json
import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_ROOT = "new_analysis_outputs/HLE"
FIGURES_ROOT = "new_figures/HLE"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def output_to_figures(output_dir):
    # Mirror a new_analysis_outputs/HLE/ path under new_figures/HLE/.
    rel = os.path.relpath(output_dir, OUTPUT_ROOT)
    return os.path.join(FIGURES_ROOT, rel)


def dir_label(output_dir):
    # Human-readable title derived from the relative path.
    rel = os.path.relpath(output_dir, OUTPUT_ROOT)
    return rel.replace(os.sep, " — ")


def find_dirs_with(root, filename):
    # Yield all directories under root that contain filename.
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            yield dirpath


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_rounds_accuracy(output_dir, title, out_path):
    data = load_json(os.path.join(output_dir, "hle_accuracy.json"))
    rounds_data = data["rounds_data"]
    sorted_keys = sorted(rounds_data.keys(), key=lambda k: int(k.split("_")[1]))
    values = [rounds_data[k]["accuracy_pct"] for k in sorted_keys]
    rounds = list(range(1, len(values) + 1))

    plt.figure(figsize=(12, 5))
    plt.plot(rounds, values, marker="o")
    plt.xticks(rounds)
    plt.xlabel("Deliberation Round")
    plt.ylabel("Accuracy (all models correct) (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [accuracy] -> {out_path}")


def plot_deference_direction(output_dir, title, out_path):
    data = load_json(os.path.join(output_dir, "hle_deference_accuracy.json"))
    if not data:
        return

    labels = sorted(data.keys())
    small_to_large = [data[k].get("s2l_pct", 0) for k in labels]
    large_to_small = [data[k].get("l2s_pct", 0) for k in labels]

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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [deference direction] -> {out_path}")


def plot_deference_accuracy_outcomes(output_dir, title, out_path):
    data = load_json(os.path.join(output_dir, "hle_deference_accuracy.json"))
    if not data:
        return

    labels = sorted(data.keys())
    s2l_w2c = [data[k].get("s2l_w2c_pct", 0) for k in labels]
    s2l_c2w = [data[k].get("s2l_c2w_pct", 0) for k in labels]
    l2s_w2c = [data[k].get("l2s_w2c_pct", 0) for k in labels]
    l2s_c2w = [data[k].get("l2s_c2w_pct", 0) for k in labels]

    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(12, 5))
    plt.bar(x - 1.5 * width, s2l_w2c, width, label="s2l: wrong → correct")
    plt.bar(x - 0.5 * width, s2l_c2w, width, label="s2l: correct → wrong")
    plt.bar(x + 0.5 * width, l2s_w2c, width, label="l2s: wrong → correct")
    plt.bar(x + 1.5 * width, l2s_c2w, width, label="l2s: correct → wrong")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Percentage of Deferences in Branch (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [deference accuracy]  -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dirs_accuracy   = sorted(find_dirs_with(OUTPUT_ROOT, "hle_accuracy.json"))
    dirs_deference  = sorted(find_dirs_with(OUTPUT_ROOT, "hle_deference_accuracy.json"))

    print(f"Found {len(dirs_accuracy)} accuracy, {len(dirs_deference)} deference director{'y' if len(dirs_deference) == 1 else 'ies'}.\n")

    for output_dir in dirs_accuracy:
        label = dir_label(output_dir)
        fig_dir = output_to_figures(output_dir)
        print(f"Plotting: {output_dir}")
        plot_rounds_accuracy(
            output_dir,
            f"Accuracy Across Rounds ({label})",
            os.path.join(fig_dir, "rounds_accuracy.png"),
        )

    for output_dir in dirs_deference:
        label = dir_label(output_dir)
        fig_dir = output_to_figures(output_dir)
        print(f"Plotting deference: {output_dir}")
        plot_deference_direction(
            output_dir,
            f"Direction of Deference ({label})",
            os.path.join(fig_dir, "deference_direction.png"),
        )
        plot_deference_accuracy_outcomes(
            output_dir,
            f"Deference Accuracy Outcomes ({label})",
            os.path.join(fig_dir, "deference_accuracy_outcomes.png"),
        )

    print(f"\nAll figures saved under '{FIGURES_ROOT}/'.")
