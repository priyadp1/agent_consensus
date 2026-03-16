import json
import os
import matplotlib.pyplot as plt

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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dirs = sorted(find_dirs_with(OUTPUT_ROOT, "hle_accuracy.json"))

    if not dirs:
        print(f"No hle_accuracy.json files found under '{OUTPUT_ROOT}'.")
    else:
        print(f"Found {len(dirs)} director{'y' if len(dirs) == 1 else 'ies'}.\n")

    for output_dir in dirs:
        label = dir_label(output_dir)
        fig_dir = output_to_figures(output_dir)
        print(f"Plotting: {output_dir}")

        plot_rounds_accuracy(
            output_dir,
            f"Accuracy Across Rounds ({label})",
            os.path.join(fig_dir, "rounds_accuracy.png"),
        )

    print(f"\nAll figures saved under '{FIGURES_ROOT}/'.")
