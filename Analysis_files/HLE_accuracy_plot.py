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

def plot_rounds_accuracy(output_dir, out_path):
    data = load_json(os.path.join(output_dir, "hle_accuracy.json"))
    rounds_data = data["rounds_data"]
    sorted_keys = sorted(rounds_data.keys(), key=lambda k: int(k.split("_")[1]))
    values = [rounds_data[k]["accuracy_pct"] for k in sorted_keys]
    rounds = list(range(1, len(values) + 1))

    plt.figure(figsize=(12, 5))
    plt.plot(rounds, values, marker="o", linewidth=2.5, markersize=6)
    plt.xticks(rounds, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Deliberation Round", fontsize=16)
    plt.ylabel("Accuracy (all models correct) (%)", fontsize=16)
    plt.title("Accuracy Across Rounds", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [accuracy]            -> {out_path}")


def plot_deference_direction(output_dir, out_path):
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
    plt.xticks(x, labels, rotation=30, ha="right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel("Percentage of Disagreements (%)")
    plt.title("Direction of Deference")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [deference direction] -> {out_path}")


def plot_deference_accuracy_outcomes(output_dir, out_path):
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
    plt.xticks(x, labels, rotation=30, ha="right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel("Percentage of Deferences in Branch (%)")
    plt.title("Deference Accuracy Outcomes")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [deference accuracy]  -> {out_path}")


def load_accuracy_values(output_dir):
    """Return (rounds list, accuracy_pct list) from hle_accuracy.json in output_dir."""
    data = load_json(os.path.join(output_dir, "hle_accuracy.json"))
    rounds_data = data["rounds_data"]
    sorted_keys = sorted(rounds_data.keys(), key=lambda k: int(k.split("_")[1]))
    values = [rounds_data[k]["accuracy_pct"] for k in sorted_keys]
    return list(range(1, len(values) + 1)), values


def find_combined_parents(root):
    """Yield parent dirs that have both named/ and anonymous/ with hle_accuracy.json."""
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root):
        if "hle_accuracy.json" in filenames:
            parent = os.path.dirname(dirpath)
            if parent in seen:
                continue
            named_metrics = os.path.join(parent, "named", "hle_accuracy.json")
            anon_metrics = os.path.join(parent, "anonymous", "hle_accuracy.json")
            if os.path.exists(named_metrics) and os.path.exists(anon_metrics):
                seen.add(parent)
                yield parent


def find_combined_rotated_parents(root):
    """Yield parent dirs that have both named_rotated/ and anonymous_rotated/ with hle_accuracy.json."""
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root):
        if "hle_accuracy.json" in filenames:
            parent = os.path.dirname(dirpath)
            if parent in seen:
                continue
            named_metrics = os.path.join(parent, "named_rotated", "hle_accuracy.json")
            anon_metrics = os.path.join(parent, "anonymous_rotated", "hle_accuracy.json")
            if os.path.exists(named_metrics) and os.path.exists(anon_metrics):
                seen.add(parent)
                yield parent


def plot_combined_variants(parent_output_dir, out_path):
    named_dir = os.path.join(parent_output_dir, "named")
    anon_dir = os.path.join(parent_output_dir, "anonymous")

    rounds_n, values_n = load_accuracy_values(named_dir)
    rounds_a, values_a = load_accuracy_values(anon_dir)

    plt.figure(figsize=(12, 5))
    plt.plot(rounds_n, values_n, marker="o", linewidth=2.5, markersize=6, label="Named")
    plt.plot(rounds_a, values_a, marker="s", linewidth=2.5, markersize=6, linestyle="--", label="Anonymous")
    plt.xticks(rounds_n, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Deliberation Round", fontsize=16)
    plt.ylabel("Accuracy (all models correct) (%)", fontsize=16)
    plt.title("Accuracy Across Rounds — Named vs Anonymous", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [combined accuracy]   -> {out_path}")


def plot_combined_rotated_variants(parent_output_dir, out_path):
    named_dir = os.path.join(parent_output_dir, "named_rotated")
    anon_dir = os.path.join(parent_output_dir, "anonymous_rotated")

    rounds_n, values_n = load_accuracy_values(named_dir)
    rounds_a, values_a = load_accuracy_values(anon_dir)

    plt.figure(figsize=(12, 5))
    plt.plot(rounds_n, values_n, marker="o", linewidth=2.5, markersize=6, label="Named Rotated")
    plt.plot(rounds_a, values_a, marker="s", linewidth=2.5, markersize=6, linestyle="--", label="Anonymous Rotated")
    plt.xticks(rounds_n, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Deliberation Round", fontsize=16)
    plt.ylabel("Accuracy (all models correct) (%)", fontsize=16)
    plt.title("Accuracy Across Rounds — Named Rotated vs Anonymous Rotated", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [combined rotated]    -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dirs_accuracy   = sorted(find_dirs_with(OUTPUT_ROOT, "hle_accuracy.json"))
    dirs_deference  = sorted(find_dirs_with(OUTPUT_ROOT, "hle_deference_accuracy.json"))

    print(f"Found {len(dirs_accuracy)} accuracy, {len(dirs_deference)} deference director{'y' if len(dirs_deference) == 1 else 'ies'}.\n")

    for output_dir in dirs_accuracy:
        fig_dir = output_to_figures(output_dir)
        print(f"Plotting: {output_dir}")
        plot_rounds_accuracy(
            output_dir,
            os.path.join(fig_dir, "rounds_accuracy.png"),
        )

    for output_dir in dirs_deference:
        fig_dir = output_to_figures(output_dir)
        print(f"Plotting deference: {output_dir}")
        plot_deference_direction(
            output_dir,
            os.path.join(fig_dir, "deference_direction.png"),
        )
        plot_deference_accuracy_outcomes(
            output_dir,
            os.path.join(fig_dir, "deference_accuracy_outcomes.png"),
        )

    combined_parents = sorted(find_combined_parents(OUTPUT_ROOT))
    print(f"\nFound {len(combined_parents)} experiment(s) with both named and anonymous variants.\n")

    for parent_output_dir in combined_parents:
        fig_dir = output_to_figures(parent_output_dir)
        print(f"Plotting combined: {parent_output_dir}")
        plot_combined_variants(
            parent_output_dir,
            os.path.join(fig_dir, "rounds_accuracy_combined.png"),
        )

    combined_rotated_parents = sorted(find_combined_rotated_parents(OUTPUT_ROOT))
    print(f"\nFound {len(combined_rotated_parents)} experiment(s) with both named_rotated and anonymous_rotated variants.\n")

    for parent_output_dir in combined_rotated_parents:
        fig_dir = output_to_figures(parent_output_dir)
        print(f"Plotting combined rotated: {parent_output_dir}")
        plot_combined_rotated_variants(
            parent_output_dir,
            os.path.join(fig_dir, "rounds_accuracy_combined_rotated.png"),
        )

    print(f"\nAll figures saved under '{FIGURES_ROOT}/'.")
