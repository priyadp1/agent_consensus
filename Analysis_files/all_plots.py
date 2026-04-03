import json
import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_ROOT = "new_analysis_outputs"
FIGURES_ROOT = "new_figures"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def output_to_figures(output_dir):
    #Mirror a new_analysis_outputs/ path under figures/.
    rel = os.path.relpath(output_dir, OUTPUT_ROOT)
    return os.path.join(FIGURES_ROOT, rel)


def dir_label(output_dir):
    #Human-readable title derived from the relative path.
    rel = os.path.relpath(output_dir, OUTPUT_ROOT)
    return rel.replace(os.sep, " — ")


def find_dirs_with(root, filename):
    #Yield all directories under root that contain filename.
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            yield dirpath


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_deference(data, out_path):
    labels = list(data.keys())
    small_to_large = [data[k]["small_to_large_pct"] for k in labels]
    large_to_small = [data[k]["large_to_small_pct"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(16, 7))
    plt.bar(x - width / 2, small_to_large, width, label="Small -> Large")
    plt.bar(x + width / 2, large_to_small, width, label="Large -> Small")
    plt.xticks(x, labels, rotation=30, ha="right", fontsize=26)
    plt.ylabel("Disagreement %", fontsize=26)
    plt.title("Direction of Disagreement", fontsize=26)
    plt.legend(fontsize=26, loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.tight_layout(pad=1.5)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [directional bars]  -> {out_path}")


def plot_rounds_disagreement(output_dir, out_path):
    metrics_path = os.path.join(output_dir, "interagent_disagree.json")
    data = load_json(metrics_path)
    rounds_data = data["rounds"]
    sorted_keys = sorted(rounds_data.keys(), key=lambda k: int(k.split("_")[1]))
    values = [rounds_data[k]["disagreement_percentage"] for k in sorted_keys]
    rounds = list(range(1, len(values) + 1))

    plt.figure(figsize=(16, 7))
    plt.plot(rounds, values, marker="o", linewidth=2.5, markersize=6)
    plt.xticks(rounds, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Deliberation Round", fontsize=20)
    plt.ylabel(" Disagreement (%)", fontsize=20)
    plt.title("Disagreement Across Rounds", fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [rounds]            -> {out_path}")


def load_disagreement_values(output_dir):
    """Return (rounds list, percentages list) from interagent_disagree.json in output_dir."""
    data = load_json(os.path.join(output_dir, "interagent_disagree.json"))
    rounds_data = data["rounds"]
    sorted_keys = sorted(rounds_data.keys(), key=lambda k: int(k.split("_")[1]))
    values = [rounds_data[k]["disagreement_percentage"] for k in sorted_keys]
    return list(range(1, len(values) + 1)), values


def find_combined_parents(root):
    """Yield parent dirs that have both named/ and anonymous/ with interagent_disagree.json."""
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root):
        if "interagent_disagree.json" in filenames:
            parent = os.path.dirname(dirpath)
            if parent in seen:
                continue
            named_metrics = os.path.join(parent, "named", "interagent_disagree.json")
            anon_metrics = os.path.join(parent, "anonymous", "interagent_disagree.json")
            if os.path.exists(named_metrics) and os.path.exists(anon_metrics):
                seen.add(parent)
                yield parent


def find_combined_rotated_parents(root):
    """Yield parent dirs that have both named_rotated/ and anonymous_rotated/ with interagent_disagree.json."""
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root):
        if "interagent_disagree.json" in filenames:
            parent = os.path.dirname(dirpath)
            if parent in seen:
                continue
            named_metrics = os.path.join(parent, "named_rotated", "interagent_disagree.json")
            anon_metrics = os.path.join(parent, "anonymous_rotated", "interagent_disagree.json")
            if os.path.exists(named_metrics) and os.path.exists(anon_metrics):
                seen.add(parent)
                yield parent


def plot_combined_variants(parent_output_dir, out_path):
    named_dir = os.path.join(parent_output_dir, "named")
    anon_dir = os.path.join(parent_output_dir, "anonymous")

    rounds_n, values_n = load_disagreement_values(named_dir)
    rounds_a, values_a = load_disagreement_values(anon_dir)

    plt.figure(figsize=(16, 7))
    plt.plot(rounds_n, values_n, marker="o", label="Named")
    plt.plot(rounds_a, values_a, marker="s", linestyle="--", label="Anonymous")
    plt.xticks(rounds_n, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Deliberation Round", fontsize=20)
    plt.ylabel(" Disagreement (%)", fontsize=20)
    plt.title("Disagreement Across Rounds — Named vs Anonymous", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [combined rounds]   -> {out_path}")


def plot_combined_rotated_variants(parent_output_dir, out_path):
    named_dir = os.path.join(parent_output_dir, "named_rotated")
    anon_dir = os.path.join(parent_output_dir, "anonymous_rotated")

    rounds_n, values_n = load_disagreement_values(named_dir)
    rounds_a, values_a = load_disagreement_values(anon_dir)

    plt.figure(figsize=(16, 7))
    plt.plot(rounds_n, values_n, marker="o", label="Named Rotated")
    plt.plot(rounds_a, values_a, marker="s", linestyle="--", label="Anonymous Rotated")
    plt.xticks(rounds_n, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Deliberation Round", fontsize=20)
    plt.ylabel(" Disagreement (%)", fontsize=20)
    plt.title("Disagreement Across Rounds — Named Rotated vs Anonymous Rotated", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [combined rotated]  -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dirs_with_directional = sorted(find_dirs_with(OUTPUT_ROOT, "directional.json"))
    dirs_with_metrics = sorted(find_dirs_with(OUTPUT_ROOT, "interagent_disagree.json"))

    print(f"Found {len(dirs_with_directional)} directional files, "
          f"{len(dirs_with_metrics)} metrics files.\n")

    for output_dir in dirs_with_directional:
        fig_dir = output_to_figures(output_dir)
        print(f"Plotting: {output_dir}")

        data = load_json(os.path.join(output_dir, "directional.json"))
        if not data:
            print("  [SKIP] Empty directional data.\n")
            continue

        plot_deference(
            data,
            os.path.join(fig_dir, "directional_bars.png"),
        )

    for output_dir in dirs_with_metrics:
        fig_dir = output_to_figures(output_dir)
        print(f"Plotting rounds: {output_dir}")

        plot_rounds_disagreement(
            output_dir,
            os.path.join(fig_dir, "rounds_disagreement.png"),
        )

    combined_parents = sorted(find_combined_parents(OUTPUT_ROOT))
    print(f"\nFound {len(combined_parents)} experiment(s) with both named and anonymous variants.\n")

    for parent_output_dir in combined_parents:
        fig_dir = output_to_figures(parent_output_dir)
        print(f"Plotting combined: {parent_output_dir}")

        plot_combined_variants(
            parent_output_dir,
            os.path.join(fig_dir, "rounds_disagreement_combined.png"),
        )

    combined_rotated_parents = sorted(find_combined_rotated_parents(OUTPUT_ROOT))
    print(f"\nFound {len(combined_rotated_parents)} experiment(s) with both named_rotated and anonymous_rotated variants.\n")

    for parent_output_dir in combined_rotated_parents:
        fig_dir = output_to_figures(parent_output_dir)
        print(f"Plotting combined rotated: {parent_output_dir}")

        plot_combined_rotated_variants(
            parent_output_dir,
            os.path.join(fig_dir, "rounds_disagreement_combined_rotated.png"),
        )

    print(f"\nAll figures saved under '{FIGURES_ROOT}/'.")
