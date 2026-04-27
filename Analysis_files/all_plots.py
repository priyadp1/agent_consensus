import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np


def shorten_model_name(name):
    """Shorten long model names to their family prefix.
    e.g. 'Llama-4-Maverick-17B-128E-Instruct-FP8' -> 'Llama', 'Phi-4' -> 'Phi'.
    Leaves lowercase-starting names like 'gpt-4.1' or 'gpt-4.1-mini' unchanged."""
    if name and name[0].isupper():
        return name.split('-')[0]
    return name


def clean_pair_label(key):
    """Strip model name from keys like 'gpt-4.1 (Agent 1) -> gpt-4.1 (Agent 2)' -> 'Agent 1 -> Agent 2'.
    For family-style keys like 'Llama-4-Maverick-... -> Phi-4', shortens model names to their prefix."""
    m = re.match(r"^\S+ \((\w+ \d+)\) -> \S+ \((\w+ \d+)\)$", key)
    if m:
        return f"{m.group(1)} -> {m.group(2)}"
    if " -> " in key:
        parts = key.split(" -> ")
        return " -> ".join(shorten_model_name(p) for p in parts)
    return key

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
    keys = list(data.keys())
    labels = [clean_pair_label(k) for k in keys]
    small_to_large = [data[k]["small_to_large_pct"] for k in keys]
    large_to_small = [data[k]["large_to_small_pct"] for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(16, 7))
    plt.bar(x - width / 2, small_to_large, width, label="Small -> Large")
    plt.bar(x + width / 2, large_to_small, width, label="Large -> Small")
    plt.xticks(x, labels, rotation=30, ha="right", fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylabel("Disagreement %", fontsize=26)
    plt.title("Direction of Deference", fontsize=26)
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


def plot_deference_each_round(data, fig_dir):
    """Save one bar chart per round transition into fig_dir.

    Each figure mirrors plot_deference() but is filtered to a single
    round transition.  Title: "Direction of Deference (Round N->N+1)".
    """
    pairs = [k for k in sorted(data.keys()) if data[k]]
    if not pairs:
        return

    # Collect all transition keys across all pairs so every round is covered.
    all_transitions = sorted(
        {t for pair in pairs for t in data[pair].keys()}
    )

    os.makedirs(fig_dir, exist_ok=True)

    for transition in all_transitions:
        # Parse e.g. "round_1_to_2" -> round number label "1->2"
        # transition format: round_{n}_to_{n+1}
        parts = transition.split("_")   # ['round', '1', 'to', '2']
        round_label = f"{parts[1]}->{parts[3]}"
        round_num = parts[1]

        labels = []
        s2l_vals = []
        l2s_vals = []

        for pair in pairs:
            if transition not in data[pair]:
                continue
            entry = data[pair][transition]
            labels.append(clean_pair_label(pair))
            s2l_vals.append(entry["small_to_large_pct"] or 0)
            l2s_vals.append(entry["large_to_small_pct"] or 0)

        if not labels:
            continue

        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(16, 7))
        plt.bar(x - width / 2, s2l_vals, width, label="Small -> Large")
        plt.bar(x + width / 2, l2s_vals, width, label="Large -> Small")
        plt.xticks(x, labels, rotation=30, ha="right", fontsize=26)
        plt.yticks(fontsize=26)
        plt.ylabel("Deference (%)", fontsize=26)
        plt.title(f"Direction of Deference (Round {round_label})", fontsize=26)
        plt.legend(fontsize=26, loc="upper left", bbox_to_anchor=(1.01, 1))
        plt.tight_layout(pad=1.5)

        out_path = os.path.join(fig_dir, f"deference_round_{round_num}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [deference round {round_label}] -> {out_path}")


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
    dirs_with_deference_rounds = sorted(find_dirs_with(OUTPUT_ROOT, "deference_each_round.json"))

    print(f"Found {len(dirs_with_directional)} directional files, "
          f"{len(dirs_with_metrics)} metrics files, "
          f"{len(dirs_with_deference_rounds)} deference-per-round files.\n")

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

    print(f"\nPlotting deference per round ({len(dirs_with_deference_rounds)} datasets).\n")
    for output_dir in dirs_with_deference_rounds:
        fig_dir = output_to_figures(output_dir)
        print(f"Plotting deference per round: {output_dir}")
        data = load_json(os.path.join(output_dir, "deference_each_round.json"))
        if not data:
            print("  [SKIP] Empty.\n")
            continue
        plot_deference_each_round(data, fig_dir)

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
