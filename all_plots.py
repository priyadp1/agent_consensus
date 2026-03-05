import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

os.makedirs("analysis_outputs", exist_ok=True)

# ── Input path constants ──────────────────────────────────────────────────────

OUTPUT_BASE_RANDOM_SEE = "analysis_outputs/GlobalOpinionsQA/agent_names/random_models"

ROUND_JSON_RANDOM_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_RANDOM_SEE, "per_round_disagreement_NORMAL.json"
)
ROUND_JSON_RANDOM_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_RANDOM_SEE, "per_round_disagreement_ROTATED.json"
)
DIRECTIONAL_JSON_RANDOM_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_RANDOM_SEE, "random_models_NORMAL_directional.json"
)
DIRECTIONAL_JSON_RANDOM_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_RANDOM_SEE, "random_models_ROTATED_directional.json"
)

OUTPUT_BASE_GPT41_SEE = "analysis_outputs/GlobalOpinionsQA/agent_names/gpt-4.1-fam"

ROUND_JSON_GPT41_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "per_round_disagreement_NORMAL.json"
)
ROUND_JSON_GPT41_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "per_round_disagreement_ROTATED.json"
)
DIRECTIONAL_JSON_GPT41_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "gpt4.1_family_NORMAL_directional.json"
)
DIRECTIONAL_JSON_GPT41_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "gpt4.1_family_ROTATED_directional.json"
)
DIRECTIONAL_JSON_GPT41_SEE_INDEP = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "gpt4.1_family_INDEP_directional.json"
)
DIRECTIONAL_JSON_GPT41_SEE_INDEP_ANON = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "gpt4.1_family_INDEP_ANON_directional.json"
)
DIRECTIONAL_JSON_GPT41_SEE_ADVERSARIAL_ANON = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "gpt4.1_family_ADVERSARIAL_ANON_directional.json"
)
DIRECTIONAL_JSON_GPT41_SEE_ADVERSARIAL_NAMED = os.path.join(
    OUTPUT_BASE_GPT41_SEE, "gpt4.1_family_ADVERSARIAL_NAMED_directional.json"
)

OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE = "analysis_outputs/GlobalOpinionsQA/agent_names/gpt-4.1-fam/without_revise"

DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_ADV_NAMED = os.path.join(
    OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE, "gpt4.1_family_ADVERSARIAL_NAMED_directional.json"
)
DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_ADV_ANON = os.path.join(
    OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE, "gpt4.1_family_ADVERSARIAL_ANON_directional.json"
)
DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_CRIT_INDEP_NAMED = os.path.join(
    OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE, "gpt4.1_family_CRITICAL_INDEPENDENT_NAMED_directional.json"
)
DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_CRIT_INDEP_ANON = os.path.join(
    OUTPUT_BASE_GPT41_SEE_WITHOUT_REVISE, "gpt4.1_family_CRITICAL_INDEPENDENT_ANON_directional.json"
)

OUTPUT_BASE_RANDOM_NO_SEE = "analysis_outputs/GlobalOpinionsQA/random_models"

ROUND_JSON_RANDOM_NO_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_RANDOM_NO_SEE, "per_round_disagreement_NORMAL.json"
)
ROUND_JSON_RANDOM_NO_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_RANDOM_NO_SEE, "per_round_disagreement_ROTATED.json"
)
DIRECTIONAL_JSON_RANDOM_NO_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_RANDOM_NO_SEE, "random_models_NORMAL_directional.json"
)
DIRECTIONAL_JSON_RANDOM_NO_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_RANDOM_NO_SEE, "random_models_ROTATED_directional.json"
)

OUTPUT_BASE_GPT41_NO_SEE = "analysis_outputs/GlobalOpinionsQA/gpt-4.1-family"

ROUND_JSON_GPT41_NO_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_GPT41_NO_SEE, "per_round_disagreement_NORMAL.json"
)
ROUND_JSON_GPT41_NO_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_GPT41_NO_SEE, "per_round_disagreement_ROTATED.json"
)
DIRECTIONAL_JSON_GPT41_NO_SEE_NORMAL = os.path.join(
    OUTPUT_BASE_GPT41_NO_SEE, "gpt4.1_family_NORMAL_directional.json"
)
DIRECTIONAL_JSON_GPT41_NO_SEE_ROTATED = os.path.join(
    OUTPUT_BASE_GPT41_NO_SEE, "gpt4.1_family_ROTATED_directional.json"
)

# ── Metrics folder paths for rounds disagreement plots ────────────────────────

METRICS_GPT41_SEE_NORMAL   = "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/agents_3_questions_2556"
METRICS_GPT41_SEE_ROTATED  = "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/agents_3_questions_2556_rotated"
METRICS_GPT41_SEE_INDEP    = "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/critical_independent_named"
METRICS_GPT41_NO_SEE_NORMAL  = "metrics/GlobalOpinionsQA/gpt-4.1-family/agents_3_questions_2089"
METRICS_GPT41_NO_SEE_ROTATED = "metrics/GlobalOpinionsQA/gpt-4.1-family/agents_3_questions_2089_rotated"

METRICS_GPT41_SEE_INDEP_ANON       = "metrics/GlobalOpinionsQA/gpt-4.1-family/critical_independent_anonymous"
METRICS_GPT41_SEE_ADVERSARIAL_ANON = "metrics/GlobalOpinionsQA/gpt-4.1-family/adversarial_anonymous"
METRICS_GPT41_SEE_ADVERSARIAL_NAMED = "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/adversarial_named"

METRICS_WITHOUT_REVISE_GPT41_ADV_NAMED        = "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/without_revise/adversarial_named"
METRICS_WITHOUT_REVISE_GPT41_ADV_ANON         = "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/without_revise/adversarial_anonymous"
METRICS_WITHOUT_REVISE_GPT41_CRIT_INDEP_NAMED = "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/without_revise/critical_independent_named"
METRICS_WITHOUT_REVISE_GPT41_CRIT_INDEP_ANON  = "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/without_revise/critical_independent_anonymous"

METRICS_RANDOM_SEE_NORMAL   = "metrics/GlobalOpinionsQA/agent_names/random_models/agents_3_questions_2556"
METRICS_RANDOM_SEE_ROTATED  = "metrics/GlobalOpinionsQA/agent_names/random_models/agents_3_questions_2556_rotated"
METRICS_RANDOM_NO_SEE_NORMAL  = "metrics/GlobalOpinionsQA/random_models/agents_3_questions_2089"
METRICS_RANDOM_NO_SEE_ROTATED = "metrics/GlobalOpinionsQA/random_models/agents_3_questions_2089_rotated"

# ── Figure output directories ─────────────────────────────────────────────────

FIG_DIR_DEFER   = "figures/directional_metrics"
FIG_DIR_ROUNDS  = "figures/rounds_disagreement"

# ── Shared helpers ────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_model_metrics(folder):
    # Load round-by-round disagreement percentages from a metrics directory.
    # Expects <folder>/interagent_disagree.json with a "rounds" key.
    path = os.path.join(folder, "interagent_disagree.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Metrics file not found for '{folder}': {path}"
        )
    data = load_json(path)
    rounds = data["rounds"]
    return [
        rounds["round_1"]["disagreement_percentage"],
        rounds["round_2"]["disagreement_percentage"],
        rounds["round_3"]["disagreement_percentage"],
    ]


# ── Directional deference plots (from plot_defer.py) ─────────────────────────

def plot_directional_bars(data, title, filename, fig_dir=FIG_DIR_DEFER):
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

    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, filename), dpi=300)
    plt.close()
    print(f"Saved -> {os.path.join(fig_dir, filename)}")


def plot_asymmetry_ratios(data, title, filename, fig_dir=FIG_DIR_DEFER):
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

    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, filename), dpi=300)
    plt.close()
    print(f"Saved -> {os.path.join(fig_dir, filename)}")


# ── Rounds disagreement plot (from plot_rounds_disagree.py) ──────────────────

def plot_rounds_disagreement(folders, title, out_dir, out_name):
    rounds = [1, 2, 3]
    plt.figure(figsize=(12, 5))

    for folder in folders:
        values = load_model_metrics(folder)
        plt.plot(rounds, values, marker="o", label=folder)

    plt.xticks(rounds)
    plt.xlabel("Deliberation Round")
    plt.ylabel("Questions with >= 1 Disagreement (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate directional deference and rounds disagreement plots"
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=[
            "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/agents_3_questions_2556",
            "metrics/GlobalOpinionsQA/agent_names/gpt-4.1-fam/agents_3_questions_2556_rotated",
        ],
        help="Metrics folder paths for the rounds disagreement line plot"
    )
    parser.add_argument(
        "--rounds-out-dir",
        default="figures/GlobalOpinionsQA/agent_names/gpt-4.1-fam-rotated_vs_normal",
        help="Output directory for the rounds disagreement figure"
    )
    parser.add_argument(
        "--rounds-out-name",
        default="disagreement_across_rounds.png",
        help="Output filename for the rounds disagreement figure"
    )
    parser.add_argument(
        "--rounds-title",
        default="Disagreement Across Deliberation Rounds",
        help="Title for the rounds disagreement figure"
    )
    args = parser.parse_args()

    # ── Directional deference: random models, see names ───────────────────────
    rand_see_norm = load_json(DIRECTIONAL_JSON_RANDOM_SEE_NORMAL)
    rand_see_rot  = load_json(DIRECTIONAL_JSON_RANDOM_SEE_ROTATED)

    plot_directional_bars(
        rand_see_norm,
        "Direction of Disagreement (Random Models — See Names)",
        "directional_bars_random_see_NORMAL.png",
    )
    plot_directional_bars(
        rand_see_rot,
        "Direction of Disagreement (Random Models Rotated — See Names)",
        "directional_bars_random_see_ROTATED.png",
    )
    plot_asymmetry_ratios(
        rand_see_norm,
        "Asymmetric Alignment Strength (Random Models — See Names)",
        "asymmetry_ratios_random_see_NORMAL.png",
    )
    plot_asymmetry_ratios(
        rand_see_rot,
        "Asymmetric Alignment Strength (Random Models Rotated — See Names)",
        "asymmetry_ratios_random_see_ROTATED.png",
    )

    # ── Directional deference: GPT-4.1, see names ─────────────────────────────
    gpt_see_norm = load_json(DIRECTIONAL_JSON_GPT41_SEE_NORMAL)
    gpt_see_rot  = load_json(DIRECTIONAL_JSON_GPT41_SEE_ROTATED)

    plot_directional_bars(
        gpt_see_norm,
        "Direction of Disagreement (GPT-4.1 Family — See Names)",
        "directional_bars_gpt41_see_NORMAL.png",
    )
    plot_directional_bars(
        gpt_see_rot,
        "Direction of Disagreement (GPT-4.1 Family Rotated — See Names)",
        "directional_bars_gpt41_see_ROTATED.png",
    )
    plot_asymmetry_ratios(
        gpt_see_norm,
        "Asymmetric Alignment Strength (GPT-4.1 Family — See Names)",
        "asymmetry_ratios_gpt41_see_NORMAL.png",
    )
    plot_asymmetry_ratios(
        gpt_see_rot,
        "Asymmetric Alignment Strength (GPT-4.1 Family Rotated — See Names)",
        "asymmetry_ratios_gpt41_see_ROTATED.png",
    )

    # ── Directional deference: GPT-4.1, see names, independent prompt ─────────
    gpt_see_indep = load_json(DIRECTIONAL_JSON_GPT41_SEE_INDEP)

    plot_directional_bars(
        gpt_see_indep,
        "Direction of Disagreement (GPT-4.1 Family — Independent Prompt)",
        "directional_bars_gpt41_see_INDEP.png",
    )
    plot_asymmetry_ratios(
        gpt_see_indep,
        "Asymmetric Alignment Strength (GPT-4.1 Family — Independent Prompt)",
        "asymmetry_ratios_gpt41_see_INDEP.png",
    )

    # ── Directional deference: GPT-4.1, critical_independent prompt, anonymous ─
    gpt_see_indep_anon = load_json(DIRECTIONAL_JSON_GPT41_SEE_INDEP_ANON)

    plot_directional_bars(
        gpt_see_indep_anon,
        "Direction of Disagreement (GPT-4.1 — Critical-Independent, Anonymous)",
        "directional_bars_gpt41_CRITICAL_INDEP_ANONYMOUS.png",
    )
    plot_asymmetry_ratios(
        gpt_see_indep_anon,
        "Asymmetric Alignment Strength (GPT-4.1 — Critical-Independent, Anonymous)",
        "asymmetry_ratios_gpt41_CRITICAL_INDEP_ANONYMOUS.png",
    )

    # ── Directional deference: GPT-4.1, adversarial prompt, anonymous ────────
    gpt_see_adv_anon = load_json(DIRECTIONAL_JSON_GPT41_SEE_ADVERSARIAL_ANON)

    plot_directional_bars(
        gpt_see_adv_anon,
        "Direction of Disagreement (GPT-4.1 — Adversarial, Anonymous)",
        "directional_bars_gpt41_ADVERSARIAL_ANONYMOUS.png",
    )
    plot_asymmetry_ratios(
        gpt_see_adv_anon,
        "Asymmetric Alignment Strength (GPT-4.1 — Adversarial, Anonymous)",
        "asymmetry_ratios_gpt41_ADVERSARIAL_ANONYMOUS.png",
    )

    # ── Directional deference: GPT-4.1, adversarial prompt, named ────────────
    gpt_see_adv_named = load_json(DIRECTIONAL_JSON_GPT41_SEE_ADVERSARIAL_NAMED)

    plot_directional_bars(
        gpt_see_adv_named,
        "Direction of Disagreement (GPT-4.1 — Adversarial, Named)",
        "directional_bars_gpt41_ADVERSARIAL_NAMED.png",
    )
    plot_asymmetry_ratios(
        gpt_see_adv_named,
        "Asymmetric Alignment Strength (GPT-4.1 — Adversarial, Named)",
        "asymmetry_ratios_gpt41_ADVERSARIAL_NAMED.png",
    )

    # ── Directional deference: GPT-4.1, without_revise — adversarial named ───
    gpt_wo_adv_named = load_json(DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_ADV_NAMED)

    plot_directional_bars(
        gpt_wo_adv_named,
        "Direction of Disagreement (GPT-4.1 — Without Revise, Adversarial, Named)",
        "directional_bars_gpt41_WITHOUT_REVISE_ADVERSARIAL_NAMED.png",
    )
    plot_asymmetry_ratios(
        gpt_wo_adv_named,
        "Asymmetric Alignment Strength (GPT-4.1 — Without Revise, Adversarial, Named)",
        "asymmetry_ratios_gpt41_WITHOUT_REVISE_ADVERSARIAL_NAMED.png",
    )

    # ── Directional deference: GPT-4.1, without_revise — adversarial anonymous
    gpt_wo_adv_anon = load_json(DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_ADV_ANON)

    plot_directional_bars(
        gpt_wo_adv_anon,
        "Direction of Disagreement (GPT-4.1 — Without Revise, Adversarial, Anonymous)",
        "directional_bars_gpt41_WITHOUT_REVISE_ADVERSARIAL_ANONYMOUS.png",
    )
    plot_asymmetry_ratios(
        gpt_wo_adv_anon,
        "Asymmetric Alignment Strength (GPT-4.1 — Without Revise, Adversarial, Anonymous)",
        "asymmetry_ratios_gpt41_WITHOUT_REVISE_ADVERSARIAL_ANONYMOUS.png",
    )

    # ── Directional deference: GPT-4.1, without_revise — critical-independent named
    gpt_wo_ci_named = load_json(DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_CRIT_INDEP_NAMED)

    plot_directional_bars(
        gpt_wo_ci_named,
        "Direction of Disagreement (GPT-4.1 — Without Revise, Critical-Independent, Named)",
        "directional_bars_gpt41_WITHOUT_REVISE_CRITICAL_INDEP_NAMED.png",
    )
    plot_asymmetry_ratios(
        gpt_wo_ci_named,
        "Asymmetric Alignment Strength (GPT-4.1 — Without Revise, Critical-Independent, Named)",
        "asymmetry_ratios_gpt41_WITHOUT_REVISE_CRITICAL_INDEP_NAMED.png",
    )

    # ── Directional deference: GPT-4.1, without_revise — critical-independent anonymous
    gpt_wo_ci_anon = load_json(DIRECTIONAL_JSON_WITHOUT_REVISE_GPT41_CRIT_INDEP_ANON)

    plot_directional_bars(
        gpt_wo_ci_anon,
        "Direction of Disagreement (GPT-4.1 — Without Revise, Critical-Independent, Anonymous)",
        "directional_bars_gpt41_WITHOUT_REVISE_CRITICAL_INDEP_ANONYMOUS.png",
    )
    plot_asymmetry_ratios(
        gpt_wo_ci_anon,
        "Asymmetric Alignment Strength (GPT-4.1 — Without Revise, Critical-Independent, Anonymous)",
        "asymmetry_ratios_gpt41_WITHOUT_REVISE_CRITICAL_INDEP_ANONYMOUS.png",
    )

    # ── Directional deference: random models, no see names ────────────────────
    rand_no_norm = load_json(DIRECTIONAL_JSON_RANDOM_NO_SEE_NORMAL)
    rand_no_rot  = load_json(DIRECTIONAL_JSON_RANDOM_NO_SEE_ROTATED)

    plot_directional_bars(
        rand_no_norm,
        "Direction of Disagreement (Random Models — No See Names)",
        "directional_bars_random_no_see_NORMAL.png",
    )
    plot_directional_bars(
        rand_no_rot,
        "Direction of Disagreement (Random Models Rotated — No See Names)",
        "directional_bars_random_no_see_ROTATED.png",
    )
    plot_asymmetry_ratios(
        rand_no_norm,
        "Asymmetric Alignment Strength (Random Models — No See Names)",
        "asymmetry_ratios_random_no_see_NORMAL.png",
    )
    plot_asymmetry_ratios(
        rand_no_rot,
        "Asymmetric Alignment Strength (Random Models Rotated — No See Names)",
        "asymmetry_ratios_random_no_see_ROTATED.png",
    )

    # ── Directional deference: GPT-4.1, no see names ─────────────────────────
    gpt_no_norm = load_json(DIRECTIONAL_JSON_GPT41_NO_SEE_NORMAL)
    gpt_no_rot  = load_json(DIRECTIONAL_JSON_GPT41_NO_SEE_ROTATED)

    plot_directional_bars(
        gpt_no_norm,
        "Direction of Disagreement (GPT-4.1 Family — No See Names)",
        "directional_bars_gpt41_no_see_NORMAL.png",
    )
    plot_directional_bars(
        gpt_no_rot,
        "Direction of Disagreement (GPT-4.1 Family Rotated — No See Names)",
        "directional_bars_gpt41_no_see_ROTATED.png",
    )
    plot_asymmetry_ratios(
        gpt_no_norm,
        "Asymmetric Alignment Strength (GPT-4.1 Family — No See Names)",
        "asymmetry_ratios_gpt41_no_see_NORMAL.png",
    )
    plot_asymmetry_ratios(
        gpt_no_rot,
        "Asymmetric Alignment Strength (GPT-4.1 Family Rotated — No See Names)",
        "asymmetry_ratios_gpt41_no_see_ROTATED.png",
    )

    print(f"\nDirectional figures saved to: {FIG_DIR_DEFER}")

    # ── Rounds disagreement: GPT-4.1, see names ───────────────────────────────
    plot_rounds_disagreement(
        folders=[METRICS_GPT41_SEE_NORMAL, METRICS_GPT41_SEE_ROTATED],
        title="Disagreement Across Rounds (GPT-4.1 — See Names)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_gpt41_see_normal_vs_rotated.png",
    )

    # ── Rounds disagreement: GPT-4.1, see names — normal vs independent prompt ─
    plot_rounds_disagreement(
        folders=[METRICS_GPT41_SEE_NORMAL, METRICS_GPT41_SEE_INDEP],
        title="Disagreement Across Rounds (GPT-4.1 — See Names: Normal vs Independent Prompt)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_gpt41_see_normal_vs_indep.png",
    )

    # ── Rounds disagreement: GPT-4.1, no see names ───────────────────────────
    plot_rounds_disagreement(
        folders=[METRICS_GPT41_NO_SEE_NORMAL, METRICS_GPT41_NO_SEE_ROTATED],
        title="Disagreement Across Rounds (GPT-4.1 — No See Names)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_gpt41_no_see_normal_vs_rotated.png",
    )

    # ── Rounds disagreement: random models, see names ─────────────────────────
    plot_rounds_disagreement(
        folders=[METRICS_RANDOM_SEE_NORMAL, METRICS_RANDOM_SEE_ROTATED],
        title="Disagreement Across Rounds (Random Models — See Names)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_random_see_normal_vs_rotated.png",
    )

    # ── Rounds disagreement: random models, no see names ─────────────────────
    plot_rounds_disagreement(
        folders=[METRICS_RANDOM_NO_SEE_NORMAL, METRICS_RANDOM_NO_SEE_ROTATED],
        title="Disagreement Across Rounds (Random Models — No See Names)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_random_no_see_normal_vs_rotated.png",
    )

    # ── Rounds disagreement: GPT-4.1 — normal anonymous vs critical-independent anonymous ─
    plot_rounds_disagreement(
        folders=[METRICS_GPT41_NO_SEE_NORMAL, METRICS_GPT41_SEE_INDEP_ANON],
        title="Disagreement Across Rounds (GPT-4.1 — Normal Anon vs Critical-Independent Anon)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_gpt41_NORMAL_vs_CRITICAL_INDEP_ANONYMOUS.png",
    )

    # ── Rounds disagreement: GPT-4.1 — critical-independent named vs anonymous ─
    plot_rounds_disagreement(
        folders=[METRICS_GPT41_SEE_INDEP, METRICS_GPT41_SEE_INDEP_ANON],
        title="Disagreement Across Rounds (GPT-4.1 — Critical-Independent: Named vs Anonymous)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_gpt41_CRITICAL_INDEP_NAMED_vs_ANONYMOUS.png",
    )

    # ── Rounds disagreement: GPT-4.1 — normal anonymous vs adversarial anonymous ─
    plot_rounds_disagreement(
        folders=[METRICS_GPT41_NO_SEE_NORMAL, METRICS_GPT41_SEE_ADVERSARIAL_ANON],
        title="Disagreement Across Rounds (GPT-4.1 — Normal Anon vs Adversarial Anon)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_gpt41_NORMAL_vs_ADVERSARIAL_ANONYMOUS.png",
    )

    # ── Rounds disagreement: GPT-4.1 — normal vs adversarial named ───────────
    plot_rounds_disagreement(
        folders=[METRICS_GPT41_SEE_NORMAL, METRICS_GPT41_SEE_ADVERSARIAL_NAMED],
        title="Disagreement Across Rounds (GPT-4.1 — Normal vs Adversarial Named)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_gpt41_NORMAL_vs_ADVERSARIAL_NAMED.png",
    )

    # ── Rounds disagreement: GPT-4.1 — adversarial named vs anonymous ────────
    plot_rounds_disagreement(
        folders=[METRICS_GPT41_SEE_ADVERSARIAL_NAMED, METRICS_GPT41_SEE_ADVERSARIAL_ANON],
        title="Disagreement Across Rounds (GPT-4.1 — Adversarial: Named vs Anonymous)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_gpt41_ADVERSARIAL_NAMED_vs_ANONYMOUS.png",
    )

    # ── Rounds disagreement: without_revise vs with_revise — adversarial named ─
    plot_rounds_disagreement(
        folders=[METRICS_GPT41_SEE_ADVERSARIAL_NAMED, METRICS_WITHOUT_REVISE_GPT41_ADV_NAMED],
        title="Disagreement Across Rounds (GPT-4.1 — Adversarial Named: With vs Without Revise)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_gpt41_ADVERSARIAL_NAMED_with_vs_without_revise.png",
    )

    # ── Rounds disagreement: without_revise vs with_revise — adversarial anonymous
    plot_rounds_disagreement(
        folders=[METRICS_GPT41_SEE_ADVERSARIAL_ANON, METRICS_WITHOUT_REVISE_GPT41_ADV_ANON],
        title="Disagreement Across Rounds (GPT-4.1 — Adversarial Anonymous: With vs Without Revise)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_gpt41_ADVERSARIAL_ANON_with_vs_without_revise.png",
    )

    # ── Rounds disagreement: without_revise vs with_revise — critical-independent named
    plot_rounds_disagreement(
        folders=[METRICS_GPT41_SEE_INDEP, METRICS_WITHOUT_REVISE_GPT41_CRIT_INDEP_NAMED],
        title="Disagreement Across Rounds (GPT-4.1 — Critical-Independent Named: With vs Without Revise)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_gpt41_CRITICAL_INDEP_NAMED_with_vs_without_revise.png",
    )

    # ── Rounds disagreement: without_revise vs with_revise — critical-independent anonymous
    plot_rounds_disagreement(
        folders=[METRICS_GPT41_SEE_INDEP_ANON, METRICS_WITHOUT_REVISE_GPT41_CRIT_INDEP_ANON],
        title="Disagreement Across Rounds (GPT-4.1 — Critical-Independent Anonymous: With vs Without Revise)",
        out_dir=FIG_DIR_ROUNDS,
        out_name="rounds_gpt41_CRITICAL_INDEP_ANON_with_vs_without_revise.png",
    )

    print(f"\nRounds figures saved to: {FIG_DIR_ROUNDS}")
