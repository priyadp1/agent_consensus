import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import textwrap

MODEL_NAME = "gpt-4.1-nano"
NUM_AGENTS = 3
NUM_QUESTIONS = 2088

RESULTS_DIR = f"results/{MODEL_NAME}/agents_{NUM_AGENTS}_questions_{NUM_QUESTIONS}"
PLOTS_ROOT = f"results/{MODEL_NAME}/plots"
os.makedirs(PLOTS_ROOT, exist_ok=True)


def letter_to_option(letter, options):
    idx = ord(letter) - ord("A")
    if 0 <= idx < len(options):
        return options[idx]
    return None


def extract_q_index(fname):
    return int(fname.replace("q_", "").replace(".json", ""))


files = sorted(
    [f for f in os.listdir(RESULTS_DIR) if f.startswith("q_") and f.endswith(".json")],
    key=extract_q_index
)


for fname in files:
    q_idx = extract_q_index(fname)
    q_plot_dir = os.path.join(PLOTS_ROOT, f"q_{q_idx}")
    os.makedirs(q_plot_dir, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, fname)) as f:
        data = json.load(f)

    question = data["question"]
    options = data["options"]
    rounds = data["rounds"]

    wrapped_question = "\n".join(textwrap.wrap(question, width=80))

    for r_idx, round_data in enumerate(rounds, start=1):
        answers = []

        for agent_data in round_data.values():
            ans = agent_data.get("answer")
            if ans and ans != "INVALID":
                opt = letter_to_option(ans, options)
                if opt:
                    answers.append(opt)

        if not answers:
            continue

        counts = Counter(answers)
        labels = list(counts.keys())
        sizes = list(counts.values())

        plt.figure()
        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.0f%%",
            startangle=90
        )
        plt.title(f"{wrapped_question}\nRound {r_idx}", fontsize=10)
        plt.axis("equal")

        out_path = os.path.join(q_plot_dir, f"round_{r_idx}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"[SAVED] {out_path}")