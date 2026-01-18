import os
import json
from collections import Counter
import matplotlib.pyplot as plt

MODEL_NAME = "gpt-4.1-nano"
NUM_AGENTS = 3
NUM_QUESTIONS = 5

RESULTS_DIR = f"results/{MODEL_NAME}/agents_{NUM_AGENTS}_questions_{NUM_QUESTIONS}"
OUT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)


def letter_to_option(letter, options):
    idx = ord(letter) - ord("A")
    if 0 <= idx < len(options):
        return options[idx]
    return "INVALID"


for fname in sorted(os.listdir(RESULTS_DIR)):
    if not fname.endswith(".json"):
        continue

    with open(os.path.join(RESULTS_DIR, fname)) as f:
        data = json.load(f)

    question = data["question"]
    options = data["options"]
    rounds = data["rounds"]

    for r_idx, round_data in enumerate(rounds, start=1):
        answers = []

        for agent_data in round_data.values():
            ans = agent_data["answer"]
            if ans != "INVALID":
                answers.append(letter_to_option(ans, options))

        if not answers:
            continue

        counts = Counter(answers)
        labels = list(counts.keys())
        sizes = list(counts.values())

        plt.figure()
        plt.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)
        plt.title(f"{question}\nRound {r_idx}")
        plt.axis("equal")

        out_path = os.path.join(
            OUT_DIR,
            fname.replace(".json", f"_round_{r_idx}.png")
        )

        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"Saved: {out_path}")