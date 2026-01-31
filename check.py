import os
import json
FOLDER_NAME="DeepSeek-R1"
RESULTS_DIR = f"results/{FOLDER_NAME}/agents_3_questions_2556"

def round1_disagree(data):
    rounds = data.get("rounds", [])
    if not rounds:
        return False

    round1 = rounds[0]
    answers = [
        agent_data.get("answer")
        for agent_data in round1.values()
        if agent_data.get("answer") not in (None, "INVALID")
    ]

    if len(answers) < 2:
        return False

    return len(set(answers)) > 1

def round2_disagree(data):
    rounds = data.get("rounds", [])
    if not rounds:
        return False

    round2 = rounds[1]
    answers = [
        agent_data.get("answer")
        for agent_data in round2.values()
        if agent_data.get("answer") not in (None, "INVALID")
    ]

    if len(answers) < 2:
        return False

    return len(set(answers)) > 1

def round3_disagree(data):
    rounds = data.get("rounds", [])
    if not rounds:
        return False

    round3 = rounds[2]
    answers = [
        agent_data.get("answer")
        for agent_data in round3.values()
        if agent_data.get("answer") not in (None, "INVALID")
    ]

    if len(answers) < 2:
        return False

    return len(set(answers)) > 1


def main():
    total = 0
    disagreement1 = 0
    disagreement2 = 0
    disagreement3 = 0

    for fname in os.listdir(RESULTS_DIR):
        if not fname.startswith("q_") or not fname.endswith(".json"):
            continue

        path = os.path.join(RESULTS_DIR, fname)
        with open(path, "r") as f:
            data = json.load(f)

        total += 1
        if round1_disagree(data):
            disagreement1 += 1
        if round2_disagree(data):
            disagreement2 += 1
        if round3_disagree(data):
            disagreement3 += 1

    if total == 0:
        print("No result files found.")
        return

    pct1 = 100 * disagreement1 / total
    pct2 = 100 * disagreement2 / total
    pct3 = 100 * disagreement3 / total
    print(f"Results for model: {FOLDER_NAME}")
    print(f"Total questions analyzed: {total}")
    print(f"Questions with disagreement in round 1: {disagreement1}")
    print(f"Questions with disagreement in round 2: {disagreement2}")
    print(f"Questions with disagreement in round 3: {disagreement3}")
    print(f"Percentage with round-1 disagreement: {pct1:.2f}%")
    print(f"Percentage with round-2 disagreement: {pct2:.2f}%")
    print(f"Percentage with round-3 disagreement: {pct3:.2f}%")


if __name__ == "__main__":
    main()