import os
import json

SINGLE_AGENT_DIR = "results/gpt-4.1-nano/single_agent_2089"
MULTI_AGENT_DIR = "results/gpt-4.1-nano/agents_3_questions_2089"

single_answers = {}
for fname in os.listdir(SINGLE_AGENT_DIR):
    if not fname.startswith("q_") or not fname.endswith(".json"):
        continue
    qid = int(fname[2:-5])
    with open(os.path.join(SINGLE_AGENT_DIR, fname)) as f:
        data = json.load(f)
    single_answers[qid] = data["answer"]

total = 0
disagree = [0, 0, 0]

for fname in os.listdir(MULTI_AGENT_DIR):
    if not fname.startswith("q_") or not fname.endswith(".json"):
        continue
    qid = int(fname[2:-5])
    if qid not in single_answers:
        continue

    with open(os.path.join(MULTI_AGENT_DIR, fname)) as f:
        data = json.load(f)

    baseline = single_answers[qid]
    rounds = data["rounds"]

    total += 1

    for r in range(min(3, len(rounds))):
        answers = [
            agent_data["answer"]
            for agent_data in rounds[r].values()
        ]
        if any(a != baseline for a in answers):
            disagree[r] += 1

print(f"Total questions analyzed: {total}")
for i in range(3):
    pct = 100 * disagree[i] / total if total > 0 else 0
    print(f"Round {i+1} disagreement vs single-agent: {disagree[i]} ({pct:.2f}%)")