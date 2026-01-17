import json
import os
import argparse
import asyncio

from model import run_model
from multiagent import agent_talk

DATA_PATH = "data/jsonl/train.jsonl"


def stream_jsonL(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def build_prompt(example):
    return f"""
You are answering a subjective public opinion question.

Question:
{example["question"]}

Answer options:
{example["options"]}

Global response distributions by country:
{example["selections"]}

Task:
Give your own considered answer and briefly explain your reasoning.
"""


def single_agent(limit):
    results_dir = f"results/single_agent_{limit}"
    os.makedirs(results_dir, exist_ok=True)

    for q_idx, example in enumerate(stream_jsonL(DATA_PATH)):
        if q_idx >= limit:
            break

        prompt = build_prompt(example)
        answer = run_model(prompt)

        with open(os.path.join(results_dir, f"answers_{q_idx}.txt"), "w") as f:
            f.write(f"Question {q_idx}\n")
            f.write(example["question"] + "\n\n")
            f.write(answer.strip())

        print(f"Saved single-agent result for question {q_idx}")


async def multi_agent(num_agents, limit, max_rounds):
    results_dir = f"results/agents_{num_agents}_questions_{limit}"
    os.makedirs(results_dir, exist_ok=True)

    agents = list(range(num_agents))

    for q_idx, example in enumerate(stream_jsonL(DATA_PATH)):
        if q_idx >= limit:
            break

        history = await agent_talk(
            agents=agents,
            question=example["question"],
            options=example["options"],
            selections=example["selections"],
            run_model=run_model,
            max_rounds=max_rounds
        )

        with open(os.path.join(results_dir, f"answers_{q_idx}.txt"), "w") as f:
            f.write(f"Question {q_idx}\n")
            f.write(example["question"] + "\n\n")

            for r_idx, round_data in enumerate(history, start=1):
                f.write(f"\n--- ROUND {r_idx} ---\n")
                for agent_id, resp in round_data.items():
                    f.write(f"Agent {agent_id}: {resp}\n")

        print(f"Saved consensus results for question {q_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=1)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)

    args = parser.parse_args()

    if args.agents == 1:
        single_agent(args.limit)
    else:
        asyncio.run(multi_agent(args.agents, args.limit, args.rounds))
