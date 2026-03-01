import pandas as pd
import os
import json
import glob
import math

def count_options(results_dir):
    option_counts = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "q_*.json"))):
        with open(path) as f:
            data = json.load(f)
        q_name = os.path.basename(path)
        option_counts[q_name] = len(data.get("options", []))
    return option_counts

def calculate_pairwise_baseline(options_counts):
    length = len(options_counts)
    sum_baseline = 0
    for i in range(length):
        random_baseline = (options_counts[i] - 1) / options_counts[i]
        sum_baseline = sum_baseline + random_baseline
    pairwise_baseline = sum_baseline / length
    return pairwise_baseline

def calculate_true_disagreement_baseline(options_counts):
    models = 3
    sum_baseline = 0
    length = len(options_counts)
    for i in range(length):
        random_baseline = 1 - (1 / math.pow(options_counts[i], models - 1))
        sum_baseline = sum_baseline + random_baseline
    true_disagreement_baseline = sum_baseline / length
    return true_disagreement_baseline


if __name__ == "__main__":
    results_dir = "results/GlobalOpinionsQA/agent_names/gpt-4.1-fam/agents_3_questions_2556/adversarial/named"
    option_counts = count_options(results_dir)
    pairwise_baseline = calculate_pairwise_baseline(list(option_counts.values()))
    true_disagreement_baseline = calculate_true_disagreement_baseline(list(option_counts.values()))
    length = len(option_counts)
    options_counts = list(option_counts.values())
    print(f"Total questions: {length}")
    print(f"Option Counts: {options_counts}")
    print(f"Pairwise Baseline: {pairwise_baseline:.4f}")
    print(f"True Disagreement Baseline: {true_disagreement_baseline:.4f}")
