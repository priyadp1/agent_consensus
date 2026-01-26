import matplotlib.pyplot as plt
import os

rounds = [1, 2, 3]

data = {
    "Disagreement with single-agent (gpt-4.1-nano)": [35.62, 28.43, 25.71],
    "3-agent disagreement (3 x gpt-4.1-nano)": [28.43, 18.91, 14.03],
}

plt.figure(figsize=(8, 5))

for label, values in data.items():
    plt.plot(rounds, values, marker="o", linewidth=2, label=label)

plt.xticks(rounds)
plt.xlabel("Deliberation Round")
plt.ylabel("Questions with >= 1 Disagreement (%)")
plt.title("Baseline Disagreement vs 3-Agent Disagreement (gpt-4.1-nano)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

output_dir = "figures/gpt-4.1-nano-baseline"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(
    os.path.join(output_dir, "baseline_vs_3-agent_disagreement.png"),
    dpi=300
)

plt.show()