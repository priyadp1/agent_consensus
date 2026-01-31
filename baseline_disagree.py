import matplotlib.pyplot as plt
import os

rounds = [1, 2, 3]
from check import FOLDER_NAME
data = {
    "Disagreement with single-agent ({FOLDER_NAME})": [19.53, 19.58, 18.33],
    "3-agent disagreement (3 x {FOLDER_NAME})": [14.03, 12.45, 9.24],
}

plt.figure(figsize=(8, 5))

for label, values in data.items():
    plt.plot(rounds, values, marker="o", linewidth=2, label=label)

plt.xticks(rounds)
plt.xlabel("Deliberation Round")
plt.ylabel("Questions with >= 1 Disagreement (%)")
plt.title("Baseline Disagreement vs 3-Agent Disagreement ({FOLDER_NAME})")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

output_dir = f"figures/gpt4.1-baseline"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(
    os.path.join(output_dir, f"{FOLDER_NAME}_baseline_vs_3-agent_disagreement.png"),
    dpi=300
)

plt.show()