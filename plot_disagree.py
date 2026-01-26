import matplotlib.pyplot as plt
import os

rounds = [1, 2, 3]

data = {
    "Mixed GPT-4.1 family": [38.92, 23.12, 19.15],
    "gpt-4.1-nano": [28.43, 18.91, 14.03],
    "gpt-4.1-mini": [20.78, 16.18, 12.64],
    "gpt-4.1": [14.03, 12.45, 9.24],
}

plt.figure(figsize=(8, 5))

for label, values in data.items():
    plt.plot(rounds, values, marker="o", label=label)

plt.xticks(rounds)
plt.xlabel("Deliberation Round")
plt.ylabel("Questions with >= 1 Disagreement (%)")
plt.title("Disagreement Across Deliberation Rounds")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

os.makedirs("figures/gpt-4.1-family/", exist_ok=True)
plt.savefig("figures/gpt-4.1-family/disagreement_across_rounds.png", dpi=300)
plt.show()