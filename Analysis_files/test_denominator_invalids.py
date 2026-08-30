import os
from collections import defaultdict

from all_models_disagree import (
    RESULTS_ROOT,
    MODEL_ORDER,
    find_leaf_dirs,
    load_results,
    get_answer,
)


def count_invalid_current_round(data):
    # denom_total: total denominator increments (pair disagreed in round n-1, both valid)
    # denom_invalid_curr: subset of those where the current-round (n) observation was INVALID
    #                      for at least one side of the pair
    denom_total = defaultdict(int)
    denom_invalid_curr = defaultdict(int)

    for item in data:
        agent_models = item.get("agent_models", {})
        agent_items = list(agent_models.items())
        rounds = item["rounds"]

        for i, (label_a, m_a) in enumerate(agent_items):
            for label_b, m_b in agent_items[i + 1:]:
                key_a = (MODEL_ORDER.get(m_a, -1), label_a)
                key_b = (MODEL_ORDER.get(m_b, -1), label_b)
                if key_a == key_b:
                    continue
                if key_a < key_b:
                    label_small, m_small, label_large, m_large = label_a, m_a, label_b, m_b
                else:
                    label_small, m_small, label_large, m_large = label_b, m_b, label_a, m_a

                key = (
                    f"{m_small} ({label_small}) -> {m_large} ({label_large})"
                    if m_small == m_large
                    else f"{m_small} -> {m_large}"
                )

                for n in range(1, len(rounds)):
                    r_prev = rounds[n - 1]
                    r_curr = rounds[n]

                    k_small_prev = m_small if m_small in r_prev else label_small
                    k_large_prev = m_large if m_large in r_prev else label_large
                    a_prev = get_answer(r_prev, k_small_prev)
                    b_prev = get_answer(r_prev, k_large_prev)

                    if a_prev == "INVALID" or b_prev == "INVALID" or a_prev == b_prev:
                        continue

                    denom_total[key] += 1

                    k_small_curr = m_small if m_small in r_curr else label_small
                    k_large_curr = m_large if m_large in r_curr else label_large
                    a_curr = get_answer(r_curr, k_small_curr)
                    b_curr = get_answer(r_curr, k_large_curr)

                    if a_curr == "INVALID" or b_curr == "INVALID":
                        denom_invalid_curr[key] += 1

    return denom_total, denom_invalid_curr


def main():
    leaf_dirs = sorted(find_leaf_dirs(RESULTS_ROOT))
    if not leaf_dirs:
        print(f"No result directories found under '{RESULTS_ROOT}'.")
        return

    grand_total = 0
    grand_invalid = 0

    for results_dir in leaf_dirs:
        data = load_results(results_dir)
        if not data:
            continue

        denom_total, denom_invalid_curr = count_invalid_current_round(data)

        dir_total = sum(denom_total.values())
        dir_invalid = sum(denom_invalid_curr.values())
        if dir_total == 0:
            continue

        grand_total += dir_total
        grand_invalid += dir_invalid

        print(f"\n{results_dir}")
        print(f"  denominator total: {dir_total}, invalid-current-round: {dir_invalid} "
              f"({100 * dir_invalid / dir_total:.2f}%)")
        for key in sorted(denom_total.keys()):
            t = denom_total[key]
            inv = denom_invalid_curr[key]
            pct = 100 * inv / t if t > 0 else 0
            print(f"    {key}: {inv}/{t} ({pct:.2f}%)")

    print("\n=== GRAND TOTAL ===")
    if grand_total > 0:
        print(f"denominator total: {grand_total}, invalid-current-round: {grand_invalid} "
              f"({100 * grand_invalid / grand_total:.2f}%)")
    else:
        print("No transitions found.")


if __name__ == "__main__":
    main()
