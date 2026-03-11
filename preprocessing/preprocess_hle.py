import json
import re
import os

INPUT_PATH = "data/jsonl/hle/test.jsonl"
OUTPUT_PATH = "data/jsonl/hle/test_processed.jsonl"


def parse_choices(question_text):
    # Split on "Answer Choices:" (case-insensitive) to separate question from options block
    parts = re.split(r"\nAnswer Choices:\s*\n", question_text, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None, None

    clean_question = parts[0].strip()
    choices_block = parts[1].strip()

    # Match lines like "A. option text" or "A) option text"
    matches = re.findall(r"^[A-Z][.)]\s+(.+)$", choices_block, re.MULTILINE)
    if len(matches) < 2:
        return None, None

    return clean_question, matches


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    total = skipped = written = 0
    with open(INPUT_PATH) as fin, open(OUTPUT_PATH, "w") as fout:
        for line in fin:
            total += 1
            item = json.loads(line)

            if item.get("answer_type") != "multipleChoice":
                skipped += 1
                continue

            clean_question, options = parse_choices(item["question"])
            if clean_question is None:
                skipped += 1
                continue

            out = {
                "question": clean_question,
                "options": options,
                "answer": item["answer"],
                "category": item.get("category", ""),
                "raw_subject": item.get("raw_subject", ""),
                "id": item["id"],
            }
            fout.write(json.dumps(out) + "\n")
            written += 1

    print(f"Total: {total} | Written: {written} | Skipped: {skipped}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
