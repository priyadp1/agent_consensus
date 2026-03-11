# Combine OpinionsQA test and validation splits
import json
import os
from glob import glob


input_files = []
output_file = "data/jsonl/anthropic-evals/persona_combined.jsonl"

for config_path in glob("data/evals/persona/*.jsonl"):
    input_files.append(config_path)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as out_f:
    for path in input_files:
        with open(path) as in_f:
            for line in in_f:
                record = json.loads(line)
                record["options"] = ["Yes", "No"]
                out_f.write(json.dumps(record) + "\n")

print(f"Combined {len(input_files)} files into {output_file}")
