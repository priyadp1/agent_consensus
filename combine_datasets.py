# Combine OpinionsQA test and validation splits
input_files = [
    "data/jsonl/OpinionsQA/test.jsonl",
    "data/jsonl/OpinionsQA/validation.jsonl",
]
output_file = "data/jsonl/OpinionsQA/combined.jsonl"

with open(output_file, "w") as out_f:
    for path in input_files:
        with open(path) as in_f:
            for line in in_f:
                out_f.write(line)

print(f"Combined {len(input_files)} files into {output_file}")
