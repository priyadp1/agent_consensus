import json
from datasets import load_dataset, get_dataset_config_names
import os
from huggingface_hub import snapshot_download
import glob

# Check and download llm_global_opinions
dataset_dir = "idk/data/jsonl/GlobalOpinionsQA"
if os.path.exists(dataset_dir) and os.listdir(dataset_dir):
    print("Dataset already exists: llm_global_opinions")
else:
    global_opinions_ds = load_dataset(
        "Anthropic/llm_global_opinions",
        cache_dir="./data"
    )
    print(global_opinions_ds)
    os.makedirs(dataset_dir, exist_ok=True)
    for split, split_data in global_opinions_ds.items():
        split_data.to_json(f"{dataset_dir}/{split}.jsonl")


# Check and download OpinionQA
dataset_dir = "idk/data/jsonl/OpinionsQA"
if os.path.exists(dataset_dir) and os.listdir(dataset_dir):
    print("Dataset already exists: OpinionQA")
else:
    opinion_qa_ds = load_dataset(
        "timchen0618/OpinionQA",
        cache_dir="./data"
    )
    print(opinion_qa_ds)
    os.makedirs(dataset_dir, exist_ok=True)
    for split, split_data in opinion_qa_ds.items():
        split_data.to_json(f"{dataset_dir}/{split}.jsonl")

# Check and download hle
dataset_dir = "data/jsonl/hle"
if os.path.exists(dataset_dir) and os.listdir(dataset_dir):
    print("Dataset already exists: hle")
else:
    hle_ds = load_dataset(
        "cais/hle",
        split = "test",
        cache_dir="./data"
    )
    print(hle_ds)
    os.makedirs(dataset_dir, exist_ok=True)
    for split, split_data in hle_ds.items():
        path = f"{dataset_dir}/{split}.jsonl"
        with open(path, "w") as f:
            for item in split_data:
                json.dump(item, f)
                f.write("\n")