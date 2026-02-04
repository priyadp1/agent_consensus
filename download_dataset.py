from datasets import load_dataset
import os

#dataset = load_dataset(
    # "Anthropic/llm_global_opinions",
    # cache_dir="./data"
#)

#print(dataset)

#os.makedirs("data/jsonl", exist_ok=True)

#for split, ds in dataset.items():
    #ds.to_json(f"data/jsonl/{split}.jsonl")


ds = load_dataset(
    "timchen0618/OpinionQA",
    cache_dir="./data"
)
print(ds)
os.makedirs("data/jsonl", exist_ok=True)
for split, dset in ds.items():
    dset.to_json(f"data/jsonl/{split}.jsonl")