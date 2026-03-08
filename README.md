# Agent Consensus

A multi-agent framework for studying consensus and opinion dynamics across language models.

## Prerequisites

- Python 3.10+

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root with your Azure credentials:

```
AZURE_ENDPOINT=your_azure_endpoint
AZURE_API_KEY=your_azure_api_key
```

## Data

### Download Datasets

**Windows:**
```bash
python preprocessing/download_dataset.py
```

**Mac/Linux:**
```bash
python3 preprocessing/download_dataset.py
```

This downloads the OpinionsQA and GlobalOpinionsQA datasets.

To download the Anthropic dataset, clone the evals repo into the `data/` folder:

```bash
cd data
git clone https://github.com/anthropics/evals
```

### Preprocess Datasets

```bash
python combine_opinionsqa_datasets.py
python combine_persona_datasets.py
```

## Running Experiments

All scripts should be run from the `experiments/` directory.

| Script | Config Dir | Description |
|--------|-----------|-------------|
| `python main.py` | `baseline_configs_20/` | Baseline multi-agent experiments — runs **named** and **anonymous** variants across GlobalOpinionsQA, OpinionsQA, and persona datasets |
| `python sys_prompt_main.py` | `sys_prompt_configs_20/` | System prompt ablations — runs `critical_independent` and `adversarial` conditions, each with named and anonymous variants |
| `python sys_prompt_no_revise.py` | `sys_prompt_configs_no_revise_20/` | Same as above but agents do not revise answers between rounds |
| `python change_answer_all.py` | `rotate_configs/` | Rotated experiments — cyclically swaps round-1 answers between agents to test position/order effects |

Each config file targets a specific dataset + model family combination (e.g. `globalqa-gpt-4.1-fam.yaml`). Results are saved under `results/` in subdirectories determined by each config's `results_root`.

## Analyzing Results

| Script | Description |
|--------|-------------|
| `python plots/all_models_disagree.py` | Saves model disagreement rate and deference rate to `analysis_outputs/` |
| `python plots/all_plots.py` | Saves plots from `analysis_outputs/` to `figures/` |
