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
python preprocessing/preprocess_opinionsqa.py
python preprocessing/preprocess_persona.py
python preprocessing/preprocess_hle.py
```

## Running Experiments

All scripts should be run from the project root.

| Script | Config Dir | Description |
|--------|-----------|-------------|
| `python experiments/main.py` | `baseline_configs_20/` | Baseline multi-agent experiments — runs **named** and **anonymous** variants across GlobalOpinionsQA, OpinionsQA, and persona datasets |
| `python experiments/sys_prompt_main.py` | `sys_prompt_configs_20/` | System prompt ablations — runs `critical_independent` and `adversarial` conditions, each with named and anonymous variants |
| `python experiments/sys_prompt_no_revise.py` | `sys_prompt_configs_no_revise_20/` | Same as above but agents do not revise answers between rounds |
| `python experiments/change_answer_all.py` | `rotate_configs/` | Rotated experiments — cyclically swaps round-1 answers between agents to test position/order effects |

Each config file targets a specific dataset + model family combination (e.g. `hle-gpt-4.1-fam.yaml`). Results are saved under `results/` in subdirectories organized by dataset and config.

**Supported datasets:** GlobalOpinionsQA, OpinionsQA, anthropic-persona, HLE

## Analyzing Results

### Disagreement Analysis

| Script | Description |
|--------|-------------|
| `python Analysis_files/all_models_disagree.py` | Walks all `results/` subdirectories and computes per-round disagreement rate, pairwise disagreement, and directional deference. Saves JSON outputs to `new_analysis_outputs/`. |
| `python Analysis_files/all_plots.py` | Reads from `new_analysis_outputs/` and saves disagreement and deference plots to `new_figures/`. |

### HLE Accuracy Analysis

| Script | Description |
|--------|-------------|
| `python Analysis_files/HLE_accuracy.py` | Walks all `results/HLE/` subdirectories and computes three metrics per directory (see below). Saves JSON outputs to `new_analysis_outputs/HLE/`. |
| `python Analysis_files/HLE_accuracy_plot.py` | Reads from `new_analysis_outputs/HLE/` and saves three plots per directory to `new_figures/HLE/` (see below). |

`HLE_accuracy.py` computes the following metrics and saves them per directory:

| Output file | Metric |
|-------------|--------|
| `hle_accuracy.json` | Per-round accuracy — a question is correct only if **all** models answered correctly |
| `hle_pairwise_accuracy.json` | Per model pair, per round — questions where **both** models in the pair answered correctly |
| `hle_deference_accuracy.json` | Among R1 disagreements, tracks direction of deference (`s2l`, `l2s`) and accuracy outcomes (`wrong→correct`, `correct→wrong`) broken down separately for each branch |

`HLE_accuracy_plot.py` generates the following plots per directory:

| Output file | Plot |
|-------------|------|
| `rounds_accuracy.png` | Accuracy across deliberation rounds |
| `deference_direction.png` | Grouped bars showing Small→Large vs Large→Small deference rates per model pair |
| `deference_accuracy_outcomes.png` | Four bars per pair showing `wrong→correct` and `correct→wrong` rates for both `s2l` and `l2s` branches |
