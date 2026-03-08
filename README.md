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

| Script | Description |
|--------|-------------|
| `python sys_prompt_main.py` | Runs experiments with system prompts |
| `python change_answer_all.py` | Runs rotated experiments |
| `python main.py` | Runs baseline multi-agent experiments |

## Analyzing Results

| Script | Description |
|--------|-------------|
| `python plots/all_models_disagree.py` | Saves model disagreement rate and deference rate to `analysis_outputs/` |
| `python plots/all_plots.py` | Saves plots from `analysis_outputs/` to `figures/` |
