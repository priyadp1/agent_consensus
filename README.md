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
| `python experiments/rotate_main.py` | `rotate_configs/` | Rotated experiments — cyclically swaps round-1 answers between agents to test position/order effects |

Each config file targets a specific dataset + model family combination (e.g. `hle-gpt-4.1-fam.yaml`). Results are saved under `results/` in subdirectories organized by dataset and config.

**Supported datasets:** GlobalOpinionsQA, OpinionsQA, anthropic-persona, Humanity's Last Exam (HLE)

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

## Repository Structure

```
agent_consensus/
├── Analysis_files/                        # Analysis scripts
│   ├── all_models_disagree.py             # Computes disagreement and deference metrics across all results
│   ├── all_plots.py                       # Generates disagreement and deference plots
│   ├── HLE_accuracy.py                    # Computes accuracy metrics for HLE results
│   └── HLE_accuracy_plot.py               # Generates accuracy and deference plots for HLE
│
├── agents/                                # Multi-agent conversation orchestration
│   ├── multiagent.py                      # Agent talk loop used by baseline and sys_prompt experiments
│   ├── multiagent2.py                     # Agent talk loop variant used by no-revise experiments
│   └── multiagent_rotate.py               # Agent talk loop for rotation experiments
│
├── baseline_configs_20/                   # Configs for experiments/main.py
│   ├── globalqa-gpt-4.1-fam.yaml
│   ├── hle-gpt-4.1-fam.yaml
│   ├── opinionsqa-gpt-4.1-fam.yaml
│   └── persona-gpt-4.1-fam.yaml
│
├── calculations/
│   └── random_baselines.py                # Computes random baseline agreement rates
│
├── data/                                  # Raw and preprocessed datasets
│   ├── jsonl/                             # Preprocessed JSONL files used by experiments
│   ├── evals/                             # Cloned Anthropic evals repo (persona dataset source)
│   ├── Anthropic___llm_global_opinions/   # Cached HuggingFace GlobalOpinionsQA dataset
│   ├── cais___hle/                        # Cached HuggingFace HLE dataset
│   └── timchen0618___opinion_qa/          # Cached HuggingFace OpinionsQA dataset
│
├── experiments/                           # Experiment runner scripts
│   ├── main.py                            # Baseline multi-agent experiments
│   ├── sys_prompt_main.py                 # System prompt experiments (agents revise answers)
│   ├── sys_prompt_no_revise.py            # System prompt experiments (no answer revision)
│   └── rotate_main.py                     # Rotation experiments (swaps round-1 answers between agents)
│
├── miscellaneous/                         # Archived configs, old outputs, and utility scripts
│   ├── convert_to_zip.py                  # Zips results directory
│   ├── extract_zip.py                     # Extracts zipped results
│   └── ...                               # Old configs, figures, and metrics from earlier runs
│
├── models/                                # Model API wrappers
│   ├── model.py                           # Azure API wrapper (used by baseline experiments)
│   └── model2.py                          # Azure API wrapper with system prompt support
│
├── new_analysis_outputs/                  # JSON outputs from analysis scripts
│   ├── GlobalOpinionsQA/
│   ├── HLE/
│   └── anthropic-persona/
│
├── new_figures/                           # Plots generated by analysis scripts
│   ├── GlobalOpinionsQA/
│   ├── HLE/
│   └── anthropic-persona/
│
├── preprocessing/                         # Data download and preprocessing scripts
│   ├── download_dataset.py                # Downloads OpinionsQA and GlobalOpinionsQA from HuggingFace
│   ├── filter_questions.py                # Filters out invalid or malformed questions
│   ├── preprocess_hle.py                  # Converts HLE dataset to JSONL
│   ├── preprocess_opinionsqa.py           # Converts OpinionsQA and GlobalOpinionsQA to JSONL
│   └── preprocess_persona.py             # Converts Anthropic persona dataset to JSONL
│
├── results/                               # Experiment outputs (one JSON file per question)
│   ├── GlobalOpinionsQA/
│   ├── HLE/
│   ├── OpinionsQA/
│   └── anthropic-persona/
│
├── rotate_configs/                        # Configs for experiments/rotate_main.py
│   ├── gpt-4.1-fam-rotate-global.yaml
│   ├── gpt-4.1-fam-rotate-hle.yaml
│   ├── gpt-4.1-fam-rotate-opinions.yaml
│   └── gpt-4.1-fam-rotate-persona.yaml
│
├── sys_prompt_configs_20/                 # Configs for experiments/sys_prompt_main.py (with revise)
│   ├── anthropic-persona-gpt-4.1-fam.yaml
│   ├── globalqa-gpt-4.1-fam.yaml
│   ├── hle-gpt-4.1-fam.yaml
│   └── opinionsqa-gpt-4.1-fam.yaml
│
├── sys_prompt_configs_no_revise_20/       # Configs for experiments/sys_prompt_no_revise.py
│   ├── anthropic-persona-gpt-4.1-fam.yaml
│   ├── globalqa-gpt-4.1-fam.yaml
│   ├── hle-gpt-4.1-fam.yaml
│   └── opinionsqa-gpt-4.1-fam.yaml
│
├── tests/                                 # Unit tests
│   ├── test_accuracy.py                   # Tests for accuracy computation
│   ├── test_disagree.py                   # Tests for disagreement computation
│   └── test_models.py                     # Tests for model API wrappers
│
├── .env                                   # Azure API credentials (not committed)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```
