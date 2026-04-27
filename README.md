# Agent Consensus

A multi-agent framework for studying consensus and model deference across varying experimental settings. Three agents deliberate over survey and factual questions across multiple rounds, allowing analysis of disagreement, deference, and accuracy as a function of model size and identity visibility. Experiments use either the GPT-4.1 family (`GPT-4.1-nano`, `GPT-4.1-mini`, `GPT-4.1`) or a random family of frontier models (`Mistral-Large-3`, `Llama-4-Maverick-17B-128E-Instruct-FP8`, `Phi-4`).

---

## Prerequisites

- Python 3.10+
- Azure OpenAI credentials

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```
AZURE_ENDPOINT=your_azure_endpoint
AZURE_API_KEY=your_azure_api_key
HF_TOKEN=your_huggingface_token
```

---

## Data

### Download Datasets

```bash
python preprocessing/download_dataset.py
```

Downloads GlobalOpinionsQA from HuggingFace. To also use the Anthropic persona dataset, clone the evals repo into `data/`:

```bash
cd data
git clone https://github.com/anthropics/evals
```

### Preprocess Datasets

```bash
python preprocessing/preprocess_persona.py
python preprocessing/preprocess_hle.py
```

---

## Running Experiments

All scripts should be run from the project root. Each experiment iterates over all `*.yaml` configs in its config directory, running **named** and **anonymous** variants for each dataset.

| Script | Config Dir | Agents | Description |
|--------|------------|--------|-------------|
| `python experiments/main.py` | `baseline_configs_20/` | Random family (`Mistral-Large-3`, `Llama-4-Maverick-17B-128E-Instruct-FP8`, `Phi-4`) & GPT-4.1 family (`GPT-4.1` , `GPT-4.1-nano` , `GPT-4.1-mini`) | Baseline multi-agent deliberation — named and anonymous variants across GlobalOpinionsQA, HLE, and persona datasets |
| `python experiments/sys_prompt_main.py` | `sys_prompt_configs_20/` | GPT-4.1 family | System prompt ablations — `critical_independent` and `adversarial` conditions, each with named and anonymous variants; agents revise answers each round |
| `python experiments/sys_prompt_no_revise.py` | `sys_prompt_configs_no_revise_20/` | GPT-4.1 family | Same as above but agents do not revise their answers between rounds |
| `python experiments/rotate_main.py` | `rotate_configs/` | GPT-4.1 family | Rotation experiments — cyclically swaps round-1 answers between agents to isolate position/order effects |
| `python experiments/null_exp.py` | `null_exp_configs/` | GPT-4.1 (`gpt-4.1` × 3) | Null/control experiment — all three agents share the same model; runs only the anonymous variant to measure same-model deliberation as a baseline |

Each experiment runs **3 agents** for up to **20 deliberation rounds**. Results are saved under `results/` organized by dataset and config.

**Supported datasets:** GlobalOpinionsQA · anthropic-persona · Humanity's Last Exam (HLE)

---

## Analysis

### Disagreement & Deference

| Script | Description |
|--------|-------------|
| `python Analysis_files/all_models_disagree.py` | Walks all `results/` subdirectories and computes per-round disagreement rate, pairwise disagreement, and directional deference (small→large vs large→small). Saves JSON outputs to `new_analysis_outputs/`. |
| `python Analysis_files/all_plots.py` | Reads from `new_analysis_outputs/` and generates disagreement and deference plots to `new_figures/`. Produces individual per-variant plots and combined named-vs-anonymous overlays. |

### HLE Accuracy

| Script | Description |
|--------|-------------|
| `python Analysis_files/HLE_accuracy.py` | Walks `results/HLE/` and computes three accuracy metrics per directory. Saves JSON outputs to `new_analysis_outputs/HLE/`. |
| `python Analysis_files/HLE_accuracy_plot.py` | Reads from `new_analysis_outputs/HLE/` and saves three plots per directory to `new_figures/HLE/`. |

`HLE_accuracy.py` computes the following metrics:

| Output file | Metric |
|-------------|--------|
| `hle_accuracy.json` | Per-round accuracy — a question is correct only if **all** agents answered correctly |
| `hle_pairwise_accuracy.json` | Per model pair, per round — questions where **both** models in the pair answered correctly |
| `hle_deference_accuracy.json` | Among round-1 disagreements, tracks deference direction (`small→large`, `large→small`) and accuracy outcomes (`wrong→correct`, `correct→wrong`) |

`HLE_accuracy_plot.py` generates the following plots:

| Output file | Plot |
|-------------|------|
| `rounds_accuracy.png` | Accuracy across deliberation rounds |
| `deference_direction.png` | Grouped bars: Small→Large vs Large→Small deference rates per model pair |
| `deference_accuracy_outcomes.png` | Four bars per pair: `wrong→correct` and `correct→wrong` rates for both deference directions |

---

## Repository Structure

```
agent_consensus/
├── Analysis_files/                        # Analysis scripts
│   ├── all_models_disagree.py             # Disagreement and deference metrics across all results
│   ├── all_plots.py                       # Disagreement and deference plots
│   ├── HLE_accuracy.py                    # Accuracy metrics for HLE results
│   └── HLE_accuracy_plot.py               # Accuracy and deference plots for HLE
│
├── agents/                                # Multi-agent conversation orchestration
│   ├── multiagent.py                      # Core agent talk loop (baseline and sys_prompt experiments)
│   ├── multiagent2.py                     # Variant used by no-revise experiments
│   └── multiagent_rotate.py               # Variant for rotation experiments
│
├── baseline_configs_20/                   # Configs for experiments/main.py (random family agents)
├── sys_prompt_configs_20/                 # Configs for experiments/sys_prompt_main.py (gpt-4.1 family)
├── sys_prompt_configs_no_revise_20/       # Configs for experiments/sys_prompt_no_revise.py (gpt-4.1 family)
├── rotate_configs/                        # Configs for experiments/rotate_main.py (gpt-4.1 family)
├── null_exp_configs/                      # Configs for experiments/null_exp.py (gpt-4.1 × 3)
│
├── data/                                  # Raw and preprocessed datasets
│   ├── jsonl/                             # Preprocessed JSONL files used by experiments
│   ├── evals/                             # Cloned Anthropic evals repo (persona dataset source)
│   ├── Anthropic___llm_global_opinions/   # Cached HuggingFace GlobalOpinionsQA dataset
│   └── cais___hle/                        # Cached HuggingFace HLE dataset
│
├── experiments/                           # Experiment runner scripts
│   ├── main.py                            # Baseline multi-agent experiments (random family)
│   ├── sys_prompt_main.py                 # System prompt experiments (with answer revision)
│   ├── sys_prompt_no_revise.py            # System prompt experiments (no answer revision)
│   ├── rotate_main.py                     # Rotation experiments
│   └── null_exp.py                        # Null/control experiment (same model for all agents)
│
├── models/                                # Model API wrappers
│   ├── model.py                           # Azure API wrapper (baseline experiments)
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
│   ├── download_dataset.py                # Downloads GlobalOpinionsQA from HuggingFace
│   ├── filter_questions.py                # Filters out invalid or malformed questions
│   ├── preprocess_hle.py                  # Converts HLE dataset to JSONL
│   └── preprocess_persona.py              # Converts Anthropic persona dataset to JSONL
│
├── results/                               # Experiment outputs (one JSON file per question)
│   ├── GlobalOpinionsQA/
│   ├── HLE/
│   └── anthropic-persona/
│
├── tests/                                 # Unit tests
│   ├── test_accuracy.py
│   ├── test_disagree.py
│   └── test_models.py
│
├── miscellaneous/                         # Utility scripts and archived outputs
├── .env                                   # Azure API credentials (not committed)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```
