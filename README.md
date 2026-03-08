# Agent_Consensus
1. Have your python version as 3.10+

2. Do pip install -r requirements.txt

3. Make ur .env file containing your Azure Endpoint and Azure API Key

4. To download datasets:
    a. If you're on Windows do python preprocessing/download_dataset.py  for OpinionsQA and GlobalOpinionsQA datasets
    b. If you're on a Mac do python3 preprocessing/download_dataset.py  for OpinionsQA and GlobalOpinionsQA datasets
    c. To download the anthropic dataset run: git clone https://github.com/anthropics/evals in the data folder

5. To preproceess datasets run:
    a. python combine_opinionsqa_datasets.py
    b. python combine_persona_datasets.py

6. To run experiments:
    a. python sys_prompt_main.py (Runs experiments with system prompts)
    b. python change_answer_all.py (Runs rotated experiments)
    c. python main.py (Runs baseline multiagent experiments)
    
7. To analyze results:
    a. python plots/all_models_disagree.py (Saves information on model disagreement rate and model deference rate to the analysis_outputs folder)
    b. python plots/all_plots.py (Saves plots from the information in the analysis_outputs folder to the figures folder)
