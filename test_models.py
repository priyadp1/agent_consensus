from model import run_model
import os
from dotenv import load_dotenv

load_dotenv()

print("FOUNDRY ENDPOINT:", os.getenv("AZURE_FOUNDRY_ENDPOINT"))
print("FOUNDRY KEY:", bool(os.getenv("AZURE_FOUNDRY_API_KEY")))
print("AZURE ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("AZURE KEY:", bool(os.getenv("AZURE_OPENAI_API_KEY")))

TEST_PROMPT = (
    "Answer the following exactly.\n\n"
    "What is 2 + 2?\n\n"
    "FINAL LINE MUST BE:\n"
    "ANSWER: <number>"
)

FOUNDRY_TEST_MODELS = [
    "DeepSeek-R1",
    "grok-3",
    "Llama-3.3-70B-Instruct",
]

AZURE_OPENAI_TEST_MODELS =[
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
]

for model in FOUNDRY_TEST_MODELS:
    print("=" * 60)
    print(f"Testing model: {model}")

    try:
        output = run_model(TEST_PROMPT, model_name=model)
        print("Raw output:")
        print(output)

    except Exception as e:
        print(f"FAILED for {model}")
        print(e)

for model in AZURE_OPENAI_TEST_MODELS:
    print("=" * 60)
    print(f"Testing model: {model}")

    try:
        output = run_model(TEST_PROMPT, model_name=model)
        print("Raw output:")
        print(output)

    except Exception as e:
        print(f"FAILED for {model}")
        print(e)