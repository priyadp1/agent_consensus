import os
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
load_dotenv()

AZURE_OPENAI_MODELS = {
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
}

FOUNDRY_MODELS = {
    "DeepSeek-R1",
    "grok-3",
    "Llama-3.3-70B-Instruct",
}

DEFAULT_MODEL_NAME = "gpt-4.1-nano"

def create_azure_openai_client():
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-12-01-preview",
    )


def create_foundry_client():
    return OpenAI(
        base_url=os.environ["AZURE_FOUNDRY_ENDPOINT"],
        api_key=os.environ["AZURE_FOUNDRY_API_KEY"],
    )

def run_model(prompt: str, model_name: str = DEFAULT_MODEL_NAME) -> str:
    try:
        if model_name in AZURE_OPENAI_MODELS:
            client = create_azure_openai_client()
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

        elif model_name in FOUNDRY_MODELS:
            client = create_foundry_client()
            response = client.chat.completions.create(
                model=model_name, 
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return response.choices[0].message.content

    except Exception as e:
        if "content_filter" in str(e).lower():
            with open("azure_filtered.log", "a") as f:
                f.write(prompt[:500] + "\n\n")
            return ""
        raise