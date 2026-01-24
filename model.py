from openai import AzureOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
DEFAULT_MODEL_NAME = "gpt-4.1-nano"

def create_model():
    return AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-12-01-preview"
)


def run_model(prompt: str, model_name: str = DEFAULT_MODEL_NAME) -> str:
    client = create_model()
    response = client.chat.completions.create(
        model=model_name,
          messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content