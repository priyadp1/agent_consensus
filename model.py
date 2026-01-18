from openai import AzureOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
MODEL_NAME = "gpt-4.1-nano"
def create_model():
    client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-12-01-preview",
)
    return client


def run_model(prompt: str):
    model = create_model()
    response = model.chat.completions.create(
        model=MODEL_NAME,
          messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content