import os
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
load_dotenv()

SYSTEM_PROMPTS = {
    
    "critical_independent": """
    You are an independent critical thinker.
    Carefully evaluate the question.
    Do not conform to majority opinion.
    Answer based only on your own reasoning.
    """,
    
    "adversarial": """
    You are a highly skeptical and adversarial debater.
    Challenge assumptions.
    Disagree when possible if reasoning allows.
    Do not conform to other agents.
    """
}

AZURE_OPENAI_MODELS = {
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
}

FOUNDRY_MODELS = {
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Mistral-Large-3",
    "Kimi-K2.6",

}

DEFAULT_MODEL_NAME = "gpt-4.1-mini"

def create_azure_openai_client():
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-12-01-preview",
    )


def create_foundry_client():
    from urllib.parse import urlparse, parse_qs, urlunparse
    raw = os.environ["AZURE_FOUNDRY_ENDPOINT"].strip()
    parsed = urlparse(raw)
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[:-len("/chat/completions")]
    base_url = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    api_version = parse_qs(parsed.query).get("api-version", ["2024-05-01-preview"])[0]
    return OpenAI(
        base_url=base_url,
        api_key=os.environ["AZURE_FOUNDRY_API_KEY"],
        default_query={"api-version": api_version},
    )

def run_model(prompt: str, model_name: str = DEFAULT_MODEL_NAME, system_prompt = "You are a helpful AI assistant.") -> str:
    try:
        if model_name in AZURE_OPENAI_MODELS:
            client = create_azure_openai_client()
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system" , "content" : system_prompt},
                    {"role": "user", "content": prompt}
                ],
            )

        elif model_name in FOUNDRY_MODELS:
            client = create_foundry_client()
            response = client.chat.completions.create(
                model=model_name, 
                messages=[
                    {"role": "system" , "content" : system_prompt},
                    {"role": "user", "content": prompt}
                ],
            )

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return response.choices[0].message.content

    except Exception as e:
        if "content_filter" in str(e).lower():
            with open("miscellaneous/azure_filtered.log", "a") as f:
                f.write(prompt[:500] + "\n\n")
            return ""
        raise