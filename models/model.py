import os
import time
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI, RateLimitError, APITimeoutError
load_dotenv()

MAX_RETRIES = 6
BASE_BACKOFF = 5  # seconds

AZURE_OPENAI_MODELS = {
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
}

FOUNDRY_MODELS = {
    "Mistral-Large-3",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Phi-4",
}

DEFAULT_MODEL_NAME = "gpt-4.1-mini"

# Per-model max_tokens cap. Llama is extremely verbose without one.
_MODEL_MAX_TOKENS: dict[str, int] = {
    "Llama-4-Maverick-17B-128E-Instruct-FP8": 400,
}

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
        timeout=300.0,
    )

def run_model(prompt: str, model_name: str = DEFAULT_MODEL_NAME) -> str:
    for attempt in range(MAX_RETRIES):
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
                extra = {}
                if model_name in _MODEL_MAX_TOKENS:
                    extra["max_tokens"] = _MODEL_MAX_TOKENS[model_name]
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    **extra,
                )

            else:
                raise ValueError(f"Unknown model name: {model_name}")

            return response.choices[0].message.content

        except (RateLimitError, APITimeoutError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = BASE_BACKOFF * (2 ** attempt)
            label = "RATE LIMIT" if isinstance(e, RateLimitError) else "TIMEOUT"
            print(f"[{label}] {model_name} error, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)

        except Exception as e:
            if "content_filter" in str(e).lower():
                with open("miscellaneous/azure_filtered.log", "a") as f:
                    f.write(prompt[:500] + "\n\n")
                return ""
            raise
