import os
import threading
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

# Max concurrent in-flight requests per model.
# Foundry models (especially Kimi) have strict rate limits.
_MODEL_CONCURRENCY = {
    "Mistral-Large-3": 1,
    "Llama-4-Maverick-17B-128E-Instruct-FP8": 2,
    "Phi-4": 2,
}
_DEFAULT_CONCURRENCY = 4

# Minimum seconds between consecutive requests to a model.
# Enforced while holding the semaphore so the gap is respected even under load.
_MODEL_MIN_INTERVAL = {
    "Mistral-Large-3": 2.0,
}

_semaphore_lock = threading.Lock()
_semaphores: dict[str, threading.Semaphore] = {}
_last_call_lock = threading.Lock()
_last_call: dict[str, float] = {}


def _get_semaphore(model_name: str) -> threading.Semaphore:
    with _semaphore_lock:
        if model_name not in _semaphores:
            limit = _MODEL_CONCURRENCY.get(model_name, _DEFAULT_CONCURRENCY)
            _semaphores[model_name] = threading.Semaphore(limit)
    return _semaphores[model_name]


def _throttle(model_name: str) -> None:
    """Sleep if needed to respect _MODEL_MIN_INTERVAL for this model."""
    min_interval = _MODEL_MIN_INTERVAL.get(model_name, 0.0)
    if min_interval <= 0:
        return
    with _last_call_lock:
        now = time.monotonic()
        gap = min_interval - (now - _last_call.get(model_name, 0.0))
        if gap > 0:
            time.sleep(gap)
        _last_call[model_name] = time.monotonic()

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
        timeout=300.0,  # Kimi can be slow; 5 min read timeout
    )

def run_model(prompt: str, model_name: str = DEFAULT_MODEL_NAME) -> str:
    with _get_semaphore(model_name):
        return _run_model_inner(prompt, model_name)


def _run_model_inner(prompt: str, model_name: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            _throttle(model_name)
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