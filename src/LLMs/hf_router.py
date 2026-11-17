"""HuggingFace Inference Providers through HF's OpenAI-compatible router.

Used by provider files for models served via HuggingFace with the inference
provider pinned per model ("<repo id>:<provider>", e.g.
"Qwen/Qwen3.6-27B:deepinfra"). Pinning keeps runs reproducible and avoids
`InferenceClient`'s "auto" routing, which fails when HF's first provider for a
model is one the installed huggingface_hub does not know (e.g. deepinfra,
ovhcloud). Authenticates with the cached `huggingface-cli login` token.
"""

from huggingface_hub import get_token
from openai import OpenAI

HF_ROUTER_URL = "https://router.huggingface.co/v1"


def hf_router_client() -> OpenAI:
    """Return an OpenAI client pointed at the HF router."""
    token = get_token()
    assert token, "No HuggingFace token found; run `huggingface-cli login`"
    return OpenAI(base_url=HF_ROUTER_URL, api_key=token)


def hf_router_summarize(client: OpenAI, target: str, prepared_text: str, temperature: float, max_tokens: int) -> str:
    """Single-turn chat completion; returns the answer text (reasoning, if any, is a separate field)."""
    response = client.chat.completions.create(
        model=target,
        messages=[{"role": "user", "content": prepared_text}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # Some providers prefix the answer with blank lines after hidden reasoning.
    return (response.choices[0].message.content or "").strip()
