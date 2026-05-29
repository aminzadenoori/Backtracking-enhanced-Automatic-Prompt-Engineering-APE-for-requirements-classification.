"""LLM backends. Each backend exposes a `chat(system, user) -> str` method and a
`.model_fn` attribute that the core algorithm calls.

Two backends are provided:
  - OllamaBackend         : local open-weight models (used in the paper)
  - OpenAICompatBackend   : any OpenAI-compatible /v1/chat/completions endpoint
"""
from __future__ import annotations

import requests


# The five open-weight instruction models used in the paper (served via Ollama).
PAPER_MODELS = {
    "qwen2:7b-instruct":      {"name": "Qwen2-7B",      "params": "7B", "attn": "RoPE + MQA",            "org": "Alibaba Cloud"},
    "falcon3:7b-instruct":    {"name": "Falcon3-7B",    "params": "7B", "attn": "MQA",                   "org": "TII (UAE)"},
    "granite3.2:8b-instruct": {"name": "Granite-3.2-8B","params": "8B", "attn": "Scaled dot-product",    "org": "IBM Research"},
    "ministral:8b-instruct":  {"name": "Ministral-8B",  "params": "8B", "attn": "Sliding Window Attn",   "org": "Mistral AI"},
    "llama3:8b-instruct":     {"name": "LLaMA-3-8B",    "params": "8B", "attn": "GQA + RoPE",            "org": "Meta AI"},
}


class OllamaBackend:
    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, system: str, user: str) -> str:
        r = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.base_url}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    @property
    def model_fn(self):
        return self.chat


class OpenAICompatBackend:
    def __init__(self, model: str, base_url: str, api_key: str = "", timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, system: str, user: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    @property
    def model_fn(self):
        return self.chat


def make_backend(backend: str, model: str, base_url: str, api_key: str = ""):
    if backend == "ollama":
        return OllamaBackend(model=model, base_url=base_url)
    if backend in ("openai-compat", "openai"):
        return OpenAICompatBackend(model=model, base_url=base_url, api_key=api_key)
    raise ValueError(f"Unknown backend: {backend}")
