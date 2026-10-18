"""API providers: one class per hosted/local model API.

Add a provider by adding a class here and registering it in registry.py."""
from __future__ import annotations

import os

from .base import APIProvider


# ----- API providers (presets) -------------------------------------------

class OpenAIProvider(APIProvider):
    slug = "openai"
    label = "OpenAI-compatible (OpenAI / vLLM / LM Studio / llama.cpp)"
    help = ("Any server speaking the OpenAI chat-completions protocol. A key you "
            "save (or the env var you name) is sent as a Bearer token; with no key, "
            "no Authorization header is sent.")
    request_template = {
        "model": "gpt-4o-mini",
        "api_key_env_var": "OPENAI_API_KEY",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "body": '{\n  "model": "gpt-4o-mini",\n  "messages": [\n    {"role": "user", "content": "{prompt}"}\n  ]\n}',
        "response_path": "choices.0.message.content",
        "headers": "",
    }

    def _build_sdk(self, t, api_key):
        from pyrit.prompt_target import OpenAIChatTarget
        endpoint = (t.endpoint_url or "https://api.openai.com/v1").rstrip("/")
        for suffix in ("/chat/completions", "/v1/chat/completions"):
            if endpoint.endswith(suffix):
                endpoint = endpoint[: -len(suffix)].rstrip("/")
        return OpenAIChatTarget(
            model_name=t.model_name,
            api_key=api_key or "not-needed",
            endpoint=endpoint,
        )


class OllamaProvider(APIProvider):
    slug = "ollama"
    label = "Ollama (local)"
    help = "A local Ollama server, via its OpenAI-compatible /v1 route. No key needed."
    request_template = {
        "model": "llama3",
        "api_key_env_var": "",
        "endpoint": "http://localhost:11434/v1/chat/completions",
        "body": '{\n  "model": "llama3",\n  "messages": [\n    {"role": "user", "content": "{prompt}"}\n  ]\n}',
        "response_path": "choices.0.message.content",
        "headers": "",
    }

    def _build_sdk(self, t, api_key):
        from pyrit.prompt_target import OpenAIChatTarget
        return OpenAIChatTarget(
            model_name=t.model_name,
            endpoint=(t.endpoint_url or "http://localhost:11434").rstrip("/"),
            api_key="ollama",
        )


class HttpGenericProvider(APIProvider):
    slug = "http_generic"
    label = "Custom HTTP / JSON API"
    help = ("Any HTTP+JSON endpoint. Paste the exact request body with a {prompt} "
            "placeholder, and set the response path to the reply field.")
    request_template = {
        "model": "",
        "api_key_env_var": "",
        "endpoint": "",
        "body": '{\n  "input": "{prompt}"\n}',
        "response_path": "output.text",
        "headers": "",
    }


    # inherits APIProvider._build_sdk, which tells the user to supply a body.


class AnthropicProvider(APIProvider):
    slug = "anthropic"
    label = "Anthropic (Claude)"
    help = ("Anthropic Messages API. Auth goes in the x-api-key header via the "
            "{api_key} placeholder (resolved from your key env var).")
    request_template = {
        "model": "claude-3-5-haiku-latest",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "endpoint": "https://api.anthropic.com/v1/messages",
        "body": ('{\n  "model": "claude-3-5-haiku-latest",\n  "max_tokens": 1024,\n'
                 '  "messages": [\n    {"role": "user", "content": "{prompt}"}\n  ]\n}'),
        "response_path": "content.0.text",
        "headers": "x-api-key: {api_key}\nanthropic-version: 2023-06-01",
    }

    def _build_sdk(self, t, api_key):
        from pyrit.prompt_target import AnthropicChatTarget  # type: ignore
        return AnthropicChatTarget(
            model_name=t.model_name,
            api_key=api_key or "not-needed",
        )


class AzureOpenAIProvider(APIProvider):
    slug = "azure_openai"
    label = "Azure OpenAI"
    help = ("Azure-hosted OpenAI deployment. The endpoint includes your resource, "
            "deployment name and api-version; auth uses the api-key header.")
    request_template = {
        "model": "<deployment>",
        "api_key_env_var": "AZURE_OPENAI_API_KEY",
        "endpoint": "https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2024-02-15-preview",
        "body": '{\n  "messages": [\n    {"role": "user", "content": "{prompt}"}\n  ]\n}',
        "response_path": "choices.0.message.content",
        "headers": "api-key: {api_key}",
    }

    def _build_sdk(self, t, api_key):
        from pyrit.prompt_target import OpenAIChatTarget
        base = (t.endpoint_url or os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
        return OpenAIChatTarget(
            model_name=t.model_name,
            api_key=api_key or "not-needed",
            endpoint=base or None,
        )


class HuggingFaceProvider(APIProvider):
    slug = "huggingface"
    label = "HuggingFace (in-process, GPU)"
    help = "Loads a HuggingFace model in-process. Needs a GPU and the model id; no HTTP body."
    raw = False
    request_template = {"model": "meta-llama/Llama-3.1-8B-Instruct"}

    def build_target(self, t, api_key, multi_turn: bool = False):
        from pyrit.prompt_target import HuggingFaceChatTarget  # type: ignore
        return HuggingFaceChatTarget(model_id=t.model_name)
