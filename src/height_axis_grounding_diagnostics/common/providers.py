from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from height_axis_grounding_diagnostics.common.io_utils import (
    ROOT,
    detect_mime_type,
    dump_sdk_response,
    encode_image_to_base64,
)


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/api"


def normalize_ollama_base_url(base_url: str | None) -> str:
    normalized = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    if not normalized.endswith("/api"):
        normalized = f"{normalized}/api"
    return normalized


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    model_env_var: str
    default_model: str | None
    api_key_env_var: str | None
    api_key_required: bool
    batch_size_env_var: str
    default_batch_size: int
    base_url_env_var: str | None = None
    default_base_url: str | None = None
    doc_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderRuntime:
    models: dict[str, str]
    batch_sizes: dict[str, int]
    api_keys: dict[str, str]
    base_urls: dict[str, str | None]


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        model_env_var="OPENAI_MODEL",
        default_model="gpt-5.4-2026-03-05",
        api_key_env_var="OPENAI_API_KEY",
        api_key_required=True,
        batch_size_env_var="OPENAI_BATCH_SIZE",
        default_batch_size=2,
        doc_sources=(
            "https://platform.openai.com/docs/api-reference/responses/compact?api-mode=responses",
            "https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=base64-encoded",
        ),
    ),
    "gemini": ProviderSpec(
        name="gemini",
        model_env_var="GEMINI_MODEL",
        default_model="gemini-3-flash-preview",
        api_key_env_var="GEMINI_API_KEY",
        api_key_required=True,
        batch_size_env_var="GEMINI_BATCH_SIZE",
        default_batch_size=2,
        doc_sources=(
            "https://ai.google.dev/gemini-api/docs/image-understanding",
            "https://ai.google.dev/gemini-api/docs/models/experimental-models?hl=zh-tw",
        ),
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        model_env_var="ANTHROPIC_MODEL",
        default_model="claude-sonnet-4-6",
        api_key_env_var="ANTHROPIC_API_KEY",
        api_key_required=True,
        batch_size_env_var="ANTHROPIC_BATCH_SIZE",
        default_batch_size=2,
        doc_sources=(
            "https://docs.anthropic.com/en/api/messages",
            "https://docs.anthropic.com/en/docs/build-with-claude/vision",
            "https://docs.anthropic.com/en/docs/about-claude/models/all-models",
        ),
    ),
    "ollama": ProviderSpec(
        name="ollama",
        model_env_var="OLLAMA_MODEL",
        default_model=None,
        api_key_env_var=None,
        api_key_required=False,
        batch_size_env_var="OLLAMA_BATCH_SIZE",
        default_batch_size=1,
        base_url_env_var="OLLAMA_BASE_URL",
        default_base_url=DEFAULT_OLLAMA_BASE_URL,
        doc_sources=(
            "https://docs.ollama.com/api/introduction",
            "https://docs.ollama.com/capabilities/vision",
            "https://docs.ollama.com/api/chat",
            "https://docs.ollama.com/api/errors",
        ),
    ),
}

SUPPORTED_PROVIDERS = tuple(PROVIDER_SPECS)
DEFAULT_MODELS = {
    name: spec.default_model
    for name, spec in PROVIDER_SPECS.items()
    if spec.default_model is not None
}
API_DOC_SOURCES = {
    name: list(spec.doc_sources)
    for name, spec in PROVIDER_SPECS.items()
}


class ProviderError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProviderClient:
    provider_name: str

    def __init__(
        self,
        model: str,
        api_key: str,
        timeout_seconds: int,
        temperature: float,
        *,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.base_url = base_url

    def run(self, *, prompt_text: str, image_path: Path) -> dict[str, Any]:
        raise NotImplementedError


def add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=SUPPORTED_PROVIDERS,
        default=list(SUPPORTED_PROVIDERS),
        help="Providers to run. Default: all supported providers.",
    )
    for provider, spec in PROVIDER_SPECS.items():
        parser.add_argument(
            f"--{provider}-model",
            help=f"Override the {provider} model. Env: {spec.model_env_var}.",
        )
        parser.add_argument(
            f"--{provider}-batch-size",
            type=int,
            help=f"Concurrent attempts for {provider}. Env: {spec.batch_size_env_var}.",
        )
        if spec.base_url_env_var:
            parser.add_argument(
                f"--{provider}-base-url",
                help=f"Override the {provider} API base URL. Env: {spec.base_url_env_var}.",
            )


def get_provider_doc_sources(providers: list[str] | tuple[str, ...]) -> dict[str, list[str]]:
    return {provider: list(PROVIDER_SPECS[provider].doc_sources) for provider in providers}


def get_installed_sdk_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "openai": None,
        "anthropic": None,
        "google-genai": None,
        "requests": None,
        "ollama": "REST API",
    }
    try:
        import openai  # type: ignore

        versions["openai"] = getattr(openai, "__version__", None)
    except ImportError:
        pass
    try:
        import anthropic  # type: ignore

        versions["anthropic"] = getattr(anthropic, "__version__", None)
    except ImportError:
        pass
    try:
        from google import genai  # type: ignore

        versions["google-genai"] = getattr(genai, "__version__", None)
    except ImportError:
        pass
    try:
        import requests  # type: ignore

        versions["requests"] = getattr(requests, "__version__", None)
    except ImportError:
        pass
    return versions


def resolve_provider_runtime(
    args: argparse.Namespace,
    *,
    dry_run: bool,
    env_file: Path | None = None,
) -> ProviderRuntime:
    models: dict[str, str] = {}
    batch_sizes: dict[str, int] = {}
    api_keys: dict[str, str] = {}
    base_urls: dict[str, str | None] = {}

    for provider in args.providers:
        spec = PROVIDER_SPECS[provider]

        model_override = getattr(args, f"{provider}_model", None)
        model = model_override or os.environ.get(spec.model_env_var) or spec.default_model
        if not model:
            if dry_run:
                model = f"{provider}-dry-run"
            else:
                raise ProviderError(
                    f"Missing model for provider `{provider}`. "
                    f"Set {spec.model_env_var} or pass --{provider}-model."
                )
        models[provider] = model

        raw_batch_size = getattr(args, f"{provider}_batch_size", None)
        if raw_batch_size is None:
            raw_batch_size = os.environ.get(spec.batch_size_env_var, spec.default_batch_size)
        try:
            batch_size = int(raw_batch_size)
        except (TypeError, ValueError) as exc:
            raise ProviderError(
                f"`{spec.batch_size_env_var}` or --{provider}-batch-size must be an integer."
            ) from exc
        if batch_size < 1:
            raise ProviderError(f"`{provider}` batch size must be at least 1")
        batch_sizes[provider] = batch_size

        if dry_run:
            api_keys[provider] = ""
        elif spec.api_key_required:
            api_key = os.environ.get(spec.api_key_env_var or "")
            if not api_key:
                location = f" in {env_file}" if env_file else ""
                raise ProviderError(
                    f"Missing API key for provider `{provider}`{location}. "
                    f"Set {spec.api_key_env_var}."
                )
            api_keys[provider] = api_key
        else:
            api_keys[provider] = os.environ.get(spec.api_key_env_var or "", "")

        base_url = None
        if spec.base_url_env_var:
            base_url_override = getattr(args, f"{provider}_base_url", None)
            base_url = (
                base_url_override
                or os.environ.get(spec.base_url_env_var)
                or spec.default_base_url
            )
        base_urls[provider] = base_url.rstrip("/") if isinstance(base_url, str) else None

    return ProviderRuntime(
        models=models,
        batch_sizes=batch_sizes,
        api_keys=api_keys,
        base_urls=base_urls,
    )


class OpenAIClient(ProviderClient):
    provider_name = "openai"

    def __init__(self, model: str, api_key: str, timeout_seconds: int, temperature: float) -> None:
        super().__init__(model, api_key, timeout_seconds, temperature)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ProviderError(
                "OpenAI SDK is not installed in the active Python environment. "
                "Install it in `.venv` and run with `./.venv/bin/python`."
            ) from exc
        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds)

    def run(self, *, prompt_text: str, image_path: Path) -> dict[str, Any]:
        mime_type, image_b64 = encode_image_to_base64(image_path)
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{image_b64}",
                            "detail": "high",
                        },
                    ],
                }
            ],
            "temperature": self.temperature,
        }

        response = self._client.responses.create(**payload)
        raw_text, response_json = dump_sdk_response(response)
        output_text = getattr(response, "output_text", None)
        return {
            "status_code": 200,
            "raw_response_text": raw_text,
            "response_json": response_json,
            "response_text": output_text,
            "request_summary": {
                "transport": "openai-python SDK",
                "endpoint": "client.responses.create",
                "model": self.model,
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "mime_type": mime_type,
                "temperature": self.temperature,
                "structured_output": False,
            },
        }


class GeminiClient(ProviderClient):
    provider_name = "gemini"

    def __init__(self, model: str, api_key: str, timeout_seconds: int, temperature: float) -> None:
        super().__init__(model, api_key, timeout_seconds, temperature)
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:
            raise ProviderError(
                "google-genai SDK is not installed in the active Python environment. "
                "Install it in `.venv` and run with `./.venv/bin/python`."
            ) from exc
        self._types = types
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=self.timeout_seconds * 1000),
        )

    def run(self, *, prompt_text: str, image_path: Path) -> dict[str, Any]:
        mime_type = detect_mime_type(image_path)
        image_part = self._types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type)
        response = self._client.models.generate_content(
            model=self.model,
            contents=[prompt_text, image_part],
            config=self._types.GenerateContentConfig(temperature=self.temperature),
        )
        raw_text, response_json = dump_sdk_response(response)
        output_text = getattr(response, "text", None)
        return {
            "status_code": 200,
            "raw_response_text": raw_text,
            "response_json": response_json,
            "response_text": output_text,
            "request_summary": {
                "transport": "google-genai SDK",
                "endpoint": "client.models.generate_content",
                "model": self.model,
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "mime_type": mime_type,
                "temperature": self.temperature,
                "structured_output": False,
            },
        }


class AnthropicClient(ProviderClient):
    provider_name = "anthropic"

    def __init__(self, model: str, api_key: str, timeout_seconds: int, temperature: float) -> None:
        super().__init__(model, api_key, timeout_seconds, temperature)
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as exc:
            raise ProviderError(
                "Anthropic SDK is not installed in the active Python environment. "
                "Install it in `.venv` and run with `./.venv/bin/python`."
            ) from exc
        self._client = Anthropic(api_key=api_key, timeout=timeout_seconds)

    def run(self, *, prompt_text: str, image_path: Path) -> dict[str, Any]:
        mime_type, image_b64 = encode_image_to_base64(image_path)
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                    ],
                }
            ],
        }
        response = self._client.messages.create(**payload)
        raw_text, response_json = dump_sdk_response(response)
        output_text = None
        if response_json:
            text_chunks = [
                item.get("text")
                for item in response_json.get("content", [])
                if item.get("type") == "text" and isinstance(item.get("text"), str)
            ]
            if text_chunks:
                output_text = "".join(text_chunks)

        return {
            "status_code": 200,
            "raw_response_text": raw_text,
            "response_json": response_json,
            "response_text": output_text,
            "request_summary": {
                "transport": "anthropic SDK",
                "endpoint": "client.messages.create",
                "model": self.model,
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "mime_type": mime_type,
                "temperature": self.temperature,
                "structured_output": False,
            },
        }


class OllamaClient(ProviderClient):
    provider_name = "ollama"

    def __init__(
        self,
        model: str,
        api_key: str,
        timeout_seconds: int,
        temperature: float,
        *,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            model,
            api_key,
            timeout_seconds,
            temperature,
            base_url=normalize_ollama_base_url(base_url),
        )
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise ProviderError(
                "requests is not installed in the active Python environment. "
                "Install requirements and run with `./.venv/bin/python`."
            ) from exc
        self._requests = requests
        self.base_url = normalize_ollama_base_url(self.base_url)

    def run(self, *, prompt_text: str, image_path: Path) -> dict[str, Any]:
        mime_type, image_b64 = encode_image_to_base64(image_path)
        endpoint = f"{self.base_url}/chat"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        try:
            response = self._requests.post(endpoint, json=payload, timeout=self.timeout_seconds)
        except self._requests.exceptions.RequestException as exc:
            raise ProviderError(f"Ollama API request failed: {exc}") from exc

        raw_response_text = response.text
        try:
            response_json = response.json()
        except ValueError:
            response_json = None

        if not response.ok:
            error_message = ""
            if isinstance(response_json, dict):
                error_message = str(response_json.get("error") or "")
            if not error_message:
                error_message = raw_response_text.strip()
            raise ProviderError(
                f"Ollama API request failed with HTTP {response.status_code}: {error_message}",
                status_code=response.status_code,
            )

        if not isinstance(response_json, dict):
            raise ProviderError("Ollama API returned a non-JSON response", status_code=response.status_code)

        message = response_json.get("message")
        output_text = message.get("content") if isinstance(message, dict) else None
        if output_text is not None and not isinstance(output_text, str):
            output_text = str(output_text)

        return {
            "status_code": response.status_code,
            "raw_response_text": raw_response_text
            or json.dumps(response_json, ensure_ascii=False, indent=2),
            "response_json": response_json,
            "response_text": output_text,
            "request_summary": {
                "transport": "ollama REST API",
                "endpoint": endpoint,
                "model": self.model,
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "mime_type": mime_type,
                "temperature": self.temperature,
                "base_url": self.base_url,
                "think": False,
                "structured_output": False,
            },
        }


def build_provider_client(
    provider: str,
    *,
    model: str,
    api_key: str,
    timeout_seconds: int,
    temperature: float,
    base_url: str | None = None,
) -> ProviderClient:
    if provider == "openai":
        return OpenAIClient(model, api_key, timeout_seconds, temperature)
    if provider == "gemini":
        return GeminiClient(model, api_key, timeout_seconds, temperature)
    if provider == "anthropic":
        return AnthropicClient(model, api_key, timeout_seconds, temperature)
    if provider == "ollama":
        return OllamaClient(
            model,
            api_key,
            timeout_seconds,
            temperature,
            base_url=base_url,
        )
    raise ProviderError(f"Unsupported provider: {provider}")
