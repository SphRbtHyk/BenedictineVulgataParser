"""Pluggable model backends for the VLM stage (stage 1).

The paper's claim is that the pipeline is **model-agnostic**: any
vision-language model able to transcribe a scanned page image works,
whether it is served by the Mistral API or by a local OpenAI-compatible
server (mlx-vlm, llama.cpp, vLLM, LM Studio, Ollama's OpenAI endpoint...).

``ModelBackend`` is the shared abstraction:

- :class:`MistralApiBackend` — the reference backend (Mistral API,
  e.g. Pixtral or any vision model served there);
- :class:`LocalBackend` — any locally hosted OpenAI-compatible model
  server or locally served checkpoint.

Both speak the same wire format (OpenAI ``/chat/completions`` with an image
URL part), so swapping backends never changes the rest of the pipeline:
the stage-1 artifact (``model_pages`` + ``page_sizes``) is identical.
"""

from __future__ import annotations

import base64
import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests
from PIL import Image

#: Default Mistral API endpoint (can be overridden for proxies/prod gateways).
MISTRAL_DEFAULT_ENDPOINT = "https://api.mistral.ai/v1"
#: Default vision model for the Mistral backend.  Any *image-capable* model
#: the account serves works (``--model``); the Pixtral models used during the
#: paper's experiments are retired, so the default is a current multimodal
#: Mistral model.
MISTRAL_DEFAULT_MODEL = "mistral-medium-latest"
#: Environment variable read when ``api_key_env`` is not given explicitly.
DEFAULT_API_KEY_ENV = "MISTRAL_API_KEY"


class BackendError(RuntimeError):
    """A backend call failed (transport, authentication or inference)."""


def _pil_to_data_url(image: Image.Image, max_width: int = 1600,
                     jpeg_quality: int = 85) -> str:
    """Encode a PIL image as a ``data:image/jpeg;base64,...`` URL.

    The image is downscaled to ``max_width`` and re-encoded as JPEG so the
    request payload stays reasonable for any backend, local or remote.
    """
    im = image.convert("RGB")
    if im.width > max_width:
        scale = max_width / im.width
        im = im.resize((max_width, round(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=jpeg_quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class ModelBackend(ABC):
    """Abstract base class of every VLM inference backend.

    A backend must expose exactly one capability: given a page image and a
    prompt, return the model's raw text response.  Subclasses decide only
    *where* the model is served; the prompt, the artifact format and every
    downstream stage are backend-independent.
    """

    name: str = "model"
    endpoint: str = ""
    model: str = ""
    timeout: float = 300.0
    temperature: float = 0.0
    max_tokens: int = 8192
    extra_headers: Dict[str, str] = field(default_factory=dict)

    @abstractmethod
    def chat_completion(self, messages: list) -> str:
        """Send one chat-completion request and return the response text."""

    def transcribe_page(self, image: Image.Image, prompt: str) -> str:
        """Transcribe one page image with ``prompt``; returns raw model text."""
        if not isinstance(prompt, str) or not prompt.strip():
            # never send a null/empty text part: APIs reject it with HTTP 422
            raise ValueError(
                "transcribe_page needs a non-empty prompt string "
                f"(got {prompt!r}); pass APPARATUS_PROMPT or your own prompt")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": _pil_to_data_url(image)}},
        ]}]
        return self.chat_completion(messages).strip()


@dataclass
class _OpenAICompatClient:
    """Shared OpenAI-compatible ``/chat/completions`` transport."""

    endpoint: str
    model: str
    api_key: Optional[str]
    timeout: float
    temperature: float
    max_tokens: int
    extra_headers: Dict[str, str]

    def chat_completion(self, messages: list) -> str:
        url = self.endpoint.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json", **self.extra_headers}
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = requests.post(url, json=payload, headers=headers,
                                 timeout=self.timeout)
        except requests.RequestException as exc:
            raise BackendError(f"{self.model}: request to {url} failed: {exc}") from exc
        if resp.status_code in (401, 403):
            raise BackendError(
                f"{self.model}: authentication failed (HTTP {resp.status_code}) "
                f"at {url} -- check the API key")
        if resp.status_code == 404:
            raise BackendError(
                f"{self.model}: endpoint not found at {url} -- check the URL "
                f"and that the model name is served there")
        if resp.status_code == 422:
            raise BackendError(
                f"{self.model}: HTTP 422 (unprocessable request) from {url}: "
                f"{resp.text[:300]}. The model likely rejects the message "
                f"shape -- a text-only model (e.g. mistral-large) cannot take "
                f"the image part; use an image-capable model (default "
                f"{MISTRAL_DEFAULT_MODEL}) via --model, or a retired model "
                f"name may no longer be served")
        if not resp.ok:
            raise BackendError(
                f"{self.model}: HTTP {resp.status_code} from {url}: "
                f"{resp.text[:500]}")
        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
            raise BackendError(
                f"{self.model}: malformed response from {url}: "
                f"{resp.text[:500]}") from exc


@dataclass
class MistralApiBackend(ModelBackend):
    """Reference backend: the Mistral API (any image-capable Mistral model).

    The endpoint URL, the model name and how to reach the API key are all
    configuration, never constants of the pipeline.  The API key is taken
    from ``api_key`` directly, else read from the environment variable named
    by ``api_key_env`` (default ``MISTRAL_API_KEY``; the CLI flag
    ``--api-key-env`` overrides that name).
    """

    def __init__(self, endpoint: str = MISTRAL_DEFAULT_ENDPOINT,
                 model: str = MISTRAL_DEFAULT_MODEL,
                 api_key: Optional[str] = None,
                 api_key_env: str = DEFAULT_API_KEY_ENV,
                 timeout: float = 300.0, temperature: float = 0.0,
                 max_tokens: int = 8192,
                 extra_headers: Optional[Dict[str, str]] = None) -> None:
        key = api_key or os.environ.get(api_key_env)
        super().__init__(name="mistral-api", endpoint=endpoint, model=model,
                         timeout=timeout, temperature=temperature,
                         max_tokens=max_tokens,
                         extra_headers=extra_headers or {})
        self._client = _OpenAICompatClient(
            endpoint=endpoint, model=model, api_key=key, timeout=timeout,
            temperature=temperature, max_tokens=max_tokens,
            extra_headers=self.extra_headers)

    def chat_completion(self, messages: list) -> str:
        return self._client.chat_completion(messages)


@dataclass
class LocalBackend(ModelBackend):
    """Local backend: any OpenAI-compatible server hosting a local model.

    Works out of the box with mlx-vlm's ``mlx_vlm.server`` (the GLM-OCR
    path used on Apple Silicon in the paper), llama.cpp's ``server``,
    vLLM, LM Studio or any endpoint that implements
    ``POST <endpoint>/chat/completions``.  The connection details
    (endpoint and model name) are given at call time, never baked in;
    a local checkpoint is exposed by serving it with one of those servers.
    """

    def __init__(self, endpoint: str, model: str = "local-model",
                 api_key: Optional[str] = None, timeout: float = 600.0,
                 temperature: float = 0.0, max_tokens: int = 8192,
                 extra_headers: Optional[Dict[str, str]] = None) -> None:
        super().__init__(name="local", endpoint=endpoint, model=model,
                         timeout=timeout, temperature=temperature,
                         max_tokens=max_tokens,
                         extra_headers=extra_headers or {})
        self._client = _OpenAICompatClient(
            endpoint=endpoint, model=model, api_key=api_key, timeout=timeout,
            temperature=temperature, max_tokens=max_tokens,
            extra_headers=self.extra_headers)

    def chat_completion(self, messages: list) -> str:
        return self._client.chat_completion(messages)


def build_backend(backend: str, *, endpoint: Optional[str] = None,
                  model: Optional[str] = None,
                  api_key: Optional[str] = None,
                  api_key_env: Optional[str] = None,
                  timeout: float = 300.0,
                  extra_headers: Optional[Dict[str, str]] = None) -> ModelBackend:
    """Factory: build a backend by name from user-supplied configuration.

    Args:
        backend: ``"mistral-api"`` or ``"local"``.
        endpoint: base URL of the API / local server (``.../v1``).
        model: model name as known to the server.
        api_key: explicit API key (overrides the environment variable).
        api_key_env: environment variable holding the API key
            (default ``MISTRAL_API_KEY``; used by the Mistral backend).
        timeout, extra_headers: passed to the backend.

    Returns:
        The configured :class:`ModelBackend` instance.
    """
    if backend == "mistral-api":
        return MistralApiBackend(
            endpoint=endpoint or MISTRAL_DEFAULT_ENDPOINT,
            model=model or MISTRAL_DEFAULT_MODEL,
            api_key=api_key,
            api_key_env=api_key_env or DEFAULT_API_KEY_ENV,
            timeout=timeout, extra_headers=extra_headers)
    if backend == "local":
        if not endpoint:
            raise ValueError(
                "the local backend needs --endpoint, e.g. "
                "http://127.0.0.1:8080/v1")
        return LocalBackend(endpoint=endpoint, model=model or "local-model",
                            api_key=api_key, timeout=timeout,
                            extra_headers=extra_headers)
    raise ValueError(
        f"unknown backend {backend!r} (expected 'mistral-api' or 'local')")
