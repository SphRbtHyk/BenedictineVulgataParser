"""Unit tests of the model backends and the stage-1 transcription wiring.

The VLM stage is served end to end against an in-process fake
OpenAI-compatible server, so the request/response contract is exercised
without any network access.  The Mistral backend shares the same transport.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from vulgata_scan.backends import (BackendError, LocalBackend,
                                   MistralApiBackend, build_backend)
from vulgata_scan.vlm.transcribe import (parse_column_json,
                                         regions_from_response)


class _FakeVLMHandler(BaseHTTPRequestHandler):
    """A minimal fake of an OpenAI-compatible /chat/completions endpoint."""

    reply = '{"desunt": "Desunt $\\Sigma^{T}$.", "left": "1. a $B$.", "right": ""}'

    def do_POST(self):  # noqa: N802 (stdlib naming)
        self._last_body = self.rfile.read(int(self.headers["Content-Length"]))
        MockServer.bodies.append(json.loads(self._last_body))
        payload = {"choices": [{"message": {"content": type(self).reply}}]}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the default stderr logging
        pass


class MockServer:
    """Runs the fake VLM on an ephemeral localhost port for one test."""

    bodies: list = []

    def __init__(self) -> None:
        self.httpd = HTTPServer(("127.0.0.1", 0), _FakeVLMHandler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)

    def start(self) -> "MockServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


def _band_image(w=400, h=300, ink_rows=None) -> Image.Image:
    """A synthetic apparatus band: white page, a first line, a left column."""
    im = Image.new("L", (w, h), 255)
    for y in range(10, 24):          # the Desunt line
        for x in range(10, 150):
            im.putpixel((x, y), 0)
    for y in range(40, 200, 20):     # 'text' lines of the left column
        for x in range(10, 120):
            im.putpixel((x, y), 40)
    for y in range(40, 200, 20):     # right column starts far right
        for x in range(260, 380):
            im.putpixel((x, y), 40)
    return im


def test_build_backend_local_requires_endpoint():
    with pytest.raises(ValueError, match="endpoint"):
        build_backend("local")
    with pytest.raises(ValueError, match="unknown backend"):
        build_backend("gpt-9000")
    back = build_backend("local", endpoint="http://127.0.0.1:9000/v1",
                        model="mlx-community/GLM-OCR-bf16")
    assert isinstance(back, LocalBackend)
    assert back.model == "mlx-community/GLM-OCR-bf16"


def test_mistral_backend_uses_env_key(monkeypatch):
    from vulgata_scan.backends import MISTRAL_DEFAULT_MODEL
    monkeypatch.setenv("MY_TEST_KEY", "sk-xyz")
    back = build_backend("mistral-api", api_key_env="MY_TEST_KEY",
                         model="mistral-medium-latest")
    assert isinstance(back, MistralApiBackend)
    assert back._client.api_key == "sk-xyz"
    assert back._client.endpoint == "https://api.mistral.ai/v1"
    assert back._client.model == "mistral-medium-latest"
    # no --model: the default is a current image-capable model (pixtral is
    # retired and text-only models get HTTP 422)
    default = build_backend("mistral-api", api_key_env="MY_TEST_KEY")
    assert default._client.model == MISTRAL_DEFAULT_MODEL
    assert MISTRAL_DEFAULT_MODEL != "pixtral-12b-2409"


def test_transcribe_page_rejects_empty_prompt():
    # a null/empty text part must never go on the wire (APIs answer HTTP 422)
    backend = LocalBackend(endpoint="http://127.0.0.1:9/v1", model="fake")
    with pytest.raises(ValueError, match="non-empty prompt"):
        backend.transcribe_page(_band_image(), "")
    with pytest.raises(ValueError, match="non-empty prompt"):
        backend.transcribe_page(_band_image(), None)  # type: ignore[arg-type]


def test_transcribe_page_none_prompt_uses_canonical_prompt(tmp_path):
    from vulgata_scan.vlm.transcribe import APPARATUS_PROMPT, transcribe_page
    captured = {}

    class _CaptureBackend(LocalBackend):
        def transcribe_page(self, image, prompt):
            captured["prompt"] = prompt
            return '{"desunt": "", "left": "1. a $B$.", "right": ""}'

    backend = _CaptureBackend(endpoint="http://127.0.0.1:9/v1", model="fake")
    png = tmp_path / "page-1.png"
    _band_image().save(png)
    regions, size = transcribe_page(png, backend, prompt=None)
    assert captured["prompt"] == APPARATUS_PROMPT
    assert regions and size == list(_band_image().size)


def test_local_backend_transcribe_roundtrip():
    server = MockServer().start()
    MockServer.bodies = []
    try:
        backend = LocalBackend(endpoint=server.endpoint, model="fake-vlm")
        reply = backend.transcribe_page(_band_image(),
                                        'Output {"desunt": "d", "left": "l"}')
        assert '"desunt"' in reply
        request = MockServer.bodies[0]
        assert request["model"] == "fake-vlm"
        assert request["messages"][0]["role"] == "user"
        parts = request["messages"][0]["content"]
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        # the image part decodes to a real JPEG
        raw = base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1])
        assert raw[:2] == b"\xff\xd8"
    finally:
        server.stop()


def test_backend_error_on_bad_status():
    class _Handler(_FakeVLMHandler):
        def do_POST(self):  # noqa: N802
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "model not found"}')
    _FakeVLMHandler.reply = _FakeVLMHandler.reply  # keep class attr untouched
    server = MockServer()
    server.httpd.RequestHandlerClass = _Handler
    server.start()
    try:
        backend = LocalBackend(endpoint=server.endpoint, model="fake")
        with pytest.raises(BackendError, match="endpoint not found"):
            backend.chat_completion([{"role": "user", "content": "hi"}])
    finally:
        server.stop()


def test_backend_error_422_hints_image_capable_model():
    # what the Mistral API answers for a text-only model (e.g. mistral-large)
    # or a retired one (pixtral-*) receiving an image part
    class _Handler(_FakeVLMHandler):
        def do_POST(self):  # noqa: N802
            body = b'{"detail":[{"msg":"Input should be a valid string"}]}'
            self.send_response(422)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = MockServer()
    server.httpd.RequestHandlerClass = _Handler
    server.start()
    try:
        backend = MistralApiBackend(endpoint=server.endpoint,
                                    model="mistral-large", api_key="k")
        with pytest.raises(BackendError, match="image-capable"):
            backend.chat_completion([{"role": "user", "content": "hi"}])
    finally:
        server.stop()


def test_parse_column_json_accepts_fenced_and_loose_replies():
    loose = '{"desunt": "Desunt $\\Sigma^{T}$.", "left": "1. a $B$.", "right": ""}'
    fenced = "```json\n" + loose + "\n```"
    for text in (loose, fenced):
        cols = parse_column_json(text)
        assert cols is not None
        assert cols["desunt"].startswith("Desunt")
        assert cols["left"] == "1. a $B$."
    assert parse_column_json("no columns here, just text") is None


def test_regions_from_response_layout():
    band = _band_image()
    reply = ('{"desunt": "Desunt $\\Sigma^{T}$.", '
             '"left": "1. apparatus text $B$.", "right": "2. right text $C$."}')
    # two columns: force both non-empty, page with a gutter
    region_list = regions_from_response(band, y0=100, y1=400,
                                        page_size=(400, 500),
                                        response=reply)
    labels = [r["label"] for r in region_list]
    assert labels[0] == "witness_line"      # the Desunt anchor, full width
    assert "apparatus" in labels            # column regions below it
    for r in region_list:
        x0, y0, x1, y1 = r["bbox_norm_1000"]
        assert 0 <= x0 < x1 <= 1000
        assert y0 < y1
    # all region bboxes live inside the band sent to the VLM
    anchor = region_list[0]
    assert anchor["content"].startswith("Desunt")
    assert any("right text" in r["content"] for r in region_list)
