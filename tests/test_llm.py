"""Basic smoke tests for the Claude client (mocked — no real API calls)."""
from unittest.mock import MagicMock, patch

import pytest

from src.llm.client import ClaudeClient, _pil_to_base64


def test_pil_to_base64_returns_string():
    from PIL import Image

    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    data, media_type = _pil_to_base64(img)
    assert isinstance(data, str)
    assert len(data) > 0
    assert media_type == "image/jpeg"


def test_pil_to_base64_handles_rgba():
    from PIL import Image

    img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
    data, media_type = _pil_to_base64(img)
    assert isinstance(data, str)


def test_build_content_text_only():
    client = ClaudeClient.__new__(ClaudeClient)
    client.system_prompt = "test"
    result = client._build_content("hello", None)
    assert result == "hello"


def test_build_content_with_image():
    from PIL import Image

    client = ClaudeClient.__new__(ClaudeClient)
    client.system_prompt = "test"
    img = Image.new("RGB", (10, 10))
    result = client._build_content("describe this", img)
    assert isinstance(result, list)
    assert result[0]["type"] == "image"
    assert result[1]["type"] == "text"
    assert result[1]["text"] == "describe this"


def test_clear_history():
    client = ClaudeClient.__new__(ClaudeClient)
    client.history = [{"role": "user", "content": "hi"}]
    client.clear_history()
    assert client.history == []
