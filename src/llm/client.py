"""Anthropic Claude client — streaming, vision, multi-turn conversation."""
from __future__ import annotations

import base64
import io
from typing import Iterator

import anthropic
from PIL import Image

from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, SYSTEM_PROMPT


class ClaudeClient:
    """
    Wrapper around the Anthropic streaming API.

    Supports:
    - Multi-turn conversation history (stateful)
    - Vision: pass a PIL Image alongside text
    - Adaptive thinking (claude-opus-4-6 default)
    - Streaming text output with optional per-token callback
    """

    def __init__(
        self,
        system_prompt: str = SYSTEM_PROMPT,
        model: str = CLAUDE_MODEL,
        max_tokens: int = 1024,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.history: list[dict] = []

    # ── public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        text: str,
        image: Image.Image | None = None,
        on_token: "((str) -> None) | None" = None,
    ) -> str:
        """
        Send a message (with optional image) and return the full response text.
        Conversation history is maintained across calls.

        Args:
            text: The user's message.
            image: Optional PIL Image sent as vision context.
            on_token: Optional callback called with each text delta as it streams.
        """
        self.history.append({"role": "user", "content": self._build_content(text, image)})
        response_text = self._stream(on_token)
        # Store only the text in history — omit thinking blocks from context
        self.history.append({"role": "assistant", "content": response_text})
        return response_text

    def ask(
        self,
        text: str,
        image: Image.Image | None = None,
        on_token: "((str) -> None) | None" = None,
    ) -> str:
        """Single-turn query — does NOT update conversation history."""
        messages = [{"role": "user", "content": self._build_content(text, image)}]
        return self._stream(on_token, messages=messages)

    def stream_tokens(
        self,
        text: str,
        image: Image.Image | None = None,
    ) -> Iterator[str]:
        """
        Generator that yields text tokens one at a time.
        Updates conversation history when exhausted.

        Example:
            for token in client.stream_tokens("Hello"):
                display.update(token)
        """
        self.history.append({"role": "user", "content": self._build_content(text, image)})
        collected: list[str] = []
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=self.system_prompt,
            messages=self.history,
        ) as stream:
            for token in stream.text_stream:
                collected.append(token)
                yield token
        self.history.append({"role": "assistant", "content": "".join(collected)})

    def clear_history(self) -> None:
        self.history.clear()

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_content(
        self, text: str, image: Image.Image | None
    ) -> list[dict] | str:
        if image is None:
            return text
        data, media_type = _pil_to_base64(image)
        return [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            },
            {"type": "text", "text": text},
        ]

    def _stream(
        self,
        on_token: "((str) -> None) | None" = None,
        messages: list[dict] | None = None,
    ) -> str:
        msgs = messages if messages is not None else self.history
        chunks: list[str] = []
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=self.system_prompt,
            messages=msgs,
        ) as stream:
            for token in stream.text_stream:
                chunks.append(token)
                if on_token:
                    on_token(token)
        return "".join(chunks)


# ── helpers ───────────────────────────────────────────────────────────────────

def _pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> tuple[str, str]:
    buf = io.BytesIO()
    # Convert RGBA → RGB so JPEG encoder doesn't complain
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buf, format=fmt)
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return data, f"image/{fmt.lower()}"
