"""Unit tests for display helpers that don't require real hardware."""
from src.display.oled import _wrap


def test_wrap_short_string():
    lines = _wrap("Hello world")
    assert lines == ["Hello world"]


def test_wrap_long_string():
    long_text = "This is a very long string that should definitely wrap across multiple lines on the display"
    lines = _wrap(long_text, max_cols=21)
    assert len(lines) > 1
    for line in lines:
        assert len(line) <= 21


def test_wrap_empty_string():
    lines = _wrap("")
    assert lines == [""]


def test_wrap_exact_width():
    text = "A" * 21
    lines = _wrap(text, max_cols=21)
    assert lines[0] == text
