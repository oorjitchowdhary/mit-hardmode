"""
OLED display driver — Inland 1.3″ SH1106, 128×64, I2C.

Hardware:
    Module    Raspberry Pi 5
    ──────    ──────────────────────────────
    VCC   →   3.3V  (physical pin 1 or 17)
    GND   →   GND   (physical pin 6, 9, 14, …)
    SCL   →   GPIO 3 / SCL  (physical pin 5)
    SDA   →   GPIO 2 / SDA  (physical pin 3)

Enable I2C on the Pi:
    sudo raspi-config  → Interface Options → I2C → Enable
    # or add to /boot/firmware/config.txt:  dtparam=i2c_arm=on

Install luma.oled:
    pip install luma.oled

Verify address (should print 0x3c):
    i2cdetect -y 1
"""
from __future__ import annotations

import textwrap

from PIL import Image, ImageDraw, ImageFont

from config.settings import OLED_HEIGHT, OLED_I2C_ADDRESS, OLED_I2C_PORT, OLED_WIDTH

# Font metrics for the default bitmap font
_CHAR_W = 6   # pixels per character (default font)
_CHAR_H = 8
_LINE_H = 10  # line height with small gap
_MAX_COLS = OLED_WIDTH // _CHAR_W    # ≈ 21 chars/line
_MAX_LINES = OLED_HEIGHT // _LINE_H  # ≈ 6 lines


class OLEDDisplay:
    """
    High-level wrapper for the SH1106 128×64 OLED.

    All draw methods accept standard PIL/Pillow constructs so you can
    compose complex layouts before calling show_image().

    Usage:
        with OLEDDisplay() as disp:
            disp.show_text("Hello Pi!")
    """

    def __init__(self) -> None:
        from luma.core.interface.serial import i2c
        from luma.oled.device import sh1106

        serial = i2c(port=OLED_I2C_PORT, address=OLED_I2C_ADDRESS)
        self._device = sh1106(serial, width=OLED_WIDTH, height=OLED_HEIGHT)
        self._font = ImageFont.load_default()

    # ── drawing API ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Blank the display."""
        img = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
        self._device.display(img)

    def show_text(
        self,
        text: str,
        x: int = 0,
        y: int = 0,
        line_height: int = _LINE_H,
    ) -> None:
        """
        Render word-wrapped text starting at (x, y).
        Long strings are automatically wrapped to fit the display width.
        """
        lines = _wrap(text, max_cols=(_MAX_COLS - x // _CHAR_W))
        img = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(lines[: _MAX_LINES]):
            draw.text((x, y + i * line_height), line, font=self._font, fill=1)
        self._device.display(img)

    def show_status(self, title: str, body: str = "") -> None:
        """
        Two-zone layout:
          • Title — inverted (white background, black text) on row 0
          • Body  — normal text below
        """
        img = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
        draw = ImageDraw.Draw(img)

        # Title bar
        title_text = title[:_MAX_COLS]
        draw.rectangle((0, 0, OLED_WIDTH - 1, _LINE_H + 1), fill=1)
        draw.text((1, 1), title_text, font=self._font, fill=0)

        # Body
        if body:
            body_lines = _wrap(body)
            for i, line in enumerate(body_lines[: _MAX_LINES - 1]):
                draw.text((0, _LINE_H + 2 + i * _LINE_H), line, font=self._font, fill=1)

        self._device.display(img)

    def show_image(self, image: Image.Image) -> None:
        """
        Display a PIL Image.
        The image is resized to 128×64 and converted to 1-bit monochrome.
        """
        mono = image.resize((OLED_WIDTH, OLED_HEIGHT), Image.LANCZOS).convert("1")
        self._device.display(mono)

    def show_canvas(self) -> "CanvasContext":
        """
        Return a context manager that gives you a raw ImageDraw object
        and flushes to the display on exit.

        Example:
            with disp.show_canvas() as draw:
                draw.ellipse((10, 10, 50, 50), outline=1)
        """
        return CanvasContext(self._device)

    def show_progress(self, label: str, fraction: float) -> None:
        """
        Show a labelled progress bar (fraction 0.0 – 1.0).
        Useful for long inference or recording operations.
        """
        img = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), label[:_MAX_COLS], font=self._font, fill=1)
        bar_y = 20
        bar_w = int(OLED_WIDTH * max(0.0, min(1.0, fraction)))
        draw.rectangle((0, bar_y, OLED_WIDTH - 1, bar_y + 10), outline=1)
        if bar_w > 0:
            draw.rectangle((0, bar_y, bar_w - 1, bar_y + 10), fill=1)
        pct = f"{int(fraction * 100)}%"
        draw.text((0, bar_y + 14), pct, font=self._font, fill=1)
        self._device.display(img)

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "OLEDDisplay":
        return self

    def __exit__(self, *_: object) -> None:
        self.clear()


class CanvasContext:
    """Returned by OLEDDisplay.show_canvas() — flushes on __exit__."""

    def __init__(self, device: object) -> None:
        self._device = device
        self._img = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
        self._draw = ImageDraw.Draw(self._img)

    def __enter__(self) -> ImageDraw.ImageDraw:
        return self._draw

    def __exit__(self, *_: object) -> None:
        self._device.display(self._img)


# ── helpers ───────────────────────────────────────────────────────────────────

def _wrap(text: str, max_cols: int = _MAX_COLS) -> list[str]:
    """Word-wrap *text* to *max_cols* characters per line."""
    return textwrap.wrap(text, width=max_cols) or [""]
