"""
Raspberry Pi AI Camera (IMX500) manager.

The IMX500 has a Sony neural-network accelerator baked into the sensor itself,
so it can run small models (object detection, classification) at the edge
without burdening the CPU or the Hailo HAT+.

Install picamera2 (system package on Raspberry Pi OS):
    sudo apt install -y python3-picamera2 python3-libcamera

IMX500 model packages (.rpk) are available at:
    https://github.com/raspberrypi/imx500-models
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from PIL import Image

from config.settings import (
    CAMERA_CAPTURE_SIZE,
    CAMERA_PREVIEW_SIZE,
    IMX500_MODEL_PATH,
)


class AICameraManager:
    """
    Manages the Raspberry Pi AI Camera (IMX500).

    Two operating modes:
      1. Plain capture  — just grab RGB frames (no on-sensor inference).
      2. IMX500 mode    — load a .rpk model; on-sensor inference results are
                          embedded in the frame metadata.

    Usage (plain):
        with AICameraManager() as cam:
            frame = cam.capture_frame()

    Usage (IMX500 on-sensor detection):
        with AICameraManager(imx500_model=IMX500_MODEL_PATH) as cam:
            frame, detections = cam.capture_with_inference()
    """

    def __init__(
        self,
        imx500_model: str | None = None,
        preview_size: tuple[int, int] = CAMERA_PREVIEW_SIZE,
        capture_size: tuple[int, int] = CAMERA_CAPTURE_SIZE,
    ) -> None:
        self._imx500_model = imx500_model
        self._preview_size = preview_size
        self._capture_size = capture_size
        self._picam2: Any = None
        self._imx500: Any = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        from picamera2 import Picamera2

        if self._imx500_model:
            from picamera2.devices.imx500 import IMX500

            self._imx500 = IMX500(self._imx500_model)
            self._picam2 = Picamera2(self._imx500.camera_num)
        else:
            self._picam2 = Picamera2()

        config = self._picam2.create_preview_configuration(
            main={"size": self._preview_size, "format": "RGB888"},
        )
        self._picam2.configure(config)
        self._picam2.start()
        # Give the sensor a moment to settle
        time.sleep(0.5)

    def stop(self) -> None:
        if self._picam2:
            self._picam2.stop()
            self._picam2.close()
            self._picam2 = None

    # ── capture ───────────────────────────────────────────────────────────────

    def capture_frame(self) -> Image.Image:
        """Return the current frame as a PIL RGB Image."""
        arr = self._picam2.capture_array("main")
        return Image.fromarray(arr)

    def capture_numpy(self) -> np.ndarray:
        """Return the current frame as a uint8 (H, W, 3) NumPy array."""
        return self._picam2.capture_array("main")

    def capture_still(self, size: tuple[int, int] | None = None) -> Image.Image:
        """
        Switch to a full-resolution still configuration, grab one frame,
        then switch back to preview.  Slower but higher quality.
        """
        still_size = size or self._capture_size
        config = self._picam2.create_still_configuration(
            main={"size": still_size, "format": "RGB888"}
        )
        self._picam2.switch_mode_and_capture_array(config, "main")
        arr = self._picam2.capture_array("main")
        # Restore preview config
        preview_config = self._picam2.create_preview_configuration(
            main={"size": self._preview_size, "format": "RGB888"}
        )
        self._picam2.switch_mode(preview_config)
        return Image.fromarray(arr)

    # ── IMX500 on-sensor inference ────────────────────────────────────────────

    def get_imx500_outputs(self) -> dict[str, np.ndarray] | None:
        """
        Retrieve on-sensor inference outputs embedded in frame metadata.
        Returns None if IMX500 mode is not active.
        """
        if self._imx500 is None or self._picam2 is None:
            return None
        metadata = self._picam2.capture_metadata()
        return self._imx500.get_outputs(metadata, add_batch=True)

    def capture_with_inference(
        self,
    ) -> tuple[Image.Image, dict[str, np.ndarray] | None]:
        """Convenience: capture frame + on-sensor inference results together."""
        frame = self.capture_frame()
        outputs = self.get_imx500_outputs()
        return frame, outputs

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "AICameraManager":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
