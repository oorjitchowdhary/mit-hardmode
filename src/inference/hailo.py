"""
Hailo-8L AI Accelerator inference engine (Raspberry Pi AI HAT+, 13 TOPS).

The AI HAT+ connects over PCIe (M.2 2242) and is exposed through the
HailoRT runtime as a standard VDevice.

Install HailoRT
───────────────
1.  Download the HailoRT package for your OS/arch from:
      https://hailo.ai/developer-zone/software-downloads/
    (Raspberry Pi 5 → ARM64, HailoRT ≥ 4.18)

2.  Install the .deb:
      sudo dpkg -i hailort_<version>_arm64.deb

3.  Install Python bindings (comes with the deb, or manually):
      pip install hailort

4.  Verify:
      hailortcli fw-control identify

Pre-compiled .hef models
─────────────────────────
Download from the Hailo Model Zoo:
  https://github.com/hailo-ai/hailo_model_zoo
Common models:
  yolov8s.hef            — object detection (80 COCO classes)
  resnet_v1_50.hef       — image classification (ImageNet)
  face_detection.hef     — lightweight face detection
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config.settings import HAILO_HEF_PATH

# Graceful import — Hailo SDK is only available on the Pi
try:
    from hailo_platform import (  # type: ignore[import]
        ConfigureParams,
        HEF,
        HailoSchedulingAlgorithm,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )

    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


class HailoInferenceEngine:
    """
    Load a .hef model and run batched inference on the Hailo-8L.

    Usage:
        with HailoInferenceEngine("models/yolov8s.hef") as engine:
            image = cam.capture_numpy()          # (H, W, 3) uint8
            preprocessed = engine.preprocess(image)
            outputs = engine.infer(preprocessed)
            detections = engine.parse_detections(outputs)

    The engine lazily raises ImportError with a helpful message if HailoRT
    is not installed, so the rest of the codebase can import this module
    safely on non-Pi environments.
    """

    def __init__(self, hef_path: str = HAILO_HEF_PATH) -> None:
        self.hef_path = hef_path
        self._device: Any = None
        self._network_group: Any = None
        self._input_vstream_params: Any = None
        self._output_vstream_params: Any = None
        self._input_shape: tuple[int, ...] | None = None
        self._input_name: str | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        _require_hailo()
        if not Path(self.hef_path).exists():
            raise FileNotFoundError(
                f"HEF model not found: {self.hef_path}\n"
                "Download models from https://github.com/hailo-ai/hailo_model_zoo"
            )

        hef = HEF(self.hef_path)
        self._device = VDevice()

        configure_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        network_groups = self._device.configure(hef, configure_params)
        self._network_group = network_groups[0]

        self._input_vstream_params = InputVStreamParams.make(
            self._network_group, quantized=False
        )
        self._output_vstream_params = OutputVStreamParams.make(
            self._network_group, quantized=False
        )

        # Cache input shape so callers can preprocess correctly
        input_infos = hef.get_input_vstream_infos()
        if input_infos:
            info = input_infos[0]
            self._input_name = info.name
            self._input_shape = tuple(info.shape)  # e.g. (1, 640, 640, 3) for YOLOv8

    def stop(self) -> None:
        if self._device is not None:
            self._device.release()
            self._device = None
            self._network_group = None

    # ── inference ─────────────────────────────────────────────────────────────

    def infer(self, input_array: np.ndarray) -> dict[str, np.ndarray]:
        """
        Run inference.

        Args:
            input_array: Pre-processed input matching the model's expected
                         shape, dtype, and value range.  Call preprocess()
                         first if you have a raw image.

        Returns:
            Dict mapping output tensor name → ndarray.
        """
        _require_hailo()
        assert self._network_group is not None, "Call start() first"

        input_data = {self._input_name: input_array}
        with self._network_group.activate():
            with InferVStreams(
                self._network_group,
                self._input_vstream_params,
                self._output_vstream_params,
            ) as pipeline:
                return pipeline.infer(input_data)

    # ── image helpers ─────────────────────────────────────────────────────────

    def preprocess(self, image: np.ndarray | Image.Image) -> np.ndarray:
        """
        Resize and normalise an image to match the model's input tensor.

        Assumes a standard float32 [0, 1] normalisation.
        For models expecting [0, 255] uint8, adjust accordingly.
        """
        if self._input_shape is None:
            raise RuntimeError("Engine not started — call start() first.")

        # Shape is (batch, H, W, C) for most Hailo models
        _, h, w, _ = self._input_shape

        if isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype(np.uint8))
        else:
            img = image

        img = img.resize((w, h), Image.LANCZOS).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)  # add batch dimension

    @property
    def input_shape(self) -> tuple[int, ...] | None:
        return self._input_shape

    # ── result parsers ────────────────────────────────────────────────────────

    @staticmethod
    def parse_detections(
        outputs: dict[str, np.ndarray],
        conf_threshold: float = 0.5,
        nms_iou: float = 0.45,
    ) -> list[dict]:
        """
        Generic post-processing for YOLO-style detection outputs.

        Returns a list of dicts:
            {"bbox": [x1, y1, x2, y2], "score": float, "class_id": int}

        NOTE: Exact tensor layout differs per model.  This is a starting
        point — adjust for your specific .hef output format.
        """
        detections: list[dict] = []
        for name, tensor in outputs.items():
            flat = tensor.reshape(-1, tensor.shape[-1]) if tensor.ndim > 2 else tensor
            for row in flat:
                if len(row) < 6:
                    continue
                x1, y1, x2, y2, score, class_id = (
                    row[0], row[1], row[2], row[3], row[4], int(row[5])
                )
                if score >= conf_threshold:
                    detections.append(
                        {"bbox": [x1, y1, x2, y2], "score": float(score), "class_id": class_id}
                    )
        return detections

    @staticmethod
    def parse_classification(
        outputs: dict[str, np.ndarray], top_k: int = 5
    ) -> list[dict]:
        """
        Post-process softmax classification output.

        Returns top-k dicts:  {"class_id": int, "score": float}
        """
        for tensor in outputs.values():
            scores = tensor.flatten()
            indices = np.argsort(scores)[::-1][:top_k]
            return [{"class_id": int(i), "score": float(scores[i])} for i in indices]
        return []

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "HailoInferenceEngine":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_hailo() -> None:
    if not HAILO_AVAILABLE:
        raise ImportError(
            "HailoRT is not installed.\n"
            "Download the SDK from https://hailo.ai/developer-zone/ and follow "
            "scripts/setup.sh for installation instructions."
        )
