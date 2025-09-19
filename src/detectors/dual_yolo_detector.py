"""
Dual YOLO detector wrapper used across the app.

This module provides a detector that runs two YOLO models on each frame and
merges their outputs:
1) A pretrained YOLOv11 model for general objects (e.g., person, book, phone, laptop/notebook)
2) A custom YOLOv11-nano weights file for exam-specific objects (e.g., earphones/headsets,
   smartwatches, calculators)

If a model path/name is invalid, initialization will raise.
is invalid, initialization will raise.

Normalized detection format:
[{"class_name": str, "class_id": int, "conf": float, "bbox": [x1, y1, x2, y2]}]
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Any

import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # type: ignore


def _normalize_class_name(raw: str) -> str:
    """Map various YOLO label variants to our internal canonical names.

    Examples
    - 'cell phone' -> 'phone'
    - 'laptop' -> 'notebook'
    - 'headphones'/'earbuds'/'earpods'/'headset' -> 'earphone'
    - 'smart watch'/'smartwatch' -> 'smartwatch'
    """
    name = str(raw).strip().lower()
    # general model mappings
    if name in ("cell phone", "mobile phone", "phone","telephone","mobile","cellphone","smart phone"):
        return "phone"
    if name == "laptop":
        return "notebook"
    # if name == "tv":
    #     return "monitor"
    # custom model mappings for wearables/headsets
    if name in ("headphones","headphone", "headset", "earbuds", "earpods", "earphone", "earphones"):
        return "earphone"
    if name in ("smart watch", "smartwatch", "smart-watch"):
        return "smartwatch"
    # keep known classes as-is
    if name in ("person", "book", "calculator", "notebook"):
        return name
    # default to raw
    return name


from typing import Any


def _predict_one(model: Any, frame: np.ndarray, device: Optional[str], conf: float, source_tag: str, class_conf: Optional[Dict[str, float]] = None) -> List[Dict]:
    results = model.predict(frame, conf=conf, verbose=False, device=device)
    out: List[Dict] = []
    if not results:
        return out
    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return out
    names = getattr(model, "names", None) or []
    for b in boxes:
        xyxy = b.xyxy[0].tolist()
        conf_v = float(b.conf.item()) if hasattr(b, "conf") else 0.0
        cls_id = int(b.cls.item()) if hasattr(b, "cls") else -1
        if isinstance(names, dict) and cls_id in names:
            raw_name = names[cls_id]
        elif isinstance(names, list) and 0 <= cls_id < len(names):
            raw_name = names[cls_id]
        else:
            raw_name = str(cls_id)
        name = _normalize_class_name(raw_name)
        # Apply per-class confidence thresholds if provided
        if class_conf and name in class_conf:
            if conf_v < float(class_conf[name]):
                continue
        out.append(
            {
                "class_name": name,
                "class_id": cls_id,
                "conf": conf_v,
                "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                "source": source_tag,
            }
        )
    return out


class DualYoloDetector:
    """Runs two YOLO models and concatenates detections.

    Parameters
    ----------
    primary_model: str
        Pretrained YOLOv11 model name or path (e.g., 'yolov11m.pt').
    secondary_model: str
        Custom YOLO weights path for exam-specific objects (e.g., 'models/model_bestV3.pt').
    device: Optional[str]
        Torch/Ultralytics device hint ('cuda'|'cpu'|None).
    """

    def __init__(
        self,
        primary_model: str = "yolov11m.pt",
        secondary_model: str = os.path.join("models", "model_bestV3.pt"),
        device: Optional[str] = None,
    ) -> None:
        if YOLO is None:
            raise ImportError("ultralytics not installed. Please install requirements and try again.")
        # Load both models strictly 
        self.primary = YOLO(primary_model)
        if not (secondary_model and os.path.exists(secondary_model)):
            # Explicitly require the custom weights path to exist as requested
            raise FileNotFoundError(f"Secondary YOLO weights not found: {secondary_model}")
        self.secondary = YOLO(secondary_model)
        self.device = device

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.4, class_conf: Optional[Dict[str, float]] = None) -> List[Dict]:
        if frame is None:
            return []
        out_a = _predict_one(self.primary, frame, self.device, conf_thresh, source_tag="primary", class_conf=class_conf)
        out_b = _predict_one(self.secondary, frame, self.device, conf_thresh, source_tag="secondary", class_conf=class_conf)
        return out_a + out_b


__all__ = ["DualYoloDetector"]


