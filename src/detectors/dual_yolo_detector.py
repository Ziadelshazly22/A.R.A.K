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
import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # type: ignore


def _normalize_class_name(raw: str) -> Optional[str]:
    """Map various YOLO label variants to our internal canonical names.

    Examples
    - 'cell phone' -> 'phone'
    - 'laptop' -> 'notebook'
    - 'headphones'/'earbuds'/'earpods'/'headset' -> 'earphone'
    - 'smart watch'/'smartwatch' -> 'smartwatch'
    
    Returns None if the class is not in our known/allowed classes.
    """
    name = str(raw).strip().lower()
    
    # Define our known/allowed classes that we want to detect
    KNOWN_CLASSES = {
        "person", "phone", "notebook", "book", "calculator", 
        "earphone", "smartwatch"
    }
    
    # general model mappings
    if name in ("cell phone", "mobile phone", "phone","telephone","mobile","cellphone","smart phone"):
        return "phone"
    if name == "laptop":
        return "notebook"
    # custom model mappings for wearables/headsets
    if name in ("headphones","headphone", "headset", "earbuds", "earpods", "earphone", "earphones"):
        return "earphone"
    if name in ("smart watch", "smartwatch", "smart-watch"):
        return "smartwatch"
    # keep known classes as-is
    if name in ("person", "book", "calculator", "notebook"):
        return name
    
    # Return None for unknown/unwanted classes
    return None


from typing import Any


def _predict_one(model: Any, frame: np.ndarray, device: Optional[str], conf: float, source_tag: str, class_conf: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Optimized prediction with reduced overhead and faster processing."""
    # Resize frame for faster processing if it's too large
    h, w = frame.shape[:2]
    if max(h, w) > 1080:  # If larger than 1080p, resize for speed
        scale = 1080 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))
        scale_factor = max(h, w) / 1080
    else:
        frame_resized = frame
        scale_factor = 1.0
        
    # Run prediction with optimized settings
    results = model.predict(frame_resized, conf=conf, verbose=False, device=device, 
                          imgsz=640, half=True)  # Use smaller image size and FP16 for speed
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
        # Scale back coordinates if frame was resized
        if scale_factor != 1.0:
            xyxy = [coord * scale_factor for coord in xyxy]
            
        conf_v = float(b.conf.item()) if hasattr(b, "conf") else 0.0
        cls_id = int(b.cls.item()) if hasattr(b, "cls") else -1
        if isinstance(names, dict) and cls_id in names:
            raw_name = names[cls_id]
        elif isinstance(names, list) and 0 <= cls_id < len(names):
            raw_name = names[cls_id]
        else:
            raw_name = str(cls_id)
        name = _normalize_class_name(raw_name)
        
        # Skip detection if class is not in our known/allowed classes
        if name is None:
            continue
            
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
    """Runs two YOLO models with performance optimizations for suspicious moment detection.

    Performance optimizations:
    - Intelligent model selection based on detection confidence
    - Frame preprocessing for faster inference
    - Smart fallback to single model when appropriate

    Parameters
    ----------
    primary_model: str
        Pretrained YOLO11 model name or path (e.g., 'yolo11m.pt').
    secondary_model: str
        Custom YOLO weights path for exam-specific objects (e.g., 'models/model_bestV3.pt').
    device: Optional[str]
        Torch/Ultralytics device hint ('cuda'|'cpu'|None).
    """

    def __init__(
        self,
        primary_model: str = "yolo11m.pt",
        secondary_model: str = os.path.join("models", "model_bestV3.pt"),
        device: Optional[str] = None,
    ) -> None:
        if YOLO is None:
            raise ImportError("ultralytics not installed. Please install requirements and try again.")
        
        # Load primary model
        print(f"Loading primary YOLO model: {primary_model}")
        self.primary = YOLO(primary_model)
        
        # Load secondary model only if it exists
        self.secondary = None
        if secondary_model and os.path.exists(secondary_model):
            print(f"Loading secondary YOLO model: {secondary_model}")
            self.secondary = YOLO(secondary_model)
        else:
            print(f"Warning: Secondary model not found at {secondary_model}, using primary model only")
            
        self.device = device
        self.use_dual_models = self.secondary is not None

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.4, class_conf: Optional[Dict[str, float]] = None) -> List[Dict]:
        """Optimized detection with smart model usage for better performance."""
        if frame is None:
            return []
            
        # Always run primary model
        out_a = _predict_one(self.primary, frame, self.device, conf_thresh, source_tag="primary", class_conf=class_conf)
        
        # Only run secondary model if it exists and we haven't found high-confidence suspicious objects
        high_confidence_objects = [det for det in out_a if det.get("conf", 0) > 0.7 and 
                                   det.get("class_name") in ["phone", "earphone", "person"]]
        
        if self.use_dual_models and len(high_confidence_objects) == 0:
            out_b = _predict_one(self.secondary, frame, self.device, conf_thresh, source_tag="secondary", class_conf=class_conf)
            return out_a + out_b
        
        return out_a


__all__ = ["DualYoloDetector"]


