"""
YOLO detector wrapper used across the app.

Overview
--------
This class abstracts ultralytics' YOLO inference so the rest of the codebase doesn't
depend on the library directly. It loads a custom model if available, otherwise a
small public model as a fallback. The detector returns a normalized list of
dicts per detection with consistent keys.

Detection output format
-----------------------
[{"class_name": str, "class_id": int, "conf": float, "bbox": [x1, y1, x2, y2]}]

Notes
-----
- "bbox" coordinates are in absolute pixel coordinates of the input frame.
- "class_names" can be provided externally (e.g. via config.yaml) to align indices
	with your trained model; otherwise we try model.names or fallback defaults.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

try:
	from ultralytics import YOLO
except Exception as e:  # pragma: no cover - ultralytics not installed in some environments
	YOLO = None  # type: ignore


DEFAULT_CLASSES = ["person", "phone", "book", "earphone"]


class YoloDetector:
	"""Thin wrapper around ultralytics.YOLO for inference.

	Parameters
	----------
	weights_path: str
		Path to custom weights. Defaults to 'models/model_bestV3.pt'.
	class_names: Optional[List[str]]
		List of class names indexed by numeric class id. If not provided, tries
		model.names from the loaded YOLO model and falls back to DEFAULT_CLASSES.
	device: Optional[str]
		Device string for torch/ultralytics (e.g., 'cuda', 'cpu'). If None, ultralytics decides.
	"""

	def __init__(
		self,
		weights_path: str = os.path.join("models", "model_bestV3.pt"),
		class_names: Optional[List[str]] = None,
		device: Optional[str] = None,
	):
		if YOLO is None:
			raise ImportError(
				"ultralytics not installed. Please install requirements and try again."
			)

		self.weights_path = weights_path
		self.device = device
		self._model = self._load_model()
		self.class_names = (
			class_names
			or (getattr(self._model, "names", None) if self._model is not None else None)
			or DEFAULT_CLASSES
		)

	def set_class_names(self, names: List[str]) -> None:
		"""Override class name mapping at runtime (e.g., after loading config).

		This is handy when you ship the same code to multiple models with different
		label orders. Passing an empty list is ignored.
		"""
		if names and isinstance(names, list):
			self.class_names = names

	def _load_model(self):
		"""Load the YOLO model (custom weights first, public fallback otherwise)."""
		# Prefer custom weights; fall back to a small public model if missing
		if YOLO is None:
			raise ImportError("ultralytics not installed; cannot load YOLO model")
		if os.path.exists(self.weights_path):
			model = YOLO(self.weights_path)
		else:
			# Fallback to a small public model available via ultralytics.
			# Note: this model's classes may not match your config list; that's OK for demo.
			model = YOLO("yolov11m.pt")
		return model

	def detect(self, frame: np.ndarray, conf_thresh: float = 0.4) -> List[Dict]:
		"""Run detection on a BGR frame and return normalized detection dicts.

		Parameters
		----------
		frame: np.ndarray
			BGR image as produced by OpenCV capture.
		conf_thresh: float
			Minimum confidence for predictions returned by ultralytics YOLO.

		Returns
		-------
		List[Dict]
			Each dict contains keys: class_name, class_id, conf, bbox
		"""
		if frame is None:
			return []

		results = self._model.predict(
			frame, conf=conf_thresh, verbose=False, device=self.device
		)
		out: List[Dict] = []
		if not results:
			return out
		res = results[0]
		boxes = getattr(res, "boxes", None)
		if boxes is None:
			return out
		# Iterate over Boxes object and normalize into plain dicts
		for b in boxes:
			xyxy = b.xyxy[0].tolist()
			conf = float(b.conf.item()) if hasattr(b, "conf") else 0.0
			cls_id = int(b.cls.item()) if hasattr(b, "cls") else -1
			name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)
			out.append(
				{
					"class_name": name,
					"class_id": cls_id,
					"conf": conf,
					"bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
				}
			)
		return out


__all__ = ["YoloDetector", "DEFAULT_CLASSES"]


