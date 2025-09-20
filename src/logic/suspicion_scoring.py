"""
Suspicion scoring engine.

This module fuses object detections (from YOLO) and gaze state (from GazeDetector)
into a numeric "suspicion score" per frame, plus a list of triggered events and a
boolean alert flag.

Hard rules (high severity)
-------------------------
- Detected phone (confidence >= phone_conf) -> add weight "phone"
- Detected earphone -> add weight "earphone"
- Detected person (another person besides examinee) -> add weight "person"

Soft rules (contextual, configurable)
------------------------------------
- Detected book/calculator when not allowed -> add their respective weights
- Sustained off-screen gaze: once gaze has been off for >= threshold seconds,
	add weight per second (gaze_off_per_sec)
- Repetitive head movement: within a time window, if the number of turns in the
	same direction exceeds the threshold, add "repetitive_head" weight

All thresholds and weights can be controlled via ScoringConfig (writable by
config.yaml and Streamlit Settings page).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple


@dataclass
class ScoringConfig:
	# Thresholds and weights
	alert_threshold: int = 5  # final score required to mark a frame as alert
	phone_conf: float = 0.45  # minimum confidence for phone detections to count
	classes: List[str] | None = None  # not used here but propagated to YOLO via pipeline
	weights: Dict[str, int] = field(
		default_factory=lambda: {
			"phone": 5,
			"earphone": 4,
			"smartwatch": 4,
			"person": 5,  # another person
			"book": 3,
			"calculator": 3,
			"notebook": 0,   # set >0 to penalize laptop/notebook presence
			# "monitor": 0,    # set >0 to penalize external monitor/TV presence
			"gaze_off_per_sec": 1,
			"repetitive_head": 2,
		}
	)
	# Detector settings (read from config.yaml, used by pipeline)
	detector_primary: str = "yolo11m.pt"
	detector_secondary: str = "models/model_bestV3.pt"
	detector_conf: float = 0.4
	detector_merge_nms: bool = True
	detector_nms_iou: float = 0.5
	# Merge mode: 'nms'(non-maximum suppression) or 'wbf' (weighted box fusion)
	detector_merge_mode: str = "wbf"
	# Per-class confidence thresholds (e.g., {"phone": 0.6, "earphone": 0.5})
	class_conf: Dict[str, float] = field(default_factory=dict)
	# Allowed items
	allow_book: bool = False      # set True to ignore 'book' detections for scoring
	allow_calculator: bool = False  # set True to ignore 'calculator' detections
	# Gaze settings
	gaze_duration_threshold: float = 2.5  # seconds before off-screen gaze increases score
	# Repetitive movement
	repeat_dir_threshold: int = 2  # how many same-direction turns within window
	repeat_window_sec: float = 10.0  # seconds window for repetitive movement rule


class TemporalHistory:
	"""Keeps short-term history for smoothing and temporal rules.

	Attributes
	----------
	events: deque[(timestamp, label)]
		A general-purpose history store for counting recent events.
	gaze_off_start: float | None
		When off-screen gaze started; None means currently on-screen.
	head_dir_hist: deque[(timestamp, 'left'|'right'|'center')]
		Records head movement direction estimates to detect repetition.
	"""

	def __init__(self, maxlen: int = 150):
		self.events: Deque[Tuple[float, str]] = deque(maxlen=maxlen)
		self.gaze_off_start: float | None = None
		self.head_dir_hist: Deque[Tuple[float, str]] = deque(maxlen=maxlen)

	def add_event(self, label: str):
		self.events.append((time.time(), label))

	def add_head_dir(self, direction: str):
		self.head_dir_hist.append((time.time(), direction))

	def count_recent(self, label: str, within_sec: float) -> int:
		now = time.time()
		return sum(1 for ts, lb in self.events if lb == label and now - ts <= within_sec)


def compute_suspicion(
	detections: List[Dict],
	gaze_state: Dict,
	history: TemporalHistory,
	config: ScoringConfig,
) -> Tuple[int, List[str], bool]:
	"""Compute suspicion score, events, and is_alert.

	Parameters
	----------
	detections: list of {class_name, class_id, conf, bbox}
		Output from YoloDetector.detect for the current frame.
	gaze_state: dict
		Output from GazeDetector.process for the current frame.
	history: TemporalHistory
		Mutable structure tracking recent behavior across frames.
	config: ScoringConfig
		Tunable thresholds and weights for the rules.
	"""
	score = 0
	events: List[str] = []

	# Hard-coded high severity objects
	for det in detections:
		name = str(det.get("class_name", ""))
		conf = float(det.get("conf", 0.0))
		if name == "phone" and conf >= config.phone_conf:
			score += config.weights.get("phone", 5)
			events.append("SUS_OBJECT:phone")
		elif name == "earphone":
			score += config.weights.get("earphone", 4)
			events.append("SUS_OBJECT:earphone")
		elif name == "smartwatch":
			score += config.weights.get("smartwatch", 4)
			events.append("SUS_OBJECT:smartwatch")
		elif name == "person":
			# Another person in frame besides examinee
			score += config.weights.get("person", 5)
			events.append("SUS_OBJECT:person")

	# Soft objects (book, calculator, notebook)
	for det in detections:
		name = str(det.get("class_name", ""))
		if name == "book" and not config.allow_book:
			score += config.weights.get("book", 3)
			events.append("SOFT_OBJECT:book")
		elif name == "calculator" and not config.allow_calculator:
			score += config.weights.get("calculator", 3)
			events.append("SOFT_OBJECT:calculator")
		elif name == "notebook":
			w = int(config.weights.get("notebook", 0))
			if w > 0:
				score += w
				events.append("SOFT_OBJECT:notebook")
		# elif name == "monitor":
		# 	w = int(config.weights.get("monitor", 0))
		# 	if w > 0:
		# 		score += w
		# 		events.append("SOFT_OBJECT:monitor")

	# Gaze behavior
	gaze = gaze_state.get("gaze", "uncertain")
	if gaze in ("off_left", "off_right", "up", "down"):
		# Start or continue off-screen timer
		if history.gaze_off_start is None:
			history.gaze_off_start = time.time()
		else:
			dur = time.time() - history.gaze_off_start
			if dur >= config.gaze_duration_threshold:
				add = int(dur * config.weights.get("gaze_off_per_sec", 1))
				score += add
				if add > 0:
					events.append("gaze_off_sustained")
	else:
		# Reset timer once gaze returns
		history.gaze_off_start = None

	# Repetitive head movement: count same-direction yaw turns recently
	yaw = float(gaze_state.get("head_pose", {}).get("yaw", 0.0))
	dir_label = "left" if yaw < -5 else "right" if yaw > 5 else "center"
	if dir_label in ("left", "right"):
		history.add_head_dir(dir_label)
		now = time.time()
		# count occurrences within window
		cnt = sum(1 for ts, d in history.head_dir_hist if d == dir_label and now - ts <= config.repeat_window_sec)
		if cnt >= config.repeat_dir_threshold:
			score += config.weights.get("repetitive_head", 2)
			events.append(f"repetitive_head:{dir_label}")

	# Temporal smoothing is partly embedded above via durations and recent counts.

	is_alert = score >= config.alert_threshold
	return score, events, is_alert


__all__ = ["ScoringConfig", "TemporalHistory", "compute_suspicion"]
