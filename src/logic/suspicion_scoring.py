"""
Suspicion scoring engine.

This module fuses object detections (from YOLO) and gaze state (from GazeDetector)
into a numeric "suspicion score" per frame, plus a list of triggered events and a
boolean alert flag.

The configuration is now managed through config_manager for optimized settings.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

from .config_manager import get_config, get_detection_weights, is_item_allowed


@dataclass
class ScoringConfig:
    """Configuration for suspicion scoring - now loads from optimized config."""
    
    def __init__(self):
        """Initialize with optimized settings from config manager."""
        config = get_config()
        
        # Core thresholds - pre-optimized for best accuracy
        self.alert_threshold = config.get('alert_threshold', 4)
        self.phone_conf = config.get('phone_conf', 0.50)
        self.classes = config.get('classes', ['person', 'phone', 'book', 'earphone', 'calculator'])
        
        # Detection weights - optimized and dynamically adjusted for exam policy
        self.weights = get_detection_weights()
        
        # Advanced detector settings - pre-optimized
        self.detector_primary = config.get('detector_primary', 'yolo11m.pt')
        self.detector_secondary = config.get('detector_secondary', 'models/model_bestV3.pt')
        self.detector_conf = config.get('detector_conf', 0.40)
        self.detector_merge_nms = config.get('detector_merge_nms', True)
        self.detector_nms_iou = config.get('detector_nms_iou', 0.45)
        self.detector_merge_mode = config.get('detector_merge_mode', 'wbf')
        self.class_conf = config.get('class_conf', {})
        
        # Gaze monitoring - optimized for accuracy
        self.gaze_duration_threshold = config.get('gaze_duration_threshold', 2.5)
        
        # Repetitive movement detection - tuned for real suspicious behavior
        self.repeat_dir_threshold = config.get('repeat_dir_threshold', 4)
        self.repeat_window_sec = config.get('repeat_window_sec', 12.0)
    
    @property
    def allow_book(self) -> bool:
        return is_item_allowed('book')
    
    @property 
    def allow_calculator(self) -> bool:
        return is_item_allowed('calculator')
    
    @property
    def allow_notebook(self) -> bool:
        return is_item_allowed('notebook')
    
    @property
    def allow_earphones(self) -> bool:
        return is_item_allowed('earphones')


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
    """Compute suspicion score, events, and is_alert with optimized detection logic.

    Parameters
    ----------
    detections: list of {class_name, class_id, conf, bbox}
        Output from YoloDetector.detect for the current frame.
    gaze_state: dict
        Output from GazeDetector.process for the current frame.
    history: TemporalHistory
        Mutable structure tracking recent behavior across frames.
    config: ScoringConfig
        Optimized thresholds and weights from config manager.
    """
    score = 0
    events: List[str] = []

    # High-priority object detection (always flagged regardless of policy)
    for det in detections:
        name = str(det.get("class_name", ""))
        conf = float(det.get("conf", 0.0))
        
        if name == "phone" and conf >= config.phone_conf:
            score += int(config.weights.get("phone", 8))
            events.append("CRITICAL_VIOLATION:phone_detected")
        elif name == "earphone" and not config.allow_earphones:
            score += int(config.weights.get("earphone", 6))
            events.append("CRITICAL_VIOLATION:earphone_detected")
        elif name == "smartwatch":
            score += int(config.weights.get("smartwatch", 6))
            events.append("CRITICAL_VIOLATION:smartwatch_detected")
        elif name == "person":
            # Another person in frame besides examinee
            score += int(config.weights.get("person", 7))
            events.append("CRITICAL_VIOLATION:unauthorized_person")

    # Policy-dependent object detection
    for det in detections:
        name = str(det.get("class_name", ""))
        
        if name == "book" and not config.allow_book:
            score += int(config.weights.get("book", 4))
            events.append("POLICY_VIOLATION:unauthorized_book")
        elif name == "calculator" and not config.allow_calculator:
            score += int(config.weights.get("calculator", 4))
            events.append("POLICY_VIOLATION:unauthorized_calculator")
        elif name == "notebook" and not config.allow_notebook:
            w = int(config.weights.get("notebook", 5))
            if w > 0:
                score += w
                events.append("POLICY_VIOLATION:unauthorized_laptop")

    # Enhanced gaze behavior monitoring
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
                    events.append(f"BEHAVIORAL:sustained_gaze_off_{gaze}")
    else:
        # Reset timer once gaze returns
        history.gaze_off_start = None

    # Enhanced repetitive head movement detection
    yaw = float(gaze_state.get("head_pose", {}).get("yaw", 0.0))
    dir_label = "left" if yaw < -8 else "right" if yaw > 8 else "center"
    
    if dir_label in ("left", "right"):
        history.add_head_dir(dir_label)
        now = time.time()
        # Count occurrences within window - more strict threshold
        cnt = sum(1 for ts, d in history.head_dir_hist 
                 if d == dir_label and now - ts <= config.repeat_window_sec)
        if cnt >= config.repeat_dir_threshold:
            score += int(config.weights.get("repetitive_head", 3))
            events.append(f"BEHAVIORAL:suspicious_head_movement_{dir_label}")

    is_alert = bool(score >= config.alert_threshold)
    return int(score), events, is_alert


__all__ = ["ScoringConfig", "TemporalHistory", "compute_suspicion"]
