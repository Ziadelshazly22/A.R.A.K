"""
Primary backend pipeline for A.R.A.K: orchestrates detectors, scoring, logging, and annotation.

Usage (CLI)
-----------
    python src/pipeline.py --session SID --student STUD --webcam
    python src/pipeline.py --session SID --student STUD --video data/samples/sample.mp4

This module is also imported by the Streamlit UI, which manages the webcam loop and
provides controls like pause/resume and manual snapshots.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import cv2
import numpy as np
import yaml

from src.detectors.yolo_detector import YoloDetector
from src.detectors.gaze_detector import GazeDetector
from src.logic.suspicion_scoring import (
    ScoringConfig,
    TemporalHistory,
    compute_suspicion,
)
from src.logger import EventLogger


def load_config_yaml(path: str) -> ScoringConfig:
    """Load scoring configuration from a YAML file if it exists, else defaults.

    The returned ScoringConfig includes class names which the pipeline passes to
    the YOLO detector to align class indices with the trained model.
    """
    if not os.path.exists(path):
        return ScoringConfig()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return ScoringConfig(
        alert_threshold=cfg.get("alert_threshold", 5),
        phone_conf=float(cfg.get("phone_conf", 0.45)),
        classes=cfg.get("classes"),
        weights=cfg.get("weights", {}),
        allow_book=bool(cfg.get("allow_book", False)),
        allow_calculator=bool(cfg.get("allow_calculator", False)),
        gaze_duration_threshold=float(cfg.get("gaze_duration_threshold", 2.5)),
        repeat_dir_threshold=int(cfg.get("repeat_dir_threshold", 2)),
        repeat_window_sec=float(cfg.get("repeat_window_sec", 10.0)),
    )


def annotate_frame(frame, detections: List[Dict], gaze_state: Dict, score: int) -> np.ndarray:
    """Draw detection boxes, labels, and gaze/score overlay onto the frame.

    The annotation style uses red for risky items (phone/earphone), green for person,
    and yellow-ish for other classes. The text overlay provides gaze and score info.
    """
    out = frame.copy()
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        name = det["class_name"]
        conf = det.get("conf", 0.0)
        color = (0, 255, 0) if name == "person" else (0, 200, 255)
        if name in ("phone", "earphone"):
            color = (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{name}:{conf:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    # Gaze overlay
    gaze = gaze_state.get("gaze", "uncertain")
    yaw = gaze_state.get("head_pose", {}).get("yaw", 0.0)
    pitch = gaze_state.get("head_pose", {}).get("pitch", 0.0)
    cv2.putText(
        out,
        f"gaze:{gaze} yaw:{yaw:.1f} pitch:{pitch:.1f} score:{score}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


class ProcessingPipeline:
    """End-to-end per-frame processing orchestrator.

    On each frame: YOLO detect -> Gaze detect -> Rule scoring -> Annotate -> Log row

    The pipeline also maintains the last processed info to support a UI "snapshot now"
    action that saves the most recent frame to disk and logs it as a special event.
    """
    def __init__(
        self,
        session_id: str,
        student_id: str,
        config_path: str = os.path.join("src", "logic", "config.yaml"),
        device: Optional[str] = None,
    ):
        self.session_id = session_id
        self.student_id = student_id
        self.cfg = load_config_yaml(config_path)
        classes = self.cfg.classes if isinstance(getattr(self.cfg, 'classes', None), list) else None
        self.yolo = YoloDetector(device=device, class_names=classes)
        self.gaze = GazeDetector()
        self.logger = EventLogger(session_id=session_id, student_id=student_id)
        self.history = TemporalHistory(maxlen=300)
        self.frame_id = 0
        # Keep last frame info for "Snapshot now" from UI
        self.last_annotated_frame = None
        self.last_score = 0
        self.last_events = []
        self.last_main_conf = 0.0
        self.last_main_bbox = [0.0, 0.0, 0.0, 0.0]
        self.last_gaze_state = {}

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, List[str], bool]:
        """Process a single BGR frame and return (annotated, score, events, is_alert)."""
        detections = self.yolo.detect(frame)
        gaze_state = self.gaze.process(frame)
        score, events, is_alert = compute_suspicion(
            detections, gaze_state, self.history, self.cfg
        )

        annotated = annotate_frame(frame, detections, gaze_state, score)

        # Log primary event per frame (Normal or SUS)
        event_type = "SUS" if is_alert else "NORMAL"
        event_subtype = ";".join(events) if events else "none"
        main_conf = max([d.get("conf", 0.0) for d in detections], default=0.0)
        main_bbox = max(
            [d.get("bbox", [0, 0, 0, 0]) for d in detections],
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]) if b else 0,
            default=[0, 0, 0, 0],
        )
        self.logger.log_event(
            frame_id=self.frame_id,
            event_type=event_type,
            event_subtype=event_subtype,
            confidence=float(main_conf),
            bbox=[float(x) for x in main_bbox],
            head_pose=gaze_state.get("head_pose", {}),
            gaze=gaze_state.get("gaze", "uncertain"),
            suspicion_score=score,
            is_alert=is_alert,
            annotated_frame=annotated,
        )

        # Store last info for snapshot helper
        self.last_annotated_frame = annotated
        self.last_score = score
        self.last_events = events
        self.last_main_conf = float(main_conf)
        self.last_main_bbox = [float(x) for x in main_bbox]
        self.last_gaze_state = dict(gaze_state)

        self.frame_id += 1
        return annotated, score, events, is_alert

    def snapshot_now(self, label: str = "SNAPSHOT") -> Optional[str]:
        """Save a manual snapshot of the last processed frame and return status.

        The snapshot is logged as a dedicated row with is_alert=True to force image
        saving, using the last frame and score already stored on the object.
        """
        if self.last_annotated_frame is None:
            return None
        # Log as a special event, forcing a snapshot
        self.logger.log_event(
            frame_id=max(0, self.frame_id - 1),
            event_type=label,
            event_subtype="manual",
            confidence=self.last_main_conf,
            bbox=self.last_main_bbox,
            head_pose=self.last_gaze_state.get("head_pose", {}),
            gaze=self.last_gaze_state.get("gaze", "uncertain"),
            suspicion_score=self.last_score,
            is_alert=True,
            annotated_frame=self.last_annotated_frame,
        )
        return "ok"


def run_realtime(args):
    """Minimal CLI loop for webcam or video file, printing periodic status.

    This avoids cv2.imshow to remain compatible with headless environments. The
    Streamlit app should be preferred for interactive review and control.
    """
    cap = cv2.VideoCapture(0) if args.webcam else cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video source")

    pipeline = ProcessingPipeline(
        session_id=args.session,
        student_id=args.student,
        device=None,
    )
    last = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, score, events, is_alert = pipeline.process_frame(frame)
        # For headless environments, avoid imshow; print a lightweight status instead.
        now = time.time()
        if now - last >= 1.0:
            print(f"score={score} alert={is_alert} events={events[:2]} ...")
            last = now
    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def parse_args():
    """Parse simple CLI arguments for demo/testing of the pipeline."""
    p = argparse.ArgumentParser()
    p.add_argument("--session", default="demo-session")
    p.add_argument("--student", default="student-001")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--webcam", action="store_true")
    src.add_argument("--video", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.webcam and not args.video:
        args.webcam = True
    run_realtime(args)
