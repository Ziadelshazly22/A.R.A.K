"""
Primary backend pipeline for A.R.A.K: orchestrates detectors, scoring, logging, and annotation.

Usage (CLI)
-----------
    python src/pipeline.py --session SID --student STUD --webcam
    python src/pipeline.py --session SID --student STUD --video data/samples/sample.mp4

This module is also imported by the Streamlit UI, which manages the webcam loop and
    python src/pipeline.py --session SID --student STUD --video data/samples/sample.mp4
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

from src.detectors.dual_yolo_detector import DualYoloDetector
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
        # Detector and thresholds extensions
        detector_primary=cfg.get("detector_primary", "yolo11m.pt"),
        detector_secondary=cfg.get("detector_secondary", os.path.join("models", "model_bestV3.pt")),
        detector_conf=float(cfg.get("detector_conf", 0.4)),
        detector_merge_nms=bool(cfg.get("detector_merge_nms", True)),
        detector_nms_iou=float(cfg.get("detector_nms_iou", 0.5)),
        detector_merge_mode=str(cfg.get("detector_merge_mode", "wbf")),
        class_conf=cfg.get("class_conf", {}),
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
        if name in ("phone", "earphone", "smartwatch"):
            color = (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        src_tag = det.get("source", "")
        label = f"{name}:{conf:.2f}" + (f" [{src_tag}]" if src_tag else "")
        cv2.putText(
            out,
            label,
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


def _iou(a: List[float], b: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        a (List[float]): Bounding box A [x1, y1, x2, y2].
        b (List[float]): Bounding box B [x1, y1, x2, y2].

    Returns:
        float: IoU value between 0.0 and 1.0.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # Calculate intersection
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    # Calculate union
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_nms(dets: List[Dict], iou_thr: float) -> List[Dict]:
    """Perform Non-Maximum Suppression (NMS) to merge overlapping detections.

    Args:
        dets (List[Dict]): List of detections, each with "bbox" and "conf".
        iou_thr (float): IoU threshold for suppression.

    Returns:
        List[Dict]: Filtered list of detections after NMS.
    """
    out: List[Dict] = []
    by_class: Dict[str, List[Dict]] = {}
    # Group detections by class
    for d in dets:
        by_class.setdefault(str(d.get("class_name", "")), []).append(d)
    # Apply NMS per class
    for cls, items in by_class.items():
        items_sorted = sorted(items, key=lambda x: float(x.get("conf", 0.0)), reverse=True)
        kept: List[Dict] = []
        for det in items_sorted:
            bb = det.get("bbox", [0, 0, 0, 0])
            if not kept:
                kept.append(det)
                continue
            # Keep detection if IoU with all kept detections is below threshold
            if all(_iou(bb, k.get("bbox", [0, 0, 0, 0])) < iou_thr for k in kept):
                kept.append(det)
        out.extend(kept)
    return out


def merge_wbf(dets: List[Dict], iou_thr: float, skip_box_thr: float = 0.0) -> List[Dict]:
    """Weighted Box Fusion (simple implementation) per class.

    This fuses overlapping boxes by averaging coordinates weighted by confidence.
    It keeps the max confidence among fused boxes and concatenates source tags.

    Args:
        dets: List of detections with keys: class_name, bbox, conf, source.
        iou_thr: IoU threshold to consider boxes as the same object.
        skip_box_thr: Minimum confidence to include a box in fusion.

    Returns:
        List of fused detections.
    """
    if not dets:
        return []
    out: List[Dict] = []
    by_class: Dict[str, List[Dict]] = {}
    for d in dets:
        if float(d.get("conf", 0.0)) < float(skip_box_thr):
            continue
        by_class.setdefault(str(d.get("class_name", "")), []).append(d)

    for cls, items in by_class.items():
        clusters: List[List[Dict]] = []
        for det in sorted(items, key=lambda x: -float(x.get("conf", 0.0))):
            bb = det.get("bbox", [0, 0, 0, 0])
            placed = False
            for cluster in clusters:
                # Compare with the rep box of the cluster (first item)
                if _iou(bb, cluster[0].get("bbox", [0, 0, 0, 0])) >= iou_thr:
                    cluster.append(det)
                    placed = True
                    break
            if not placed:
                clusters.append([det])

        # Fuse each cluster
        for cluster in clusters:
            if not cluster:
                continue
            total_w = sum(float(d.get("conf", 0.0)) for d in cluster)
            if total_w <= 0:
                # fallback: keep the highest conf
                best = max(cluster, key=lambda x: float(x.get("conf", 0.0)))
                out.append(best)
                continue
            # Weighted average of coordinates
            xs1 = sum(float(d["bbox"][0]) * float(d.get("conf", 0.0)) for d in cluster) / total_w
            ys1 = sum(float(d["bbox"][1]) * float(d.get("conf", 0.0)) for d in cluster) / total_w
            xs2 = sum(float(d["bbox"][2]) * float(d.get("conf", 0.0)) for d in cluster) / total_w
            ys2 = sum(float(d["bbox"][3]) * float(d.get("conf", 0.0)) for d in cluster) / total_w
            max_conf = max(float(d.get("conf", 0.0)) for d in cluster)
            # Merge source tags
            sources = sorted(set(str(d.get("source", "")) for d in cluster if d.get("source")))
            src = "+".join(sources)
            out.append({
                "class_name": cls,
                "class_id": int(cluster[0].get("class_id", -1)),
                "conf": float(max_conf),
                "bbox": [float(xs1), float(ys1), float(xs2), float(ys2)],
                "source": src,
            })
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
        # Initialize dual detectors: pretrained YOLOv11 and custom nano weights
        # Primary is name-based, secondary expects your weights at models/model_bestV3.pt
        # Detector settings from config
        primary = getattr(self.cfg, 'detector_primary', 'yolo11m.pt') if hasattr(self.cfg, 'detector_primary') else 'yolo11m.pt'
        secondary = getattr(self.cfg, 'detector_secondary', os.path.join('models', 'model_bestV3.pt')) if hasattr(self.cfg, 'detector_secondary') else os.path.join('models', 'model_bestV3.pt')
        self.det_conf = float(getattr(self.cfg, 'detector_conf', 0.4))
        self.det_merge = bool(getattr(self.cfg, 'detector_merge_nms', True))
        self.det_iou = float(getattr(self.cfg, 'detector_nms_iou', 0.5))
        self.det_merge_mode = str(getattr(self.cfg, 'detector_merge_mode', 'wbf')).lower()
        self.class_conf = dict(getattr(self.cfg, 'class_conf', {}))

        self.yolo = DualYoloDetector(
            primary_model=primary,
            secondary_model=secondary,
            device=device,
        )
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
        
        # Add recent events deque for WebRTC compatibility
        self.recent_events: Deque[str] = deque(maxlen=50)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, List[str], bool]:
        """Process a single BGR frame and return (annotated, score, events, is_alert)."""
        detections = self.yolo.detect(frame, conf_thresh=self.det_conf, class_conf=self.class_conf)
        if self.det_merge:
            if self.det_merge_mode == 'wbf':
                detections = merge_wbf(detections, iou_thr=self.det_iou, skip_box_thr=min(self.class_conf.values()) if self.class_conf else 0.0)
            else:
                detections = merge_nms(detections, self.det_iou)
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
        """Manual snapshots are disabled. Only automatic snapshots during suspicious moments are allowed.

        This method always returns "disabled" to indicate that manual snapshots
        are not permitted. Snapshots are only taken automatically during frame
        processing when is_alert=True (suspicious moments).
        
        Returns:
            "disabled" - Manual snapshots are not allowed
        """
        return "disabled"


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
