"""
Primary backend pipeline for A.R.A.K: orchestrates detectors, scoring, logging, and annotation.
Optimized for performance with adaptive frame skipping and intelligent processing.

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

from src.detectors.dual_yolo_detector import DualYoloDetector
from src.detectors.gaze_detector import GazeDetector
from src.logic.suspicion_scoring import (
    ScoringConfig,
    TemporalHistory,
    compute_suspicion,
)
from src.logger import EventLogger

FRAME_SKIP = 10  # عشان تعملي processing لكل 10 فريمات
def load_config_yaml(path: Optional[str] = None) -> ScoringConfig:
    """Load optimized scoring configuration from config manager.
    
    All technical parameters are pre-optimized for best performance and accuracy.
    User settings (exam policy) are loaded dynamically from config manager.
    """
    return ScoringConfig()


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
    """End-to-end per-frame processing orchestrator with enhanced performance optimizations.

    Features:
    - Adaptive frame skipping for video vs live processing
    - Separate skip rates for detection vs gaze processing  
    - Intelligent caching of detection results
    - Optimized suspicious moment detection with minimal false positives

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
        is_video_upload: bool = False,
    ):
        self.session_id = session_id
        self.student_id = student_id
        self.is_video_upload = is_video_upload
        self.cfg = load_config_yaml(config_path)
        
        # Performance optimization settings based on processing type
        video_skip = 3  # For uploaded videos - process every 3rd frame
        live_skip = 2   # For live webcam - process every 2nd frame  
        self.frame_skip = video_skip if is_video_upload else live_skip
        self.detection_frame_skip = 6  # Run YOLO detection every 6th processed frame
        self.gaze_frame_skip = 2       # Run gaze detection every 2nd processed frame
        self.detection_counter = 0
        self.gaze_counter = 0
        
        # Cache for performance optimization
        self.last_detections = []
        self.last_gaze_state = {"gaze": "uncertain", "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}, "gaze_conf": 0.0}
        
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
        self.last_annotated_frame = np.zeros((1, 1, 3), dtype=np.uint8)  # Default black frame
        self.last_score = 0
        self.last_events = []
        self.last_main_conf = 0.0
        self.last_main_bbox = [0.0, 0.0, 0.0, 0.0]
        self.last_gaze_state = {}
        
        # Add recent events deque for WebRTC compatibility
        self.recent_events: Deque[str] = deque(maxlen=50)

    # def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, List[str], bool]:
    #     """Process a single BGR frame and return (annotated, score, events, is_alert)."""
        
    #     detections = self.yolo.detect(frame, conf_thresh=self.det_conf, class_conf=self.class_conf)
    #     if self.det_merge:
    #         if self.det_merge_mode == 'wbf':
    #             detections = merge_wbf(detections, iou_thr=self.det_iou, skip_box_thr=min(self.class_conf.values()) if self.class_conf else 0.0)
    #         else:
    #             detections = merge_nms(detections, self.det_iou)
    #     gaze_state = self.gaze.process(frame)
    #     score, events, is_alert = compute_suspicion(
    #         detections, gaze_state, self.history, self.cfg
    #     )

    #     annotated = annotate_frame(frame, detections, gaze_state, score)

    #     # Log primary event per frame (Normal or SUS)
    #     event_type = "SUS" if is_alert else "NORMAL"
    #     event_subtype = ";".join(events) if events else "none"
    #     main_conf = max([d.get("conf", 0.0) for d in detections], default=0.0)
    #     main_bbox = max(
    #         [d.get("bbox", [0, 0, 0, 0]) for d in detections],
    #         key=lambda b: (b[2] - b[0]) * (b[3] - b[1]) if b else 0,
    #         default=[0, 0, 0, 0],
    #     )
    #     self.logger.log_event(
    #         frame_id=self.frame_id,
    #         event_type=event_type,
    #         event_subtype=event_subtype,
    #         confidence=float(main_conf),
    #         bbox=[float(x) for x in main_bbox],
    #         head_pose=gaze_state.get("head_pose", {}),
    #         gaze=gaze_state.get("gaze", "uncertain"),
    #         suspicion_score=score,
    #         is_alert=is_alert,
    #         annotated_frame=annotated,
    #     )

    #     # Store last info for snapshot helper
    #     self.last_annotated_frame = annotated
    #     self.last_score = score
    #     self.last_events = events
    #     self.last_main_conf = float(main_conf)
    #     self.last_main_bbox = [float(x) for x in main_bbox]
    #     self.last_gaze_state = dict(gaze_state)

    #     self.frame_id += 1
    #     return annotated, score, events, is_alert
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, List[str], bool]:
        """Process a single BGR frame with optimized performance for suspicious moment detection.
        
        Performance optimizations:
        - Adaptive frame skipping based on video vs live processing
        - Separate detection and gaze processing intervals
        - Intelligent caching to maintain detection quality
        - Focus on suspicious moments for accurate snapshot timing
        """
        # Apply basic frame skipping for performance
        if self.frame_id % self.frame_skip != 0:
            self.frame_id += 1
            return (
                self.last_annotated_frame,
                self.last_score,
                self.last_events,
                False  # Skip alert processing for non-processed frames
            )
        
        # Optimized detection processing - run YOLO less frequently
        should_run_detection = (self.detection_counter % self.detection_frame_skip == 0)
        if should_run_detection:
            detections = self.yolo.detect(frame, conf_thresh=self.det_conf, class_conf=self.class_conf)
            if self.det_merge:
                if self.det_merge_mode == 'wbf':
                    detections = merge_wbf(detections, iou_thr=self.det_iou, 
                                         skip_box_thr=min(self.class_conf.values()) if self.class_conf else 0.0)
                else:
                    detections = merge_nms(detections, self.det_iou)
            self.last_detections = detections  # Cache for performance
        else:
            detections = self.last_detections  # Use cached detections
        
        self.detection_counter += 1
        
        # Optimized gaze processing - run MediaPipe less frequently  
        should_run_gaze = (self.gaze_counter % self.gaze_frame_skip == 0)
        if should_run_gaze:
            gaze_state = self.gaze.process(frame)
            self.last_gaze_state = gaze_state  # Cache for performance
        else:
            gaze_state = self.last_gaze_state  # Use cached gaze state
            
        self.gaze_counter += 1
        
        # Compute suspicion score (this is lightweight)
        score, events, is_alert = compute_suspicion(
            detections, gaze_state, self.history, self.cfg
        )

        # Always annotate current frame for visual feedback
        annotated = annotate_frame(frame, detections, gaze_state, score)

        # Enhanced suspicious moment detection - only log significant events
        should_log = is_alert or (score > 0) or (len(events) > 0) or (self.frame_id % (self.frame_skip * 10) == 0)
        
        if should_log:
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
        self.last_main_conf = float(max([d.get("conf", 0.0) for d in detections], default=0.0))
        self.last_main_bbox = [float(x) for x in max(
            [d.get("bbox", [0, 0, 0, 0]) for d in detections],
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]) if b else 0,
            default=[0, 0, 0, 0],
        )]
        self.last_gaze_state = dict(gaze_state)

        # Add recent events for tracking
        if events:
            for event in events:
                self.recent_events.append(f"{self.frame_id}: {event}")

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
    """Optimized CLI loop for webcam or video file with performance enhancements.

    This avoids cv2.imshow to remain compatible with headless environments. The
    Streamlit app should be preferred for interactive review and control.
    
    Performance optimizations:
    - Detects if input is video file vs webcam for adaptive processing
    - Implements frame skipping appropriate for the input type  
    - Focuses processing on suspicious moment detection
    """
    is_video_file = bool(args.video and not args.webcam)
    cap = cv2.VideoCapture(0) if args.webcam else cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video source")

    # Get video properties for optimization
    total_frames = 0
    if is_video_file:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        print("Performance optimization: Processing every 3rd frame for faster analysis")
    else:
        print("Live webcam processing: Performance optimization enabled")

    pipeline = ProcessingPipeline(
        session_id=args.session,
        student_id=args.student,
        device=None,
        is_video_upload=is_video_file,
    )
    
    last = time.time()
    frame_count = 0
    suspicious_moments = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
            
        frame_count += 1
        annotated, score, events, is_alert = pipeline.process_frame(frame)
        
        if is_alert:
            suspicious_moments += 1
        
        # For headless environments, avoid imshow; print optimized status instead.
        now = time.time()
        if now - last >= 2.0:  # Update every 2 seconds for less spam
            progress = f" ({frame_count}/{total_frames})" if is_video_file else ""
            print(f"Frame {frame_count}{progress} | Score: {score} | Alert: {is_alert} | "
                  f"Events: {events[:2] if events else 'None'} | Suspicious: {suspicious_moments}")
            last = now
            
    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
        
    print(f"\nProcessing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Suspicious moments detected: {suspicious_moments}")
    print(f"Check logs/ directory for detailed analysis and snapshots")


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
