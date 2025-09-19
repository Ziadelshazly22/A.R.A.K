"""
Event logger utilities: append structured rows to a session CSV and save image snapshots.

What this module does
---------------------
- Creates a logs/ directory with a per-session CSV file named `events_{session_id}.csv`.
- Appends one row per frame (or special event) with detection/gaze metadata and score.
- Optionally saves a JPEG snapshot for alert frames in logs/snapshots/{session_id}/.

CSV schema (column order)
-------------------------
session_id, student_id, timestamp, frame_id, event_type, event_subtype, confidence,
bbox, head_pose_pitch, head_pose_yaw, head_pose_roll, gaze, suspicion_score, snapshot_path

Notes
-----
- "bbox" is stored as a stringified list [x1, y1, x2, y2] for convenience.
- "event_type" is typically one of: NORMAL | SUS | SNAPSHOT (manual) but can be extended.
- "event_subtype" is a semicolon-separated list of rule triggers (e.g., "SUS_OBJECT:phone").
- "snapshot_path" is empty for non-alerts unless a manual snapshot is requested by the UI.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import pandas as pd
import cv2


LOGS_DIR = os.path.join("logs")
SNAPSHOTS_DIR = os.path.join(LOGS_DIR, "snapshots")


def ensure_dirs(session_id: str) -> None:
	"""Ensure the base logs/ folder and session snapshots folder exist.

	Parameters
	----------
	session_id: str
		Unique session identifier; used to segregate snapshot images.
	"""
	# logs/
	os.makedirs(LOGS_DIR, exist_ok=True)
	# logs/snapshots/{session_id}/
	os.makedirs(os.path.join(SNAPSHOTS_DIR, session_id), exist_ok=True)


@dataclass
class LogEvent:
	"""A single row in the events CSV file.

	All attributes are serialized in a fixed order by EventLogger.log_event.
	"""
	session_id: str
	student_id: str
	timestamp: float
	frame_id: int
	event_type: str
	event_subtype: str
	confidence: float
	bbox: str
	head_pose_pitch: float
	head_pose_yaw: float
	head_pose_roll: float
	gaze: str
	suspicion_score: int
	snapshot_path: str


class EventLogger:
	"""Lightweight CSV-based logger with optional image snapshot saving.

	Parameters
	----------
	session_id: str
		Unique session identifier; forms part of filenames.
	student_id: str
		ID or label of the examinee for filtering and auditing.
	"""

	def __init__(self, session_id: str, student_id: str):
		self.session_id = session_id
		self.student_id = student_id
		ensure_dirs(session_id)
		self.csv_path = os.path.join(LOGS_DIR, f"events_{session_id}.csv")
		if not os.path.exists(self.csv_path):
			with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
				writer = csv.writer(f)
				writer.writerow(
					[
						"session_id",
						"student_id",
						"timestamp",
						"frame_id",
						"event_type",
						"event_subtype",
						"confidence",
						"bbox",
						"head_pose_pitch",
						"head_pose_yaw",
						"head_pose_roll",
						"gaze",
						"suspicion_score",
						"snapshot_path",
					]
				)

	def log_event(
		self,
		frame_id: int,
		event_type: str,
		event_subtype: str,
		confidence: float,
		bbox: List[float],
		head_pose: Dict[str, float],
		gaze: str,
		suspicion_score: int,
		is_alert: bool,
		annotated_frame: Optional[Any] = None,
	) -> None:
		"""Append a row to the per-session CSV and optionally save a snapshot image.

		When `is_alert` is True and `annotated_frame` is provided, a JPEG snapshot is
		written to logs/snapshots/{session_id}/ with a timestamp-based filename.

		Parameters
		----------
		frame_id: int
			Zero-based index of the processed frame in this session.
		event_type: str
			Logical type of the event (e.g., NORMAL | SUS | SNAPSHOT).
		event_subtype: str
			Further details (e.g., semicolon-separated triggers) or "none".
		confidence: float
			Confidence of the main detection rendered in the frame (best box).
		bbox: List[float]
			Bounding box [x1, y1, x2, y2] of the main detection (optional).
		head_pose: Dict[str, float]
			Dictionary with keys pitch/yaw/roll (degrees), if available.
		gaze: str
			Gaze label from the gaze detector (e.g., on_screen, off_left, ...).
		suspicion_score: int
			Computed suspicion score for this frame.
		is_alert: bool
			Whether this frame meets or exceeds the alert threshold.
		annotated_frame: Optional[cv2.Mat]
			The image to save if an alert snapshot is desired.
		"""
		ts = time.time()  # seconds since epoch
		snapshot_path = ""
		# Save snapshot only when alerting (or when the caller forces is_alert=True for manual snapshots)
		if is_alert and annotated_frame is not None:
			snap_dir = os.path.join(SNAPSHOTS_DIR, self.session_id)
			fname = f"{int(ts)}_{frame_id}.jpg"
			snapshot_path = os.path.join(snap_dir, fname)
			try:
				cv2.imwrite(snapshot_path, annotated_frame)
			except Exception:
				# Don't break logging if image writing fails for any reason
				snapshot_path = ""

		row = LogEvent(
			session_id=self.session_id,
			student_id=self.student_id,
			timestamp=ts,
			frame_id=frame_id,
			event_type=event_type,
			event_subtype=event_subtype,
			confidence=float(confidence or 0.0),
			bbox=str(bbox),
			head_pose_pitch=float(head_pose.get("pitch", 0.0)),
			head_pose_yaw=float(head_pose.get("yaw", 0.0)),
			head_pose_roll=float(head_pose.get("roll", 0.0)),
			gaze=gaze,
			suspicion_score=int(suspicion_score),
			snapshot_path=snapshot_path,
		)

		# Append to CSV in append mode; each call writes exactly one row.
		with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow([
				row.session_id,
				row.student_id,
				row.timestamp,
				row.frame_id,
				row.event_type,
				row.event_subtype,
				row.confidence,
				row.bbox,
				row.head_pose_pitch,
				row.head_pose_yaw,
				row.head_pose_roll,
				row.gaze,
				row.suspicion_score,
				row.snapshot_path,
			])

	def export(self, out_path: str) -> str:
		"""Export the per-session CSV to a new file.

		Behavior is selected from the output file extension:
		- .csv  -> writes CSV
		- .xls/.xlsx -> writes Excel (requires openpyxl)

		Returns
		-------
		str
			The path provided in `out_path` for convenience/chaining.
		"""
		if not os.path.exists(self.csv_path):
			raise FileNotFoundError("No events CSV yet")
		df = pd.read_csv(self.csv_path)
		ext = os.path.splitext(out_path)[1].lower()
		if ext in (".xlsx", ".xls"):
			df.to_excel(out_path, index=False)
		else:
			df.to_csv(out_path, index=False)
		return out_path


__all__ = ["EventLogger", "LOGS_DIR", "SNAPSHOTS_DIR"]

