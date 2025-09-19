"""
MediaPipe Face Mesh-based gaze direction estimator.

This class detects eye regions using MediaPipe and estimates gaze direction by
tracking iris centers relative to eyelid landmarks. It's designed to be robust
enough for proctoring heuristics without full 3D head pose estimation.

Public API
----------
- calibrate(frame): Use current frame to set a baseline eye center while the
	user looks at the screen/camera. Improves on-screen vs off-screen separation.
- process(frame): Return a dictionary:
	{
		'gaze': 'on_screen'|'off_left'|'off_right'|'up'|'down'|'uncertain',
		'gaze_conf': float in [0,1],
		'head_pose': { 'pitch': float, 'yaw': float, 'roll': float },
		'pupil_rel': { 'x': float, 'y': float }
	}

Notes
-----
- The "head_pose" here is a rough proxy derived from relative iris locations; for
	real yaw/pitch/roll you would fit a 3D face model and use solvePnP.
- A small moving average buffer is used to stabilize noisy single-frame estimates.
"""
from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Dict, Optional
from mediapipe.python.solutions import face_mesh

class GazeDetector:
	# Landmark indices based on MediaPipe Face Mesh
	LEFT_EYE_IDX = [33, 133]
	RIGHT_EYE_IDX = [362, 263]
	LEFT_IRIS_CENTER = 468
	RIGHT_IRIS_CENTER = 473
	RIGHT_TOP = 159
	RIGHT_BOTTOM = 145
	LEFT_TOP = 386
	LEFT_BOTTOM = 374

	def __init__(self, smoothing: int = 5):
		"""Initialize MediaPipe FaceMesh and smoothing buffers.

		Parameters
		----------
		smoothing: int
			Window size of the moving average for iris positions to reduce jitter.
		"""

		self.mp_face_mesh = face_mesh
		self.face_mesh = self.mp_face_mesh.FaceMesh(
			refine_landmarks=True,
			max_num_faces=1,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
		)
		self.buffer_x, self.buffer_y = deque(maxlen=smoothing), deque(maxlen=smoothing)
		self.calibrated = False
		self.baseline_x, self.baseline_y = 0.5, 0.5

	def _normalize(self, c, mn, mx):
		"""Normalize a point c into [0,1] range using min (mn) and max (mx) corners."""
		return (
			(c[0] - mn[0]) / (mx[0] - mn[0] + 1e-6),
			(c[1] - mn[1]) / (mx[1] - mn[1] + 1e-6),
		)

	def calibrate(self, frame) -> bool:
		"""Calibrate baseline using current frame (user looks at the camera/screen).

		Returns True if a non-uncertain gaze was obtained and baseline stored.
		"""
		state = self.process(frame)
		if state["gaze"] != "uncertain":
			self.baseline_x = state["pupil_rel"]["x"]
			self.baseline_y = state["pupil_rel"]["y"]
			self.calibrated = True
			return True
		return False

	def process(self, frame) -> Dict:
		"""Process a BGR frame and estimate gaze direction and a rough head pose."""
		h, w = frame.shape[:2]
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = self.face_mesh.process(rgb)

		if not results.multi_face_landmarks:
			return {
				"gaze": "uncertain",
				"gaze_conf": 0.0,
				"head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
				"pupil_rel": {"x": 0.5, "y": 0.5},
			}

		lm = results.multi_face_landmarks[0].landmark
		l_min = (
			int(lm[self.LEFT_EYE_IDX[0]].x * w),
			int(lm[self.LEFT_EYE_IDX[0]].y * h),
		)
		l_max = (
			int(lm[self.LEFT_EYE_IDX[1]].x * w),
			int(lm[self.LEFT_EYE_IDX[1]].y * h),
		)
		l_center = (
			int(lm[self.LEFT_IRIS_CENTER].x * w),
			int(lm[self.LEFT_IRIS_CENTER].y * h),
		)
		r_min = (
			int(lm[self.RIGHT_EYE_IDX[0]].x * w),
			int(lm[self.RIGHT_EYE_IDX[0]].y * h),
		)
		r_max = (
			int(lm[self.RIGHT_EYE_IDX[1]].x * w),
			int(lm[self.RIGHT_EYE_IDX[1]].y * h),
		)
		r_center = (
			int(lm[self.RIGHT_IRIS_CENTER].x * w),
			int(lm[self.RIGHT_IRIS_CENTER].y * h),
		)

		lx, ly = self._normalize(l_center, l_min, l_max)
		rx, ry = self._normalize(r_center, r_min, r_max)
		nx = (lx + rx) / 2
		ny = (ly + ry) / 2

	# Combine with absolute pupil center for minor stabilization; this helps when eyelid
	# corners are partially occluded but the iris centers are still tracked well.
		pupil_x = (l_center[0] + r_center[0]) / 2 / max(w, 1)
		pupil_y = (l_center[1] + r_center[1]) / 2 / max(h, 1)
		nx = 0.7 * nx + 0.3 * pupil_x
		ny = 0.7 * ny + 0.3 * pupil_y

		self.buffer_x.append(nx)
		self.buffer_y.append(ny)
		nx = float(np.mean(self.buffer_x))
		ny = float(np.mean(self.buffer_y))

		# Calibration shift
		if self.calibrated:
			nx -= (self.baseline_x - 0.5)
			ny -= (self.baseline_y - 0.5)

		# Vertical ratio from iris and eyelid landmarks
		r_iris_y = lm[self.RIGHT_IRIS_CENTER].y * h
		r_top_y = lm[self.RIGHT_TOP].y * h
		r_bottom_y = lm[self.RIGHT_BOTTOM].y * h
		l_iris_y = lm[self.LEFT_IRIS_CENTER].y * h
		l_top_y = lm[self.LEFT_TOP].y * h
		l_bottom_y = lm[self.LEFT_BOTTOM].y * h
		r_ratio = (r_iris_y - r_top_y) / (r_bottom_y - r_top_y + 1e-6)
		l_ratio = (l_iris_y - l_top_y) / (l_bottom_y - l_top_y + 1e-6)
		iris_vert_ratio = (r_ratio + l_ratio) / 2.0

		# Decide gaze direction using simple thresholds on normalized iris positions
		gaze = "on_screen"
		gaze_conf = 0.6
		if nx < 0.45:
			gaze = "off_left"
			gaze_conf = min(1.0, (0.45 - nx) * 5 + 0.6)
		elif nx > 0.60:
			gaze = "off_right"
			gaze_conf = min(1.0, (nx - 0.60) * 5 + 0.6)

		down_thresh = 0.35
		if iris_vert_ratio > 0.45:
			gaze = "up"
			gaze_conf = min(1.0, (iris_vert_ratio - 0.45) * 2 + 0.6)
		elif iris_vert_ratio < down_thresh:
			gaze = "down"
			gaze_conf = min(1.0, (down_thresh - iris_vert_ratio) * 2 + 0.6)

	# Placeholder head pose (requires solvePnP for real 3D estimation)
		head_pose = {"pitch": float((0.5 - ny) * 30.0), "yaw": float((nx - 0.5) * 30.0), "roll": 0.0}

		return {
			"gaze": gaze,
			"gaze_conf": float(np.clip(gaze_conf, 0.0, 1.0)),
			"head_pose": head_pose,
			"pupil_rel": {"x": float(nx), "y": float(ny)},
		}


__all__ = ["GazeDetector"]
