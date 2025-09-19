## A.R.A.K — Academic Resilience & Authentication Kernel

Local, rules-based proctoring engine with YOLO object detection, MediaPipe gaze estimation, temporal suspicion scoring, CSV/Excel logging, and a Streamlit UI.

### Quick setup

On Windows (double-click):

- Run `scripts/QuickSetup.bat` to create a venv and install requirements automatically.
	- Alternatively, run it from PowerShell:
		- `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\QuickSetup.ps1`

1) Create and activate a virtual environment and install deps:

```bash
bash scripts/setup_venv.sh
```

On Windows (PowerShell), use Git Bash or WSL for the Bash script; or do it manually:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Run Streamlit UI:

```powershell
streamlit run src/ui/streamlit_app.py
```

3) Optional: run pipeline from CLI (webcam by default):

```powershell
python src/pipeline.py --webcam --session demo --student s001
```

### Models

- Place your YOLO weights at `models/model_bestV3.pt`. If missing, the app falls back to `yolov8n.pt`.
- Class mapping defaults to: `['person','phone','book','earphone']`.

### Config

Defaults live in `src/logic/config.yaml` and can be edited via the Settings page:

- alert_threshold (default 5)
- phone_conf (0.45)
- weights: phone (5), earphone (4), person (5), book (3), calculator (3), gaze_off_per_sec (1), repetitive_head (2)
- allow_book, allow_calculator (false)
- gaze_duration_threshold (2.5s)
- repeat_dir_threshold (2), repeat_window_sec (10s)

### Pages overview

- Home: branding and quick intro
- Live Detection: webcam processing with on-screen score and events
- Upload Video: upload a video and process frames offline; see results in Logs
- Settings: adjust thresholds and weights and allowed items
- Logs & Review: view CSV logs and snapshots, filter and export to CSV/Excel

### Security & Privacy

- Use with participant consent only. Snapshots are stored locally under `logs/snapshots/` per session ID.
- Define a retention policy that matches your institution's requirements and delete logs after use.

### Design notes

- Performance vs Accuracy: YOLOv8n fallback is light and CPU-friendly but less accurate; custom weights recommended for best results. MediaPipe face mesh runs on CPU and may limit FPS. The pipeline aims for ~5 FPS on typical laptops; adjust webcam resolution to improve speed.

### Troubleshooting

- If webcam fails in Streamlit, try closing other apps that use the camera or run the CLI pipeline.
- If ultralytics downloads yolov8n.pt on first run, ensure internet access for the first execution; afterwards, it is cached locally.

