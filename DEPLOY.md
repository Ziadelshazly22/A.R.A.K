# DEPLOY: A.R.A.K — Academic Resilience & Authentication Kernel

## How to run locally

1. Create and activate a virtual environment and install dependencies.
   - Linux/macOS (or Git Bash/WSL on Windows):
     - bash scripts/setup_venv.sh
   - Native Windows PowerShell:
     - python -m venv venv
     - venv\Scripts\Activate.ps1
     - pip install -r requirements.txt
2. Start the Streamlit app:
   - streamlit run src/ui/streamlit_app.py

## Test with a sample video

- Put a video under data/samples/ (e.g., data/samples/demo.mp4), then in the UI open "Upload Video" and choose the file.
- Or run the pipeline directly:
  - python src/pipeline.py --video data/samples/demo.mp4 --session demo --student s001

## Where to find logs & snapshots

- Logs are stored as CSV files in logs/ (one per session): events_<session_id>.csv
- Alert snapshots are saved under logs/snapshots/<session_id>/ as JPG files. Each corresponds to an alerting frame.

## Proctor quick guide to alerts

- SUS_OBJECT:phone / earphone / person — Immediate high-severity events. A phone, earphone, or another person detected in the frame.
- SOFT_OBJECT:book / calculator — Context-dependent. If not allowed in Settings, they contribute to suspicion score.
- gaze_off_sustained — Gaze off-screen for longer than configured threshold; contributes per second.
- repetitive_head:left/right — Repeated head turns in the same direction within time window.

- The total suspicion_score is shown on the video overlay and in the Logs table. If it reaches the alert_threshold (default 5), an alert is raised and a snapshot is saved.

## Notes and trade-offs

- Performance vs accuracy: The default fallback model yolov8n is light and fast but less accurate. For better performance, place a tuned model at models/model_bestV3.pt.
- MediaPipe face mesh runs on CPU; to increase FPS, reduce camera resolution or processing frame rate.
- All processing is local; ensure you have consent from participants and a defined retention period for logs and snapshots.
