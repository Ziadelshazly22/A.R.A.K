from __future__ import annotations
import io
import os
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline import ProcessingPipeline
from src.logger import LOGS_DIR
import yaml
import importlib.util

# Optional dependency: streamlit-webrtc
# We only import these when available so the app still runs without WebRTC.
_webrtc_spec = importlib.util.find_spec("streamlit_webrtc")
HAS_WEBRTC = _webrtc_spec is not None
if HAS_WEBRTC:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode  # type: ignore
    import av  # type: ignore
else:
    webrtc_streamer = None  # type: ignore
    WebRtcMode = None  # type: ignore
    av = None  # type: ignore


ASSETS_DIR = os.path.join("src", "ui", "assets")
STYLE_PATH = os.path.join(ASSETS_DIR, "styles.css")
CONFIG_PATH = os.path.join("src", "logic", "config.yaml")

# Optional: simple team roster for the About page. Update these entries to your real team.
# You can safely edit this list or move it to a YAML later if you prefer.
TEAM_MEMBERS = [
    {"name": "Member 1", "role": "Role / Responsibility"},
    {"name": "Member 2", "role": "Role / Responsibility"},
    {"name": "Member 3", "role": "Role / Responsibility"},
]


def load_styles():
    """Inject enhanced CSS styles into the Streamlit app."""
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Add custom page config for better branding
    st.set_page_config(
        page_title="A.R.A.K - Academic Proctoring System",
        page_icon=os.path.join(ASSETS_DIR,"A_R_A_K_ICON"),
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Center the content along the main vertical diagonal of the sidebar
    st.markdown(
        """
        <style>
        .css-18e3th9 {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def load_cfg():
    """Read YAML settings used by the scoring engine and detectors."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_cfg(cfg):
    """Persist YAML settings; called from the Settings page."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def page_home():
    """Enhanced landing page with A.R.A.K branding and smooth animations."""
    # Logo section with enhanced styling
    logo_path = os.path.join(ASSETS_DIR,"A_R_A_K_Logo.svg")
    if os.path.exists(logo_path):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="logo-container fade-in">', unsafe_allow_html=True)
            # st.image(logo_path, use_column_width=True)
            st.image(logo_path, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced title with gradient and animation
    st.markdown("""
    <div class="fade-in">
        <h1 class="brand-title" style="font-size: 3.5rem; margin: 2rem 0;">
            A.R.A.K — أَرَاك
        </h1>
        <h2 class="brand-gradient-text" style="text-align: center; font-size: 1.5rem; margin-bottom: 2rem;">
            Academic Resilience & Authentication Kernel
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced description with glass card effect
    st.markdown("""
    <div class="glass-card fade-in" style="text-align: center; animation-delay: 0.3s;">
        <h3 style="color: var(--brand-accent); margin-bottom: 1rem;">🔒 Secure • 💻 Local • 🤖 AI-Powered Proctoring</h3>
        <p style="font-size: 1.1rem; line-height: 1.6; color: var(--text-secondary);">
            Experience next-generation academic integrity monitoring with real-time object detection, 
            gaze tracking, and behavioral analysis. All processing happens locally for maximum privacy and security.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights with animations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card slide-in-left metric-container" style="animation-delay: 0.5s;">
            <h4 style="color: var(--brand-primary);">🎯 Live Detection</h4>
            <p>Real-time webcam monitoring with instant alerts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card fade-in metric-container" style="animation-delay: 0.7s;">
            <h4 style="color: var(--brand-secondary);">📹 Video Analysis</h4>
            <p>Upload and analyze recorded exam sessions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card slide-in-right metric-container" style="animation-delay: 0.9s;">
            <h4 style="color: var(--brand-accent);">📊 Smart Reports</h4>
            <p>Comprehensive logs and analytical insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation hint with enhanced styling
    st.markdown("""
    <div class="glass-card fade-in" style="margin-top: 2rem; text-align: center; animation-delay: 1.1s;">
        <p style="color: var(--text-muted);">
            📍 <strong>Navigate using the sidebar</strong> to access Live Detection, Upload Video, Settings, and Logs & Review
        </p>
    </div>
    """, unsafe_allow_html=True)


def page_live():
    """Enhanced live webcam detection page with modern UI and animations."""
    
    st.markdown("""
    <div class="fade-in">
        <h2 class="brand-gradient-text" style="text-align: center; margin-bottom: 2rem;">
            🔴 Live Detection & Monitoring
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration section with glass card effect
    st.markdown('<div class="glass-card slide-in-left">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Session Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        session_id = st.text_input("📋 Session ID", value="live-session", help="Unique identifier for this monitoring session")
    with col2:
        student_id = st.text_input("👤 Student ID", value="student-001", help="Student identifier for tracking")
    
    calibrate_on_start = st.checkbox("🎯 Calibrate gaze on start", value=False, help="Calibrate gaze detection for better accuracy")
    
    use_webrtc = False
    if HAS_WEBRTC:
        use_webrtc = st.checkbox("🌐 Use WebRTC camera (browser)", value=False, help="Use browser camera for better performance")
    else:
        st.markdown("💡 **Tip:** Install streamlit-webrtc for enhanced browser camera support: `pip install streamlit-webrtc`")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Control buttons with enhanced styling
    st.markdown('<div class="glass-card slide-in-right" style="animation-delay: 0.3s;">', unsafe_allow_html=True)
    st.markdown("### 🎮 Session Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start = st.button("▶️ Start", help="Begin live monitoring session")
    with col2:
        pause_resume = st.button("⏸️ Pause/Resume", help="Pause or resume current session")

    with col3:
        stop = st.button("⏹️ Stop", help="Stop monitoring and release camera")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Session state initialization
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "paused" not in st.session_state:
        st.session_state.paused = False
    if "calibrate_request" not in st.session_state:
        st.session_state.calibrate_request = False
    if "snapshot_request" not in st.session_state:
        st.session_state.snapshot_request = False
    if "recent_events" not in st.session_state:
        st.session_state.recent_events = deque(maxlen=10)
    
    # Button actions
    if start:
        st.session_state.pipeline = ProcessingPipeline(session_id=session_id, student_id=student_id)
        st.session_state.running = True
        st.session_state.paused = False
        if calibrate_on_start:
            st.session_state.calibrate_request = True
        st.success("🚀 Monitoring session started successfully!")
        
    if pause_resume:
        st.session_state.paused = not st.session_state.paused
        if st.session_state.paused:
            st.warning("⏸️ Session paused")
        else:
            st.info("▶️ Session resumed")

    if stop:
        st.session_state.running = False
        st.info("⏹️ Monitoring session stopped")

    # Enhanced status display
    if st.session_state.get("running", False):
        st.markdown("""
        <div class="glass-card fade-in" style="margin-top: 1rem; animation-delay: 0.5s;">
            <h3 style="color: var(--brand-accent); text-align: center;">📊 Live Status Monitor</h3>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.get("pipeline") is not None and st.session_state.get("running", False):
        pipeline = st.session_state.pipeline
        if pipeline is None:
            st.warning("Pipeline not initialized")
            return
        placeholder = st.empty()
        info_ph = st.empty()
        events_box = st.empty()

        ctx = None
        cam = None
        if use_webrtc and HAS_WEBRTC and webrtc_streamer is not None and WebRtcMode is not None:
            session_val = session_id
            student_val = student_id

            class Processor:
                def __init__(self):
                    # Reuse or create pipeline - avoid accessing st.session_state in recv()
                    if 'pipeline' in st.session_state and st.session_state.pipeline is not None:
                        self.pipeline = st.session_state.pipeline
                    else:
                        self.pipeline = ProcessingPipeline(session_id=session_val, student_id=student_val)
                        st.session_state.pipeline = self.pipeline
                    
                    # Store state flags to avoid accessing st.session_state in recv()
                    self.paused = False
                    self.calibrate_requested = False

                def recv(self, frame):  # type: ignore
                    """Process frame in WebRTC thread - avoid Streamlit context access."""
                    # Since we are in the WebRTC path, 'av' must be available.
                    assert av is not None
                    
                    # Update local state from session state with fallback handling
                    try:
                        # Minimize session state access to reduce context warnings
                        self.paused = getattr(st.session_state, 'paused', False) if hasattr(st, 'session_state') else False
                        self.calibrate_requested = getattr(st.session_state, 'calibrate_request', False) if hasattr(st, 'session_state') else False
                    except:
                        # If context access fails completely, use safe defaults
                        self.paused = False
                        self.calibrate_requested = False
                    
                    # Convert frame to numpy array
                    img = frame.to_ndarray(format='bgr24')
                    
                    if self.paused:
                        if self.pipeline.last_annotated_frame is not None:
                            return av.VideoFrame.from_ndarray(self.pipeline.last_annotated_frame, format='bgr24')
                        else:
                            # Return original frame if no processed frame available
                            return av.VideoFrame.from_ndarray(img, format='bgr24')
                    
                    if self.calibrate_requested:
                        try:
                            self.pipeline.gaze.calibrate(img)
                            # Reset calibration flag if possible
                            if hasattr(st, 'session_state') and hasattr(st.session_state, 'calibrate_request'):
                                st.session_state.calibrate_request = False
                        except:
                            # If calibration or context access fails, continue without calibration
                            pass
                    
                    try:
                        # Process the frame
                        annotated, score, events, is_alert = self.pipeline.process_frame(img)
                        
                        # Store events in pipeline to avoid session state access
                        if events:
                            for ev in events:
                                self.pipeline.recent_events.appendleft(ev)
                        
                        return av.VideoFrame.from_ndarray(annotated, format='bgr24')
                    except Exception as e:
                        # If processing fails, return original frame to keep stream alive
                        print(f"Frame processing error: {e}")
                        return av.VideoFrame.from_ndarray(img, format='bgr24')

            ctx = webrtc_streamer(  # type: ignore
                key="arak-live",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=Processor,  # type: ignore
            )

            # Monitor status in a light loop while WebRTC is playing. We avoid heavy operations
            # here to keep the UI responsive.
            while st.session_state.get("running", False) and ctx and ctx.state.playing:
                pl = st.session_state.get("pipeline")
                if pl is not None:
                    score = pl.last_score
                    is_alert = score >= getattr(pl, 'cfg').alert_threshold
                    
                    # Enhanced status display with metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>Suspicion Score</h4>
                            <h2 style="color: {'var(--danger)' if is_alert else 'var(--success)'};">
                                {score}
                            </h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        status_text = "🚨 ALERT" if is_alert else "✅ NORMAL"
                        status_color = "var(--danger)" if is_alert else "var(--success)"
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>Status</h4>
                            <h3 style="color: {status_color};">
                                {status_text}
                            </h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        threshold = getattr(pl, 'cfg').alert_threshold
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>Alert Threshold</h4>
                            <h2 style="color: var(--text-muted);">
                                {threshold}
                            </h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Progress bar for suspicion level
                    progress = min(score / threshold, 1.0) if threshold > 0 else 0.0
                    st.progress(progress, text=f"Suspicion Level: {score}/{threshold}")
                    
                    # Events display with enhanced styling
                    if hasattr(pl, 'recent_events') and pl.recent_events:
                        st.markdown("#### 📝 Recent Events")
                        events_text = " • ".join(list(pl.recent_events)[-5:])  # Show last 5 events
                        st.markdown(f"<div class='glass-card'>{events_text}</div>", unsafe_allow_html=True)
                    else:
                        events_box.write({"recent_events": list(st.session_state.recent_events)})
                    
                    if st.session_state.get("snapshot_request", False):
                        try:
                            result = pl.snapshot_now()
                            st.session_state.snapshot_request = False
                            if result == "disabled":
                                st.toast("Manual snapshots disabled: Only automatic snapshots during suspicious moments", icon="🚫")
                            elif result == "ok":
                                st.toast("Snapshot saved to logs/snapshots/", icon="✅")
                            elif result == "not_suspicious":
                                st.toast("Snapshot rejected: Frame not suspicious enough", icon="⚠️")
                            else:
                                st.toast("Snapshot failed: No frame available", icon="❌")
                        except Exception as e:
                            st.toast(f"Snapshot failed: {e}", icon="❌")
                time.sleep(0.25)
        else:
            cam = cv2.VideoCapture(0)
            while st.session_state.get("running", False):
                ok, frame = cam.read()
                if not ok:
                    break
                if st.session_state.get("paused", False):
                    # Show last frame without processing
                    if pipeline.last_annotated_frame is not None:
                        placeholder.image(cv2.cvtColor(pipeline.last_annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                    time.sleep(0.05)
                    continue
                if st.session_state.get("calibrate_request", False):
                    pipeline.gaze.calibrate(frame)
                    st.session_state.calibrate_request = False
                annotated, score, events, is_alert = pipeline.process_frame(frame)
                placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                info_ph.markdown(
                    f"Score: {'<span class=\"status-alert\">'+str(score)+'</span>' if is_alert else '<span class=\"status-ok\">'+str(score)+'</span>'}",
                    unsafe_allow_html=True,
                )
                # Recent events list
                if events:
                    for ev in events:
                        st.session_state.recent_events.appendleft(ev)
                events_box.write({"recent_events": list(st.session_state.recent_events)})
                # Snapshot on demand (disabled)
                if st.session_state.get("snapshot_request", False):
                    result = pipeline.snapshot_now()
                    st.session_state.snapshot_request = False
                    if result == "disabled":
                        st.toast("Manual snapshots disabled: Only automatic snapshots during suspicious moments", icon="🚫")
                    elif result == "ok":
                        st.toast("Snapshot saved to logs/snapshots/", icon="✅")
                    elif result == "not_suspicious":
                        st.toast("Snapshot rejected: Frame not suspicious enough", icon="⚠️")
                    else:
                        st.toast("Snapshot failed: No frame available", icon="❌")
                # Yield to UI
                time.sleep(0.01)
            if cam is not None:
                try:
                    cam.release()
                except Exception:
                    pass


def page_upload():
    """Offline analysis for uploaded videos; writes an annotated MP4 and logs events."""
    st.header("Upload Video")
    session_id = st.text_input("Session ID", value="upload-session")
    student_id = st.text_input("Student ID", value="student-001")
    up = st.file_uploader("Select a video file", type=["mp4", "mov", "avi", "mkv"]) 
    run = st.button("Run Analysis")
    if up and run:
        # Save temp file
        tmp_path = os.path.join("data", "samples", f"tmp_{int(time.time())}.mp4")
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(up.read())

        pipeline = ProcessingPipeline(session_id=session_id, student_id=student_id)
        cap = cv2.VideoCapture(tmp_path)
        placeholder = st.empty()
        prog = st.progress(0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    # Prepare annotated video writer
        out_dir = os.path.join("logs", "videos", session_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"annotated_{int(time.time())}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, score, events, is_alert = pipeline.process_frame(frame)
            placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
            try:
                writer.write(annotated)
            except Exception:
                pass
            i += 1
            prog.progress(min(1.0, i / total))
        cap.release()
        try:
            writer.release()
        except Exception:
            pass
        st.success("Completed analysis. Check Logs & Review page for results.")
        # Offer annotated video download
        if os.path.exists(out_path):
            with open(out_path, "rb") as vf:
                st.download_button("Download annotated video", data=vf.read(), file_name=os.path.basename(out_path), mime="video/mp4")
        # Quick summary from CSV
        csv_path = os.path.join("logs", f"events_{session_id}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.write("Summary counts (event_type):")
            st.write(df["event_type"].value_counts())


def page_settings():
    """Interactive editor for config.yaml.

    Teammates can experiment with thresholds and weights and immediately apply
    them to the running app.
    """
    st.header("Settings")
    cfg = load_cfg()
    st.subheader("Thresholds")
    cfg["alert_threshold"] = st.slider("Alert threshold", 1, 15, int(cfg.get("alert_threshold", 5)))
    cfg["phone_conf"] = st.slider("Phone confidence threshold", 0.1, 0.9, float(cfg.get("phone_conf", 0.45)))

    st.subheader("Weights")
    for k in ["phone", "earphone", "smartwatch", "person", "book", "calculator", "notebook", "monitor", "gaze_off_per_sec", "repetitive_head"]:
        cfg.setdefault("weights", {})
        cfg["weights"][k] = st.slider(f"Weight: {k}", 0, 10, int(cfg["weights"].get(k, 3)))

    st.subheader("Gaze & Behavior")
    cfg["gaze_duration_threshold"] = st.slider(
        "Gaze off duration threshold (s)", 0.5, 6.0, float(cfg.get("gaze_duration_threshold", 2.5))
    )
    cfg["repeat_dir_threshold"] = st.slider("Repetitive head N", 1, 5, int(cfg.get("repeat_dir_threshold", 2)))
    cfg["repeat_window_sec"] = st.slider(
        "Repetitive head window (s)", 2.0, 30.0, float(cfg.get("repeat_window_sec", 10.0))
    )

    st.subheader("Allowed Items")
    cfg["allow_book"] = st.toggle("Allow book/paper", value=bool(cfg.get("allow_book", False)))
    cfg["allow_calculator"] = st.toggle("Allow calculator", value=bool(cfg.get("allow_calculator", False)))

    st.subheader("Detector Settings")
    cfg["detector_primary"] = st.text_input("Primary YOLO model (name or path)", value=str(cfg.get("detector_primary", "yolo11m.pt")))
    cfg["detector_secondary"] = st.text_input("Secondary YOLO weights path", value=str(cfg.get("detector_secondary", "models/model_bestV3.pt")))
    cfg["detector_conf"] = float(st.slider("Detector confidence", 0.1, 0.9, float(cfg.get("detector_conf", 0.4))))
    cfg["detector_merge_nms"] = st.toggle("Merge overlapping boxes", value=bool(cfg.get("detector_merge_nms", True)))
    cfg["detector_merge_mode"] = st.selectbox("Merge mode", options=["wbf", "nms"], index=(0 if str(cfg.get("detector_merge_mode", "wbf")).lower()=="wbf" else 1))
    cfg["detector_nms_iou"] = float(st.slider("Merge IoU threshold", 0.1, 0.9, float(cfg.get("detector_nms_iou", 0.5))))

    with st.expander("Per-class confidence thresholds"):
        # Provide common classes; keep existing values if present
        defaults = {
            "phone": 0.5,
            "earphone": 0.5,
            "smartwatch": 0.5,
            "person": 0.3,
            "book": 0.4,
            "calculator": 0.5,
            "notebook": 0.4,
        }
        class_conf = cfg.get("class_conf", {}) or {}
        for k, dv in defaults.items():
            class_conf[k] = float(st.slider(f"min conf: {k}", 0.0, 0.95, float(class_conf.get(k, dv))))
        # Allow custom key/value additions via text
        st.caption("Add/override custom class:conf (comma-separated pairs, e.g., 'tv:0.4, monitor:0.5')")
        free = st.text_input("Custom pairs", value="")
        if free.strip():
            try:
                parts = [p.strip() for p in free.split(",") if p.strip()]
                for p in parts:
                    if ":" in p:
                        name, val = p.split(":", 1)
                        name = name.strip()
                        valf = float(val.strip())
                        if name:
                            class_conf[name] = valf
            except Exception:
                st.warning("Could not parse custom pairs; please use 'name:0.5' format")
        cfg["class_conf"] = class_conf

    if st.button("Save Settings"):
        save_cfg(cfg)
        st.success("Settings saved.")


def page_logs():
    """Tabular viewer for per-session CSV logs with filters and snapshot preview."""
    st.header("Logs & Review")
    base_logs_dir = str(LOGS_DIR) if LOGS_DIR else "logs"
    if not os.path.exists(base_logs_dir):
        st.info("No logs yet.")
        return
    csv_files = [f for f in os.listdir(base_logs_dir) if f.endswith('.csv')]
    if not csv_files:
        st.info("No logs yet.")
        return
    choice = st.selectbox("Select session log", options=csv_files)
    path = os.path.join(base_logs_dir, str(choice))
    df = pd.read_csv(path)
    st.dataframe(df.tail(200), use_container_width=True)

    # Basic filters
    sid = st.text_input("Filter by student_id")
    etype = st.text_input("Filter by event_type contains")
    if sid:
        df = df[df["student_id"].astype(str).str.contains(sid)]
    if etype:
        df = df[df["event_type"].astype(str).str.contains(etype)]
    st.dataframe(df, use_container_width=True)

    # Export
    col1, col2 = st.columns(2)
    if col1.button("Export CSV"):
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="events_export.csv",
            mime="text/csv",
        )
    if col2.button("Export Excel"):
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        st.download_button(
            label="Download Excel",
            data=out.getvalue(),
            file_name="events_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.subheader("Snapshot preview")
    idx = st.number_input("Row index", min_value=0, max_value=max(len(df) - 1, 0), value=0)
    if len(df) > 0:
        row = df.iloc[int(idx)]
        snap = row.get("snapshot_path", "")
        if isinstance(snap, str) and snap and os.path.exists(snap):
            st.image(snap)
        else:
            st.info("No snapshot for this row.")


def page_about():
    """About page with project description, team, and logos.

    - Shows the project logo if `assets/logo.png` exists, otherwise falls back to
      rendering `assets/logo.txt` (ASCII) or a helpful message.
    - Shows the sponsor logo provided at `assets/NTI logo.png` when present.
    - Lists team members from the `TEAM_MEMBERS` list near the top of this file.
    """
    st.header("About A.R.A.K")
    st.caption("Academic Resilience & Authentication Kernel")

 
    st.markdown("""
    ### Project Overview
    A.R.A.K is a local-first, rules-based proctoring toolkit. It combines:
    - Face and gaze analysis (MediaPipe) for engagement and pose signals.
    - Object detection (YOLO) to identify disallowed items (e.g., phones, earphones).
    - A transparent scoring engine so instructors can adjust thresholds and weights.
Vision: Empowering educational institutions with a reliable, transparent, and privacy-conscious proctoring solution.
Mission: The goal is to assist proctoring with explainable signals while keeping data on-device.

                                         وَكَفَىٰ بِاللَّهِ رَقِيبًا
                                  (وقال النبيُّ ﷺ:(مَن غشَّنا فليس منا
                
    See the [GitHub repository](https://github.com/Ziadelshazly22/A.R.A.K) for more information.
    """)

    st.markdown("""
    ### Meet the team behind A.R.A.K:
    """)
    for member in TEAM_MEMBERS:
        name = member.get("name", "Member")
        github = member.get("github", "#")
        linkedin = member.get("linkedin", "#")
        st.markdown(f"  - [GitHub]({github}) | [LinkedIn]({linkedin})")
        st.markdown("---")

    
    st.subheader("Contact Us")
    st.markdown("""
    For inquiries, feedback, or support, please reach out to us:
    - 📧 Email: [contact@arak.com](mailto:ziad.m.elshazly@gmail.com)
    - 🌐 Website: [www.arak.com](https://www.arak.com)
    """)
#left bottom corner sponsor on about page 
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Sponsor")
        st.caption("National Telecommunication Institute | المعهد القومي للإتصالات")
        sponsor_logo = os.path.join(ASSETS_DIR, "NTI logo.png")
        if os.path.exists(sponsor_logo):
            st.image(sponsor_logo, caption="National Telecommunication Institute (NTI)", width=150)
        else:
            st.info("National Telecommunication Institute (NTI)")


def main():
    """Enhanced Streamlit entry point with improved branding and navigation."""
    # Enhanced page configuration
    st.set_page_config(
        page_title="A.R.A.K - Academic Proctoring System",
        page_icon=os.path.join(ASSETS_DIR, "A_R_A_K_ICON.jpg"),
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/Ziadelshazly22/A.R.A.K',
            'Report a bug': 'https://github.com/Ziadelshazly22/A.R.A.K/issues',
            'About': "A.R.A.K - Academic Resilience & Authentication Kernel"
        }
    )
    
    load_styles()
    
    # Enhanced sidebar with branding
    with st.sidebar:
        # Logo in sidebar
        logo_path = os.path.join(ASSETS_DIR, "A_R_A_K_Logo.svg")
        if os.path.exists(logo_path):
            st.image(logo_path, width=150)
        
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <h3 class="brand-gradient-text">A.R.A.K</h3>
            <p style="font-size: 0.8rem; color: var(--text-muted);">Academic Proctoring System</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced navigation
        st.markdown("### 📍 Navigation")
        page = st.radio(
            "",
            ["🏠 Home", "🔴 Live Detection", "📹 Upload Video", "⚙️ Settings", "📊 Logs & Review", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        # Status indicator
        st.markdown("---")
        st.markdown("### 📡 System Status")
        if st.session_state.get("running", False):
            st.markdown("🟢 **ACTIVE** - Monitoring in progress")
        else:
            st.markdown("🔴 **STANDBY** - Ready to monitor")
    
    # Route to appropriate page
    page_name = page.split(" ", 1)[1] if " " in page else page
    
    if "Home" in page:
        page_home()
    elif "Live Detection" in page:
        page_live()
    elif "Upload Video" in page:
        page_upload()
    elif "Settings" in page:
        page_settings()
    elif "Logs" in page:
        page_logs()
    else:
        page_about()
    
    # Footer with branding
    st.markdown("""
    <div class="footer">
        <p>© 2025 A.R.A.K - Academic Resilience & Authentication Kernel</p>
        <p style="font-size: 0.8rem;">Built with ❤️ for Academic Integrity</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
