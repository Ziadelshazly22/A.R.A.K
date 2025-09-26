from __future__ import annotations
import io
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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
    {"name": "Ziad Mahmoud ElShazly", "github": "https://github.com/Ziadelshazly22", "linkedin": "https://eg.linkedin.com/in/ziad-elshazly"},
    {"name": "Saaid Ayad", "github": "https://github.com/SaidAyad73", "linkedin": "https://www.linkedin.com/in/saaid-ayad-399321255/"},
    {"name": "Asmaa Mohammed Abdelgaber", "github": "https://github.com/Asmaagaber89782", "linkedin": "https://www.linkedin.com/in/asmaa-gaber-412b1224a/"},
    {"name": "Alaa Haitham Mohamed", "github": "https://github.com/Alaa-Haithem", "linkedin": "https://www.linkedin.com/in/alaa-haitham-493005344?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app"},
    {"name": "Osama Ali Mohamed Ali", "github": "https://github.com/osamagaa", "linkedin": "https://github.com/osamagaa"}
]


def load_styles():
    """Inject enhanced CSS styles into the Streamlit app."""
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Add custom page config for better branding
    st.set_page_config(
        page_title="A.R.A.K - Academic Proctoring System",
        page_icon=os.path.join(ASSETS_DIR,"A_R_A_K_ICON.png"),
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
    # """Enhanced landing page with A.R.A.K branding and smooth animations."""
    # # Logo section with enhanced styling
    # logo_path = os.path.join(ASSETS_DIR,"A_R_A_K_ICON.png")
    # if os.path.exists(logo_path):
    #     col1, col2, col3 = st.columns([1, 2, 1])
    #     with col2:
    #         st.markdown('<div class="logo-container fade-in" style="display: flex; justify-content: center; align-items: center;">', unsafe_allow_html=True)
    #         # st.image(logo_path, use_column_width=True)
    #         st.image(logo_path, width=100)
    #         st.markdown('</div>', unsafe_allow_html=True)
    
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
        # Save uploaded file with optimized processing indicator
        tmp_path = os.path.join("data", "samples", f"tmp_{int(time.time())}.mp4")
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(up.read())

        # Initialize pipeline with video upload optimization
        st.info("🚀 **Performance Optimization Active**: Processing every 3rd frame for faster analysis while maintaining accuracy for suspicious moment detection.")
        pipeline = ProcessingPipeline(session_id=session_id, student_id=student_id, is_video_upload=True)
        cap = cv2.VideoCapture(tmp_path)
        placeholder = st.empty()
        prog = st.progress(0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        
        # Performance stats tracking
        start_time = time.time()
        suspicious_moments = 0
        
    # Prepare annotated video writer
        out_dir = os.path.join("logs", "videos", session_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"annotated_{int(time.time())}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        i = 0
        
        # Create status container for real-time updates
        status_container = st.container()
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, score, events, is_alert = pipeline.process_frame(frame)
            
            if is_alert:
                suspicious_moments += 1
            
            # Update display less frequently for better performance
            if i % 10 == 0 or is_alert:  # Update every 10th frame or when suspicious
                placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                
            # Update status information
            if i % 30 == 0:  # Update stats every 30 frames
                elapsed = time.time() - start_time
                fps_actual = i / elapsed if elapsed > 0 else 0
                with status_container:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Frames Processed", i)
                    with col2:
                        st.metric("Processing FPS", f"{fps_actual:.1f}")
                    with col3:
                        st.metric("Suspicious Moments", suspicious_moments)
                    with col4:
                        st.metric("Current Score", score)
            
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
    """Simplified settings page - only essential user-configurable options."""
    
    st.markdown("""
    <div class="fade-in">
        <h2 class="brand-gradient-text" style="text-align: center; margin-bottom: 2rem;">
            ⚙️ Exam Configuration
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card slide-in-left">', unsafe_allow_html=True)
    st.markdown("### 📋 Exam Policy Settings")
    st.markdown("Configure what items are allowed during the exam:")
    
    # Import the config manager
    try:
        from src.logic.config_manager import get_user_settings, update_user_settings
    except ImportError:
        st.error("Configuration manager not available. Please check installation.")
        return
    
    # Get current settings
    current_settings = get_user_settings()
    
    # Create a clean, simple interface
    col1, col2 = st.columns(2)
    
    with col1:
        allow_book = st.toggle(
            "📖 Books/Textbooks", 
            value=current_settings.get('allow_book', False),
            help="Allow students to use books or printed materials"
        )
        allow_calculator = st.toggle(
            "🧮 Calculators", 
            value=current_settings.get('allow_calculator', False),
            help="Allow calculators or computation devices"
        )
    
    with col2:
        allow_notebook = st.toggle(
            "💻 Laptops/Notebooks", 
            value=current_settings.get('allow_notebook', False),
            help="Allow laptop computers or notebooks"
        )
        allow_earphones = st.toggle(
            "🎧 Earphones/Headphones", 
            value=current_settings.get('allow_earphones', False),
            help="Allow earphones or headphones"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Save settings section
    st.markdown('<div class="glass-card slide-in-right" style="animation-delay: 0.3s;">', unsafe_allow_html=True)
    st.markdown("### 💾 Save Configuration")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💾 Save Settings", type="primary"):
            new_settings = {
                'allow_book': allow_book,
                'allow_notebook': allow_notebook,
                'allow_calculator': allow_calculator,
                'allow_earphones': allow_earphones
            }
            
            if update_user_settings(new_settings):
                st.success("✅ Settings saved successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to save settings")
    
    with col2:
        if st.button("🔄 Reset to Default"):
            default_settings = {
                'allow_book': False,
                'allow_notebook': False,
                'allow_calculator': False,
                'allow_earphones': False
            }
            
            if update_user_settings(default_settings):
                st.success("✅ Settings reset to default!")
                st.rerun()
            else:
                st.error("❌ Failed to reset settings")
    
    with col3:
        st.info("💡 All technical parameters are pre-optimized for best detection accuracy and performance.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Current policy summary
    st.markdown('<div class="glass-card fade-in" style="animation-delay: 0.6s;">', unsafe_allow_html=True)
    st.markdown("### 📊 Current Exam Policy")
    
    policy_summary = []
    if allow_book:
        policy_summary.append("📖 Books/Textbooks: **ALLOWED**")
    else:
        policy_summary.append("📖 Books/Textbooks: **PROHIBITED**")
        
    if allow_calculator:
        policy_summary.append("🧮 Calculators: **ALLOWED**")
    else:
        policy_summary.append("🧮 Calculators: **PROHIBITED**")
        
    if allow_notebook:
        policy_summary.append("💻 Laptops/Notebooks: **ALLOWED**")
    else:
        policy_summary.append("💻 Laptops/Notebooks: **PROHIBITED**")
        
    if allow_earphones:
        policy_summary.append("🎧 Earphones/Headphones: **ALLOWED**")
    else:
        policy_summary.append("🎧 Earphones/Headphones: **PROHIBITED**")
    
    # Always prohibited items
    policy_summary.extend([
        "📱 Phones: **ALWAYS PROHIBITED**",
        "⌚ Smartwatches: **ALWAYS PROHIBITED**",
        "👥 Unauthorized persons: **ALWAYS PROHIBITED**"
    ])
    
    for item in policy_summary:
        st.markdown(f"- {item}")
    
    st.markdown('</div>', unsafe_allow_html=True)


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
    st.dataframe(df.tail(200), width="stretch")

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
    A.R.A.K (Academic Resilience & Authentication Kernel) — also known in Arabic as أَرَاكَ — is an AI-powered proctoring system designed to ensure academic integrity in online examinations. Our solution combines real-time face recognition, behavior analysis, and gaze tracking to detect cheating attempts with high accuracy while maintaining a seamless exam experience.

    We address one of the most critical challenges in remote education: trust and authenticity. Traditional online exams are vulnerable to unauthorized aids and undetected cheating, which undermines fairness and credibility. A.R.A.K provides an advanced yet accessible solution that empowers universities, training institutions, and exam providers to monitor exams effectively.

    ### Our Mission
    To safeguard academic integrity and build trust in digital education by leveraging AI-driven monitoring tools that are reliable, transparent, and user-friendly.

    ### Key Features
    - **Real-time Detection**: Identifies suspicious objects (phones, headphones, extra persons) and behaviors.
    - **Customizable Rules**: Optional controls for items like calculators or papers, adapting to each exam policy.
    - **Behavioral Monitoring**: Tracks gaze direction, screen focus, and unusual activities.
    - **Secure Logging**: Generates downloadable reports with timestamps, snapshots, and student IDs for human review.
    - **Seamless Experience**: Intuitive UI with video upload testing and live webcam monitoring options.

    ### Our Vision

    - We aspire to become a trusted partner in the future of secure digital education, offering institutions the confidence to conduct exams anywhere in the world without compromising integrity.

                                         وَكَفَىٰ بِاللَّهِ رَقِيبًا
                                  (وقال النبيُّ ﷺ:(مَن غشَّنا فليس منا

 
                    """)    
    st.markdown("""
    ### Technology Stack       
    See the [GitHub repository](https://github.com/Ziadelshazly22/A.R.A.K) for more information.
    """)

    st.markdown("""
    ### Meet the team behind A.R.A.K:
    """)
    for member in TEAM_MEMBERS:
        name = member.get("name", "Member")
        github = member.get("github", "#")
        linkedin = member.get("linkedin", "#")
        st.markdown(f"  - **{name}**: [GitHub]({github}) | [LinkedIn]({linkedin})")
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
        logo_path = os.path.join(ASSETS_DIR, "A_R_A_K_Logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=150 ,use_container_width=True)
        
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <h3 class="brand-gradient-text">A.R.A.K — أَرَاك</h3>
            <h3 class="brand-gradient-text">Academic Resilience & Authentication Kernel</h3
            <p style="font-size: 0.8rem; color: var(--text-muted);">Academic Proctoring System</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced navigation
        st.markdown("### 📍 Navigation")
        page = st.radio(
            "Choose a page:",
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
