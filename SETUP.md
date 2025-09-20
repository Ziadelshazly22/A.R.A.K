# A.R.A.K Enhanced Setup Guide

## 🚀 Professional Installation & Configuration

### System Prerequisites

#### Minimum Professional Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python Version**: Python 3.9 or higher (Python 3.11 recommended for optimal performance)
- **Memory**: 4GB RAM minimum (8GB recommended for enhanced performance)
- **Storage**: 2GB free disk space (10GB recommended for session data)
- **Camera**: HD webcam with stable mounting and clear image quality

#### Recommended Professional Setup
- **Processor**: Quad-core 3.0GHz+ CPU for optimal AI processing
- **Memory**: 8GB+ RAM for smooth multi-session operation
- **Graphics**: Optional GPU acceleration for enhanced YOLO processing
- **Network**: Stable internet connection for initial model downloads
- **Display**: Full HD monitor for optimal interface experience

### 🎯 Enhanced Installation Methods

#### Method 1: Professional Quick Setup (Recommended)

##### Windows PowerShell (Enhanced)
```powershell
# Professional one-command setup with enhanced features
.\scripts\QuickSetup.ps1

# Alternative with execution policy bypass
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\QuickSetup.ps1
```

##### Windows Command Prompt
```batch
# One-click professional deployment
scripts\QuickSetup.bat

# Double-click option available in Windows Explorer
```

##### Linux/macOS/WSL
```bash
# Cross-platform professional setup
bash scripts/setup_venv.sh

# With enhanced permissions if needed
chmod +x scripts/setup_venv.sh && bash scripts/setup_venv.sh
```

#### Method 2: Manual Professional Installation

##### Step 1: Enhanced Virtual Environment Creation
```powershell
# Windows PowerShell - Professional virtual environment
python -m venv venv --upgrade-deps
venv\Scripts\Activate.ps1

# Verify enhanced Python version
python --version  # Should show Python 3.9+
```

```bash
# Linux/macOS - Professional virtual environment  
python3 -m venv venv --upgrade-deps
source venv/bin/activate

# Verify enhanced Python version
python3 --version  # Should show Python 3.9+
```

##### Step 2: Enhanced Dependency Installation
```bash
# Professional dependency installation with enhanced features
pip install --upgrade pip setuptools wheel

# Install enhanced requirements with optimization
pip install -r requirements.txt

# Verify critical packages
pip list | grep -E "(streamlit|ultralytics|mediapipe|opencv)"
```

##### Step 3: Enhanced Model Preparation
```bash
# Professional model setup
mkdir -p models

# Download enhanced YOLO models (if not present)
# Primary model: yolo11m.pt (high accuracy)
# Fallback model: yolo11n.pt (fast processing)
# Custom model: model_bestV3.pt (institution-specific)
```

### 🎨 Enhanced Interface Launch

#### Professional Web Interface (Primary Method)
```bash
# Launch enhanced glass morphism interface
streamlit run src/ui/streamlit_app.py

# With custom port and enhanced options
streamlit run src/ui/streamlit_app.py --server.port 8501 --server.headless true
```

#### Quick Launch Options (Windows)
```batch
# Enhanced one-click startup
scripts\QuickStart.bat

# Alternative PowerShell quick start
.\scripts\QuickStart.ps1
```

#### Advanced Direct Pipeline (Testing)
```bash
# Enhanced pipeline testing with webcam
python src/pipeline.py --webcam --session test_session --student test_001

# Enhanced video file processing
python src/pipeline.py --video data/samples/demo.mp4 --session demo --student s001
```

### 🔧 Enhanced Configuration Setup

#### Professional Settings Configuration

##### Enhanced Config File Location
```bash
# Primary configuration file
src/logic/config.yaml

# Backup and versioning recommended
cp src/logic/config.yaml src/logic/config.yaml.backup
```

##### Professional Configuration Options
```yaml
# Enhanced A.R.A.K Configuration
alert_threshold: 5.0          # Professional suspicion threshold
detection:
  phone_confidence: 0.45      # Enhanced phone detection sensitivity
  person_confidence: 0.50     # Unauthorized person detection
  gaze_threshold: 2.5         # Professional gaze monitoring (seconds)
  
ui_settings:
  glass_morphism: true        # Enable glass morphism effects
  animations: true            # Enable smooth animations
  branding: "institutional"   # Professional branding mode
  
security:
  manual_snapshots: false     # Enhanced security - automatic only
  audit_logging: true         # Professional audit trail
  encryption: optional       # Enhanced data protection
```

#### Enhanced Model Configuration
```bash
# Professional model placement
models/
├── yolo11m.pt              # Primary high-accuracy model
├── yolo11n.pt              # Fallback fast model  
└── model_bestV3.pt         # Custom institutional model
```

### 📊 Enhanced Verification & Testing

#### Professional System Verification

##### Interface Accessibility Test
```bash
# Enhanced interface verification
curl -s http://localhost:8501/healthz

# Professional browser compatibility test
# Navigate to: http://localhost:8501
# Verify: Glass morphism effects, animations, branding
```

##### Enhanced Detection System Test
```bash
# Professional webcam detection test
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(f'Enhanced camera test: {\"Success\" if ret else \"Failed\"}')
print(f'Frame resolution: {frame.shape if ret else \"N/A\"}')
cap.release()
"
```

##### Professional Model Loading Test
```bash
# Enhanced YOLO model verification
python -c "
from ultralytics import YOLO
try:
    model = YOLO('yolo11n.pt')  
    print('Enhanced YOLO model: Success')
except Exception as e:
    print(f'Enhanced YOLO model: Failed - {e}')
"
```

#### Enhanced Sample Testing

##### Professional Video Testing
```bash
# Place test video in enhanced sample directory
mkdir -p data/samples
# Copy your test video as: data/samples/test_exam.mp4

# Enhanced pipeline testing
python src/pipeline.py --video data/samples/test_exam.mp4 --session test --student test_001
```

##### Enhanced Live Detection Testing
1. **Launch Enhanced Interface**: Navigate to `http://localhost:8501`
2. **Professional Navigation**: Use glass morphism sidebar → "Live Detection"
3. **Enhanced Monitoring**: Activate webcam with professional controls
4. **Verification Testing**: Test object detection with sample items
5. **Professional Analytics**: Review real-time statistics and alerts

### 🔒 Enhanced Security Setup

#### Professional Privacy Configuration

##### Enhanced Data Protection
```bash
# Professional secure directories
mkdir -p logs/{snapshots,videos,events}
chmod 750 logs/  # Enhanced directory permissions

# Professional retention policy setup
# Configure automatic cleanup in config.yaml
retention_days: 30  # Institutional policy compliance
```

##### Enhanced Audit Trail Setup
```yaml
# Professional audit configuration
audit_settings:
  log_level: "INFO"           # Professional logging detail
  event_tracking: true        # Enhanced event monitoring  
  access_logging: true        # Professional access control
  performance_monitoring: true # Enhanced system metrics
```

#### Professional Compliance Setup

##### Enhanced GDPR Compliance
```bash
# Professional privacy documentation
mkdir -p docs/privacy
touch docs/privacy/data_retention_policy.md
touch docs/privacy/consent_procedures.md
touch docs/privacy/gdpr_compliance.md
```

##### Enhanced Institutional Integration
```yaml
# Professional institutional settings
institution:
  name: "Your University Name"
  department: "Academic Integrity Office"
  contact: "proctoring@university.edu"
  compliance_officer: "privacy@university.edu"
```

### 🚦 Enhanced Troubleshooting Guide

#### Professional Diagnostic Steps

##### Enhanced System Health Check
```bash
# Professional system diagnostic
python -c "
import sys, cv2, streamlit, ultralytics, mediapipe
print(f'Python: {sys.version}')
print(f'OpenCV: {cv2.__version__}')
print(f'Streamlit: {streamlit.__version__}')
print(f'Ultralytics: {ultralytics.__version__}')
print(f'MediaPipe: {mediapipe.__version__}')
print('Enhanced A.R.A.K system check: Complete')
"
```

##### Enhanced Camera Troubleshooting
```bash
# Professional camera diagnostic
python -c "
import cv2
cameras = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.read()[0]:
        cameras.append(i)
    cap.release()
print(f'Enhanced cameras detected: {cameras}')
"
```

##### Enhanced Performance Optimization
```bash
# Professional performance monitoring
python -c "
import psutil
print(f'CPU cores: {psutil.cpu_count()}')
print(f'Available RAM: {psutil.virtual_memory().available // (1024**3)} GB')
print(f'Disk space: {psutil.disk_usage(\".\").free // (1024**3)} GB')
print('Enhanced system resources: Verified')
"
```

#### Professional Support Resources

##### Enhanced Error Resolution
- **Model Loading Issues**: Verify internet connection for initial downloads
- **Camera Access Problems**: Check permissions and close competing applications
- **Performance Issues**: Adjust frame rate and resolution in settings
- **Interface Problems**: Verify modern browser compatibility (Chrome/Firefox recommended)

##### Professional Optimization Tips
- **Resource Management**: Close unnecessary applications during monitoring
- **Network Optimization**: Ensure stable connection for model downloads
- **Display Optimization**: Use Full HD monitor for optimal interface experience
- **Storage Management**: Regular cleanup of session data per retention policy

### 🏆 Enhanced Production Readiness

#### Professional Deployment Checklist

##### Pre-Deployment Verification
- [ ] **Enhanced System Requirements**: Hardware and software prerequisites met
- [ ] **Professional Configuration**: Institution-specific settings configured
- [ ] **Enhanced Security Setup**: Privacy and compliance measures implemented
- [ ] **Professional Testing**: Comprehensive system validation completed
- [ ] **Enhanced Documentation**: Staff training materials prepared
- [ ] **Professional Monitoring**: System health tracking configured

##### Post-Deployment Optimization
- [ ] **Enhanced Performance Monitoring**: Real-time system metrics tracking
- [ ] **Professional User Training**: Staff education on enhanced features
- [ ] **Enhanced Maintenance Schedule**: Regular system updates and optimization
- [ ] **Professional Support Procedures**: Help desk and troubleshooting protocols
- [ ] **Enhanced Compliance Validation**: Regular privacy and security audits

#### Professional Maintenance Guidelines

##### Enhanced System Updates
```bash
# Professional system maintenance
git pull origin main          # Enhanced codebase updates
pip install -r requirements.txt --upgrade  # Enhanced dependencies
python scripts/maintenance.py  # Professional system optimization
```

##### Professional Monitoring Setup
- **Enhanced Logging**: Comprehensive event and performance tracking
- **Professional Alerts**: Real-time system health notifications
- **Enhanced Analytics**: Regular performance and accuracy analysis
- **Professional Backup**: Regular configuration and data backup procedures

---

**A.R.A.K Enhanced Setup** - *Professional Academic Proctoring Installation & Configuration*
**Ensuring Seamless Deployment of Advanced Technology Excellence**