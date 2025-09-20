# DEPLOY: A.R.A.K — Academic Resilience & Authentication Kernel

## Enhanced Deployment Guide

**A.R.A.K** features a modern glass morphism UI, intelligent automatic snapshot system, and professional-grade monitoring capabilities. This guide covers deployment of the enhanced system with all new features.

## 🚀 Professional Local Deployment

### Quick Start with Enhanced Scripts

#### Windows PowerShell (Recommended)
```powershell
# Professional setup with enhanced features
scripts/QuickSetup.ps1
```

#### Windows Command Prompt
```batch
# One-click deployment
scripts/QuickSetup.bat
```

#### Linux/macOS/WSL
```bash
# Cross-platform setup
bash scripts/setup_venv.sh
```

### Manual Professional Setup

1. **Create Enhanced Virtual Environment**
   ```powershell
   # Windows PowerShell
   python -m venv venv
   venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   
   ```bash
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Launch Enhanced Streamlit Interface**
   ```powershell
   # Professional web interface with glass morphism UI
   streamlit run src/ui/streamlit_app.py
   ```

3. **Quick Start Option (Windows)**
   ```batch
   # One-click launch with auto-setup
   scripts/QuickStart.bat
   ```

## 🎯 Enhanced Testing & Demo

### Professional Video Testing
- **Sample Location**: Place test videos in `data/samples/` directory
- **Enhanced UI**: Use the modern "Upload Video" interface with animated feedback
- **Supported Formats**: MP4, AVI, MOV with optimal quality detection

### Enhanced Web Interface Testing
1. Access the modern glass morphism interface at `http://localhost:8501`
2. Navigate through professional pages:
   - **Home**: Enhanced branding with A.R.A.K logo and smooth animations
   - **Live Detection**: Real-time monitoring with animated statistics
   - **Upload Video**: Modern file upload with progress indicators
   - **Settings**: Professional configuration with glass morphism panels
   - **Logs & Review**: Advanced analytics dashboard

### Direct Pipeline Testing (Advanced)
```bash
# Enhanced pipeline with automatic snapshots
python src/pipeline.py --video data/samples/demo.mp4 --session demo --student s001
```

## 📊 Enhanced Output & Evidence System

### Intelligent Event Logging
- **Enhanced Location**: `logs/events_<session_id>.csv`
- **Advanced Data**: Timestamps, violation confidence, behavioral metadata, scoring analytics
- **Professional Format**: Structured CSV with enhanced filtering capabilities

### Automatic Evidence Capture System
- **Smart Location**: `logs/snapshots/<session_id>/`
- **Intelligent Triggers**: AI-powered automatic snapshots during suspicious moments only
- **High-Quality Output**: Timestamped JPG evidence with metadata
- **Enhanced Security**: Manual snapshots completely disabled for integrity
- **Professional Naming**: `timestamp_frame_number.jpg` format

### Optional Session Recording
- **Professional Storage**: `logs/videos/<session_id>/`
- **Quality Options**: Configurable recording quality and compression
- **Privacy Compliant**: Optional feature with clear consent requirements

## 🎨 Enhanced UI Features Guide

### Glass Morphism Interface
- **Professional Design**: Frosted glass effects with enhanced transparency
- **Smooth Animations**: Fluid transitions between interface states
- **Interactive Elements**: Hover effects and responsive feedback
- **Brand Integration**: A.R.A.K logo with Arabic typography support

### Advanced Navigation
- **Sidebar Menu**: Professional navigation with smooth transitions
- **Page Transitions**: Animated switching between application sections
- **Responsive Design**: Optimized for desktop, tablet, and mobile viewing
- **Accessibility**: Enhanced keyboard navigation and screen reader support

### Real-time Dashboard Features
- **Live Statistics**: Animated counters and progress indicators
- **Visual Alerts**: Professional notification system with smooth alerts
- **Interactive Controls**: Modern sliders, buttons, and input fields
- **Professional Feedback**: Immediate visual response to user actions

## ⚙️ Enhanced Configuration Guide

### Professional Alert System
Understanding the intelligent alert categories:

#### High-Severity Violations (Immediate Alerts)
- **SUS_OBJECT:phone** — Mobile device detected with high confidence
- **SUS_OBJECT:earphone** — Audio device detected during restricted exam
- **SUS_OBJECT:person** — Unauthorized individual present in exam area

#### Context-Dependent Violations (Configurable)
- **SOFT_OBJECT:book** — Reference material (allowed/restricted per exam settings)
- **SOFT_OBJECT:calculator** — Calculation device (configurable based on exam requirements)

#### Behavioral Analysis Violations
- **gaze_off_sustained** — Attention deviation beyond configured thresholds
- **repetitive_head:left/right** — Suspicious head movement patterns indicating cheating

#### Intelligent Scoring System
- **Dynamic Thresholds**: AI-powered suspicion accumulation (default: 5.0)
- **Context Awareness**: Smart scoring based on exam type and duration
- **Professional Calibration**: Institution-specific threshold optimization

### Enhanced Model Configuration

#### Primary Detection Models
- **High Accuracy**: `yolo11m.pt` — Professional-grade detection for critical exams
- **Performance Optimized**: `yolo11n.pt` — Fast processing for real-time monitoring
- **Custom Models**: `models/model_bestV3.pt` — Institution-specific trained models

#### Advanced Detection Settings
- **Confidence Thresholds**: Precision-tuned for academic environments
- **Object Classification**: Enhanced recognition of academic violation objects
- **Gaze Tracking**: MediaPipe-powered attention monitoring with cultural considerations

## 🔒 Enhanced Security & Compliance

### Professional Security Features
- **Automatic-Only Snapshots**: Complete elimination of manual evidence tampering
- **Local Processing**: All data remains within institutional infrastructure
- **Encrypted Storage**: Optional encryption for sensitive examination data
- **Audit Trails**: Comprehensive logging of all system activities

### Privacy & Compliance Excellence
- **GDPR Compliance**: European data protection regulation adherence
- **Consent Management**: Clear participant agreement protocols
- **Data Retention**: Configurable lifecycle management policies
- **Access Controls**: Role-based permissions for institutional staff

### Professional Usage Protocols
- **Institutional Consent**: Mandatory participant agreement procedures
- **Retention Policies**: Clear data lifecycle and deletion schedules
- **Staff Training**: Comprehensive proctoring system orientation
- **Regular Calibration**: Ongoing system optimization and validation

## 🚦 Enhanced Performance Optimization

### Professional System Requirements

#### Minimum Professional Deployment
- **CPU**: Dual-core 2.5GHz+ processor
- **RAM**: 4GB system memory
- **Storage**: 2GB available disk space
- **Camera**: 720p webcam with stable mounting
- **Network**: Reliable connection for system updates

#### Recommended Institutional Setup
- **CPU**: Quad-core 3.0GHz+ processor with AI acceleration
- **RAM**: 8GB+ system memory for smooth operation
- **Storage**: 10GB+ available space for session data
- **Camera**: 1080p HD webcam with professional mounting
- **GPU**: Optional NVIDIA/AMD GPU for enhanced AI processing

### Performance Enhancement Strategies
- **Model Selection**: Balance accuracy vs speed based on exam criticality
- **Resolution Optimization**: Adjust camera settings for optimal detection
- **Processing Optimization**: Frame rate adjustment for resource management
- **Network Efficiency**: Local processing to minimize bandwidth requirements

## 🆘 Enhanced Troubleshooting & Support

### Professional Diagnostic Tools
- **System Health Monitoring**: Real-time performance analytics
- **Enhanced Logging**: Comprehensive error tracking and resolution
- **Configuration Validation**: Automatic settings verification
- **Performance Metrics**: Detailed system utilization reporting

### Common Professional Solutions
- **Camera Compatibility**: Enhanced driver detection and optimization guidance
- **Performance Issues**: Intelligent resource management recommendations
- **UI Responsiveness**: Modern browser compatibility optimization
- **Model Loading**: Automatic fallback systems and recovery protocols

### Advanced Support Features
- **Professional Documentation**: Comprehensive troubleshooting guides
- **Diagnostic Reports**: Automated system health assessments
- **Configuration Backup**: Settings preservation and restoration
- **Update Management**: Seamless system enhancement deployment

## 🏆 Production Deployment Excellence

### Institutional Deployment Checklist
- [ ] **Infrastructure Assessment**: System requirements validation
- [ ] **Security Review**: Privacy and data protection compliance
- [ ] **Staff Training**: Comprehensive system operation training
- [ ] **Configuration Optimization**: Institution-specific threshold calibration
- [ ] **Testing Protocol**: Comprehensive validation with sample scenarios
- [ ] **Backup Procedures**: Data protection and recovery protocols
- [ ] **Monitoring Setup**: Real-time system health tracking
- [ ] **Maintenance Schedule**: Regular system optimization and updates

### Professional Monitoring & Maintenance
- **Real-time Analytics**: Continuous system performance tracking
- **Proactive Alerts**: Advanced notification systems for technical issues
- **Regular Updates**: Seamless enhancement deployment procedures
- **Quality Assurance**: Ongoing accuracy validation and optimization

### Scalability Considerations
- **Multi-Session Support**: Concurrent examination monitoring capabilities
- **Resource Management**: Intelligent system resource allocation
- **Load Balancing**: Optimal performance under high usage scenarios
- **Infrastructure Scaling**: Guidelines for institutional expansion

---

**A.R.A.K Enhanced Deployment** - *Professional Academic Integrity Through Advanced Technology*
**Ensuring Seamless Deployment of Modern Proctoring Excellence**
