# A.R.A.K — Academic Resilience & Authentication Kernel

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![YOLO](https://img.shields.io/badge/YOLO-v8/v11-darkgreen.svg)](https://ultralytics.com)

**A.R.A.K** is a cutting-edge academic proctoring system that combines advanced computer vision with intelligent behavioral analysis to ensure exam integrity. Featuring a modern glass morphism UI, enhanced branding, and intelligent monitoring, A.R.A.K delivers professional-grade proctoring with seamless user experience.

## ✨ Enhanced Features

### 🎨 Modern User Interface
- **Glass Morphism Design**: Professional frosted glass effects and transparency
- **Enhanced Branding**: Custom A.R.A.K logo integration with Arabic text support
- **Smooth Animations**: Fluid transitions and interactive hover effects
- **Responsive Layout**: Optimized for all screen sizes and devices
- **Professional Styling**: Corporate-grade color schemes and typography

### 🔒 Advanced Security
- **Automatic Snapshots Only**: Intelligent evidence capture during suspicious moments
- **Manual Snapshot Disabled**: Enhanced security by preventing manual intervention
- **Real-time Monitoring**: Continuous behavioral analysis and threat detection
- **Intelligent Scoring**: Context-aware suspicion assessment algorithms

### 🚀 Performance Optimizations
- **Dual YOLO Architecture**: Primary (yolo11m.pt) and fallback models for accuracy
- **MediaPipe Integration**: Advanced gaze tracking and facial analysis
- **Optimized Processing**: Smooth frame processing with minimal latency
- **Resource Management**: Efficient memory and CPU utilization

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Webcam or video files for testing
- Windows, macOS, or Linux

### Installation

#### Using Quick Setup Scripts:
- **Windows PowerShell**: Run `scripts/QuickSetup.ps1`
- **Windows Batch**: Run `scripts/QuickSetup.bat`
- **Linux/macOS**: Run `bash scripts/setup_venv.sh`

#### Manual Setup:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Windows Command Prompt:
venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

#### Launch Enhanced Web Interface:
```bash
# Quick start (Windows)
scripts/QuickStart.bat
# or manually:
streamlit run src/ui/streamlit_app.py
```

#### Direct Pipeline Execution:
```bash
python src/pipeline.py --video data/samples/demo.mp4 --session test --student s001
```

## 🎯 Core Capabilities

### Advanced Monitoring
- **Dual YOLO Detection**: Primary and fallback models for comprehensive object recognition
- **Real-time Gaze Tracking**: MediaPipe-powered eye movement and attention analysis
- **Behavioral Pattern Recognition**: Intelligent detection of suspicious activities
- **Automatic Evidence Capture**: Smart snapshot system during violation moments

### Enhanced Violation Detection
- **High-Severity Objects**: Mobile phones, earphones, unauthorized persons
- **Context-Dependent Objects**: Books, calculators (configurable based on exam rules)
- **Behavioral Anomalies**: Sustained gaze deviation, repetitive head movements
- **Environmental Threats**: Multiple faces, suspicious hand movements

### Professional Interface Features
- **Real-time Dashboard**: Live monitoring with animated statistics
- **Interactive Controls**: Smooth configuration adjustments
- **Visual Feedback**: Immediate response to user interactions
- **Professional Navigation**: Intuitive sidebar and main content areas

## 📁 Enhanced Project Structure

```
A.R.A.K/
├── src/
│   ├── pipeline.py              # Enhanced processing engine
│   ├── logger.py                # Advanced event logging
│   ├── detectors/               # Computer vision modules
│   │   ├── dual_yolo_detector.py    # Dual YOLO implementation
│   │   └── gaze_detector.py         # Advanced gaze tracking
│   ├── logic/                   # Intelligent scoring algorithms
│   │   ├── config.yaml              # Configuration settings
│   │   └── suspicion_scoring.py     # Enhanced scoring logic
│   └── ui/                      # Modern Streamlit interface
│       ├── streamlit_app.py         # Enhanced web application
│       └── assets/                  # UI resources and styling
├── models/                      # YOLO and custom models
│   ├── yolo11m.pt                  # Primary detection model
│   ├── yolo11n.pt                  # Fallback lightweight model
│   └── model_bestV3.pt             # Custom tuned model
├── data/samples/                # Test video repository
├── logs/                        # Enhanced logging system
│   ├── snapshots/                  # Automatic evidence captures
│   └── videos/                     # Session recordings
├── scripts/                     # Setup and utility scripts
└── requirements.txt             # Enhanced dependencies
```

## ⚙️ Enhanced Configuration

### Professional Settings
Edit `src/logic/config.yaml` to customize:
- Advanced detection thresholds
- Intelligent alert parameters
- Enhanced logging preferences
- Model selection and optimization

### Model Configuration
- **Primary Model**: `yolo11m.pt` (High accuracy, professional grade)
- **Fallback Model**: `yolo11n.pt` (Fast processing, real-time)
- **Custom Models**: Place specialized models in `models/` directory

### UI Customization
- Glass morphism effects and transparency levels
- Brand colors and corporate styling
- Animation timing and smooth transitions
- Responsive breakpoints and layouts

## 📊 Enhanced Output & Analytics

### Intelligent Event Logs
- **Location**: `logs/events_<session_id>.csv`
- **Enhanced Data**: Timestamps, violation types, confidence scores, behavioral metadata
- **Analytics**: Trend analysis and pattern recognition

### Automatic Evidence System
- **Location**: `logs/snapshots/<session_id>/`
- **Smart Triggers**: AI-powered suspicion threshold detection
- **High Quality**: Timestamped evidence with metadata
- **Security**: Manual snapshots completely disabled for integrity

### Professional Reporting
- **Session Analytics**: Comprehensive violation summaries
- **Visual Dashboard**: Real-time monitoring interface
- **Export Capabilities**: Professional report generation

## 🎨 UI Enhancement Details

### Glass Morphism Design
- **Frosted Glass Effects**: Professional transparency with backdrop blur
- **Smooth Gradients**: Corporate-grade color transitions
- **Enhanced Shadows**: Professional depth and layering

### Branding Integration
- **A.R.A.K Logo**: Custom branded interface elements
- **Arabic Typography**: Authentic cultural integration
- **Professional Colors**: Corporate blue and gradient schemes

### Interactive Elements
- **Hover Animations**: Smooth user feedback on interactions
- **Transition Effects**: Fluid movement between interface states
- **Responsive Design**: Optimal experience across all devices

## 🚦 Professional Usage Guidelines

### Academic Institutions
- **Enhanced Security**: Automatic-only snapshot system for integrity
- **Professional Interface**: Modern UI suitable for institutional use
- **Compliance Ready**: Privacy and data protection considerations
- **Training Resources**: Comprehensive documentation and guides

### Technical Excellence
- **Performance Monitoring**: Real-time system metrics
- **Quality Assurance**: Enhanced false positive reduction
- **Scalability**: Optimized for multiple concurrent sessions
- **Professional Support**: Comprehensive troubleshooting guides

## 📋 Enhanced System Requirements

### Minimum Specifications
- **CPU**: Dual-core 2.5GHz processor
- **RAM**: 4GB system memory
- **Storage**: 2GB free disk space
- **Camera**: 720p webcam with clear image quality

### Recommended Professional Setup
- **CPU**: Quad-core 3.0GHz+ processor
- **RAM**: 8GB+ system memory
- **Storage**: 10GB+ free disk space
- **Camera**: 1080p HD webcam
- **GPU**: Optional for enhanced AI processing

### Enhanced Performance
- **Network**: Stable internet for updates
- **Display**: Full HD monitor for optimal interface experience
- **Audio**: Optional for enhanced monitoring capabilities

## 🔒 Enhanced Security & Privacy

### Professional Security Features
- **Local Processing**: All data remains on institutional servers
- **Automatic Evidence**: Intelligent capture without manual intervention
- **Encrypted Storage**: Optional encryption for sensitive data
- **Access Controls**: Role-based permissions and audit trails

### Privacy Compliance
- **GDPR Ready**: European data protection regulation compliance
- **Configurable Retention**: Flexible data lifecycle management
- **Consent Management**: Student privacy protection protocols
- **Audit Capabilities**: Comprehensive activity logging

## 🆘 Enhanced Troubleshooting

### Common Solutions
- **Camera Issues**: Enhanced permission detection and driver guidance
- **Performance Optimization**: Intelligent resource management suggestions
- **UI Problems**: Modern browser compatibility and optimization
- **Model Loading**: Automatic fallback and recovery systems

### Professional Support
- **Comprehensive Logs**: Enhanced error tracking and diagnostics
- **Configuration Validation**: Automatic settings verification
- **Performance Analytics**: Real-time system health monitoring
- **Expert Documentation**: Professional-grade troubleshooting guides

## 🤝 Contributing to Excellence

### Development Standards
1. **Code Quality**: Professional coding standards and documentation
2. **UI Guidelines**: Modern design principles and accessibility
3. **Testing Protocol**: Comprehensive validation and quality assurance
4. **Security Review**: Enhanced security and privacy protection

### Enhancement Areas
- **AI Models**: Advanced detection algorithm improvements
- **UI/UX**: Continuous interface enhancement and user experience
- **Performance**: Optimization and scalability improvements
- **Integration**: Third-party system compatibility

## 🏆 Professional Acknowledgments

### Advanced Technologies
- **YOLO v11**: State-of-the-art object detection framework
- **MediaPipe**: Google's advanced computer vision platform
- **Streamlit**: Modern web application framework
- **OpenCV**: Professional computer vision library

### Design Excellence
- **Glass Morphism**: Modern UI design principles
- **Professional Branding**: Corporate identity integration
- **Responsive Design**: Cross-platform compatibility
- **Accessibility**: Inclusive design standards

---

**A.R.A.K** - *Academic Resilience & Authentication Kernel*
**Ensuring Academic Integrity Through Advanced Technology & Modern Design**

