# A.R.A.K Data Processing & Normalization Guide

This guide explains how to use the enhanced data processing features in A.R.A.K to make your CSV log files more human-readable and suitable for analysis.

## 📋 Overview

A.R.A.K generates detailed CSV log files during monitoring sessions. While these files contain comprehensive data, they use technical formats that can be difficult to interpret. The data processing module transforms this raw data into human-readable formats with:

- **Human-readable timestamps** instead of Unix timestamps
- **Descriptive event categories** instead of technical codes
- **Suspicion level classifications** (Normal, Low Risk, High Risk, etc.)
- **Head pose interpretations** (Looking Up, Looking Left, etc.)
- **Comprehensive Excel reports** with multiple analysis sheets

## 🚀 Quick Start

### Method 1: Using the Streamlit Interface (Recommended)

1. **Start the A.R.A.K application:**
   ```bash
   python -m streamlit run src/ui/streamlit_app.py
   ```

2. **Navigate to "Logs & Review" in the sidebar**

3. **Enable "Enhanced View" toggle** for processed, human-readable data

4. **Select your log file** and explore the enhanced features:
   - Session summary with key metrics
   - Filtered and color-coded data display
   - Export options for Excel reports
   - Snapshot preview with context

### Method 2: Using the Command Line

#### Process All Log Files (Windows)
```batch
# Using the batch script (easiest)
scripts\ProcessLogs.bat

# Using PowerShell
scripts\ProcessLogs.ps1

# Using Python directly
python process_logs.py
```

#### Process All Log Files (Linux/Mac)
```bash
# Using Python
python process_logs.py

# With verbose output
python process_logs.py --verbose
```

#### Process Specific File
```bash
# Process a specific CSV file
python process_logs.py --input logs/events_session1.csv

# Specify output directory
python process_logs.py --input logs/events_session1.csv --output my_reports/
```

## 📊 What Gets Processed

### Raw Data → Human-Readable Data

| Raw Data | Processed Data |
|----------|----------------|
| `timestamp: 1759916364.6160989` | `datetime_utc: 2025-10-08 14:32:44 UTC` |
| `event_subtype: CRITICAL_VIOLATION:unauthorized_person` | `event_summary: 1 violation(s)` |
| `suspicion_score: 7` | `suspicion_level: Low Risk` |
| `head_pose_pitch: -0.5718763` | `pose_primary_direction: Looking Forward` |
| `gaze: off_left` | `gaze_description: Looking Left` |
| `confidence: 0.88916015625` | `confidence_percentage: 88.9%` |

### Enhanced Fields Added

The processor adds many new columns to make analysis easier:

#### Timestamp Processing
- `timestamp_datetime_utc`: Human-readable UTC datetime
- `timestamp_date`: Date only (YYYY-MM-DD)
- `timestamp_time`: Time only (HH:MM:SS)
- `timestamp_hour`, `timestamp_minute`: For time-based analysis
- `timestamp_day_of_week`: Monday, Tuesday, etc.

#### Event Analysis
- `event_summary`: Brief description of events in the frame
- `event_violations`: List of detected violations
- `event_behavioral_issues`: List of behavioral anomalies
- `event_detected_objects`: List of unauthorized objects

#### Suspicion Scoring
- `suspicion_level`: Normal, Low Risk, Medium Risk, High Risk, Critical Risk
- `suspicion_color`: Color coding for visualization
- `suspicion_description`: Detailed explanation of risk level

#### Head Pose & Gaze
- `pose_primary_direction`: Primary head direction (Looking Up, Left, etc.)
- `pose_pitch_description`: Pitch interpretation (Looking Down, Level, Looking Up)
- `pose_yaw_description`: Yaw interpretation (Turned Left, Facing Forward, Turned Right)
- `gaze_description`: Human-readable gaze direction

#### Detection Data
- `bbox_width`, `bbox_height`, `bbox_area`: Bounding box dimensions
- `bbox_center_x`, `bbox_center_y`: Center coordinates
- `bbox_formatted`: Human-readable coordinate description
- `confidence_percentage`: Confidence as percentage
- `confidence_level`: High, Medium, Low confidence

#### Session Data
- `session_duration_frame`: Frame sequence in session
- `has_snapshot`: Boolean indicating if snapshot exists
- `snapshot_filename`: Just the filename for easier reference

## 📁 Output Structure

Processed files are saved in the `logs/processed/` directory with the following structure:

```
logs/
├── events_session1.csv                    # Original raw data
├── events_session2.csv
└── processed/                             # Processed files
    ├── events_session1_processed.xlsx     # Enhanced Excel report
    ├── events_session2_processed.xlsx
    └── summary_report.xlsx                # Combined analysis
```

### Excel Report Structure

Each processed Excel file contains multiple sheets:

1. **Processed_Data**: Main data with all enhanced columns
2. **Summary_Report**: Session overview and statistics
3. **Violations_Only**: Filtered view of suspicious events only

## 🔍 Analysis Features

### Session Summary

Each processed file includes comprehensive session analytics:

```
📋 Session Overview:
├── Total Frames: 1,234
├── Duration: 45.2 minutes  
├── Students: 1
├── Snapshots: 67
└── Time Span: 2025-10-08 14:30:00 to 15:15:12

🚨 Suspicion Analysis:
├── Total Alerts: 89
├── Max Score: 263
├── Average Score: 12.4
└── Risk Distribution:
    ├── Normal: 1,145 frames (92.8%)
    ├── Low Risk: 67 frames (5.4%)
    ├── High Risk: 18 frames (1.5%)
    └── Critical Risk: 4 frames (0.3%)

🔍 Detection Summary:
├── Unauthorized Person: 45 detections
├── Unauthorized Laptop: 23 detections
└── Sustained Gaze Off-Screen: 12 incidents
```

### Filtering and Search

The Streamlit interface provides advanced filtering:

- **Student ID**: Filter by specific student
- **Event Type**: Filter by SUS (suspicious) or NORMAL events
- **Risk Level**: Filter by suspicion level
- **Time Range**: Filter by date/time (coming soon)
- **Object Type**: Filter by detected objects (coming soon)

### Export Options

Multiple export formats are available:

1. **Filtered CSV**: Current filtered data as CSV
2. **Excel Report**: Comprehensive report with multiple sheets
3. **Process All**: Batch process all log files at once

## ⚙️ Advanced Usage

### Custom Processing Scripts

You can integrate the data processor into your own scripts:

```python
from src.data_processor import ARAKDataProcessor

# Initialize processor
processor = ARAKDataProcessor()

# Process a file
df = processor.process_csv_file('logs/events_session1.csv')

# Generate summary
summary = processor.create_summary_report(df)

# Export to Excel
processor.export_processed_data(df, 'my_report.xlsx', 'excel')
```

### Command Line Options

```bash
# Full command line options
python process_logs.py --help

Usage: process_logs.py [OPTIONS]

Options:
  -i, --input FILE     Input CSV file path
  -o, --output DIR     Output directory  
  -a, --all           Process all CSV files
  --logs-dir DIR      Logs directory (default: logs/)
  --format FORMAT     Output format: excel or csv
  -v, --verbose       Verbose output
  --help              Show help message
```

### Batch Processing

Process multiple sessions at once:

```bash
# Process all files with verbose output
python process_logs.py --all --verbose

# Process all files to custom directory
python process_logs.py --all --output reports/

# Process all files as CSV format
python process_logs.py --all --format csv
```

## 🎯 Use Cases

### Academic Administrators
- **Compliance Reporting**: Generate formatted reports for academic integrity reviews
- **Trend Analysis**: Identify patterns across multiple exam sessions
- **Student Performance**: Analyze individual student behavior over time

### Researchers
- **Behavioral Analysis**: Study patterns in academic behavior during exams
- **System Validation**: Validate A.R.A.K detection accuracy
- **Data Export**: Export data for external analysis tools

### IT Administrators
- **System Monitoring**: Track system performance and detection efficiency
- **Quality Assurance**: Verify data integrity and completeness
- **Batch Processing**: Automate report generation for large datasets

## 🔧 Troubleshooting

### Common Issues

**"Data processor module not available"**
- Ensure you're running from the A.R.A.K project directory
- Check that `src/data_processor.py` exists

**"No CSV files found"**
- Run monitoring sessions to generate log files first
- Check that log files are in the `logs/` directory

**"Processing failed"**
- Check for corrupted CSV files
- Verify Python dependencies are installed
- Run with `--verbose` for detailed error messages

**Excel files won't open**
- Ensure you have Excel or LibreOffice installed
- Check that openpyxl package is installed: `pip install openpyxl`

### Performance Tips

- **Large Files**: For files with >10,000 rows, processing may take 1-2 minutes
- **Memory Usage**: Each processed file uses ~3x the memory of the original
- **Batch Processing**: Process files individually if experiencing memory issues

## 📚 Technical Details

### Data Processing Pipeline

1. **Load Raw CSV**: Read original log file using pandas
2. **Timestamp Conversion**: Convert Unix timestamps to multiple readable formats
3. **Event Parsing**: Parse event subtypes into categorized lists
4. **Coordinate Processing**: Convert bounding box strings to numeric data
5. **Head Pose Normalization**: Interpret pose angles into directional descriptions
6. **Suspicion Categorization**: Map numeric scores to risk levels
7. **Summary Generation**: Create aggregate statistics and insights
8. **Export Formatting**: Generate Excel workbook with multiple sheets

### Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **openpyxl**: Excel file generation
- **streamlit**: Web interface (optional)

### File Format Support

- **Input**: CSV files generated by A.R.A.K
- **Output**: Excel (.xlsx) or CSV files
- **Encoding**: UTF-8 with full Unicode support

## 🔮 Future Enhancements

Planned improvements for future versions:

- **Real-time Processing**: Process data as it's generated
- **Dashboard Integration**: Live analytics dashboard
- **Custom Reports**: User-defined report templates
- **Data Visualization**: Built-in charts and graphs
- **API Integration**: REST API for external systems
- **Machine Learning**: Automated pattern detection

## 📞 Support

For issues or questions about data processing:

1. Check this documentation first
2. Run processing with `--verbose` flag for detailed output
3. Check the GitHub Issues page
4. Contact the development team

---

**Happy analyzing! 📊**