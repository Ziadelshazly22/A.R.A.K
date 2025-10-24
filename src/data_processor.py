"""
Data Processing and Normalization Module for A.R.A.K Academic Proctoring System

This module provides functions to clean, normalize, and make CSV log data more human-readable.
It handles timestamp conversion, data type normalization, and adds computed fields for better analysis.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import ast
import re


class ARAKDataProcessor:
    """
    A comprehensive data processor for A.R.A.K log files.
    Handles cleaning, normalization, and human-readable formatting of CSV data.
    """
    
    def __init__(self):
        self.violation_categories = {
            'CRITICAL_VIOLATION': 'Critical Violation',
            'POLICY_VIOLATION': 'Policy Violation', 
            'BEHAVIORAL': 'Behavioral Anomaly',
            'NORMAL': 'Normal Activity'
        }
        
        self.object_names = {
            'unauthorized_person': 'Unauthorized Person',
            'unauthorized_phone': 'Unauthorized Phone',
            'unauthorized_book': 'Unauthorized Book',
            'unauthorized_laptop': 'Unauthorized Laptop/Notebook',
            'unauthorized_calculator': 'Unauthorized Calculator',
            'unauthorized_earphone': 'Unauthorized Earphone/Headphone',
            'unauthorized_smartwatch': 'Unauthorized Smartwatch'
        }
        
        self.behavioral_patterns = {
            'sustained_gaze_off': 'Sustained Off-Screen Gaze',
            'suspicious_head_movement': 'Suspicious Head Movement',
            'repetitive_motion': 'Repetitive Motion Pattern'
        }
        
        self.gaze_directions = {
            'on_screen': 'Looking at Screen',
            'off_left': 'Looking Left',
            'off_right': 'Looking Right', 
            'up': 'Looking Up',
            'down': 'Looking Down',
            'off_screen': 'Looking Off-Screen'
        }

    def clean_timestamp(self, timestamp: float) -> Dict[str, Any]:
        """Convert Unix timestamp to human-readable datetime formats."""
        try:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return {
                'datetime_utc': dt.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'date': dt.strftime('%Y-%m-%d'),
                'time': dt.strftime('%H:%M:%S'),
                'hour': dt.hour,
                'minute': dt.minute,
                'day_of_week': dt.strftime('%A'),
                'unix_timestamp': timestamp
            }
        except (ValueError, OSError):
            return {
                'datetime_utc': 'Invalid Timestamp',
                'date': 'Unknown',
                'time': 'Unknown',
                'hour': 0,
                'minute': 0,
                'day_of_week': 'Unknown',
                'unix_timestamp': timestamp
            }

    def parse_event_subtype(self, event_subtype: str) -> Dict[str, List[str]]:
        """Parse and categorize event subtypes into human-readable categories."""
        if pd.isna(event_subtype) or event_subtype == 'none':
            return {
                'violations': [],
                'behavioral_issues': [],
                'detected_objects': [],
                'summary': 'No specific issues detected'
            }
        
        violations = []
        behavioral_issues = []
        detected_objects = []
        
        # Split by semicolon to handle multiple events
        events = [e.strip() for e in str(event_subtype).split(';')]
        
        for event in events:
            if ':' in event:
                category, detail = event.split(':', 1)
                category = category.strip()
                detail = detail.strip()
                
                if category == 'CRITICAL_VIOLATION':
                    obj_name = self.object_names.get(detail, detail.replace('_', ' ').title())
                    violations.append(f"Critical: {obj_name}")
                    detected_objects.append(obj_name)
                    
                elif category == 'POLICY_VIOLATION':
                    obj_name = self.object_names.get(detail, detail.replace('_', ' ').title())
                    violations.append(f"Policy: {obj_name}")
                    detected_objects.append(obj_name)
                    
                elif category == 'BEHAVIORAL':
                    behavior_parts = detail.split('_')
                    if len(behavior_parts) >= 3 and behavior_parts[0] == 'sustained' and behavior_parts[1] == 'gaze':
                        direction = '_'.join(behavior_parts[2:])
                        direction_name = self.gaze_directions.get(direction, direction.replace('_', ' ').title())
                        behavioral_issues.append(f"Sustained gaze: {direction_name}")
                    elif 'head_movement' in detail:
                        direction = detail.split('_')[-1] if '_' in detail else 'unknown'
                        behavioral_issues.append(f"Suspicious head movement: {direction.title()}")
                    else:
                        behavior_name = self.behavioral_patterns.get(detail, detail.replace('_', ' ').title())
                        behavioral_issues.append(behavior_name)
            else:
                # Handle events without category prefix
                violations.append(event.replace('_', ' ').title())
        
        # Create summary
        summary_parts = []
        if violations:
            summary_parts.append(f"{len(violations)} violation(s)")
        if behavioral_issues:
            summary_parts.append(f"{len(behavioral_issues)} behavioral issue(s)")
        
        summary = '; '.join(summary_parts) if summary_parts else 'Normal activity'
        
        return {
            'violations': violations,
            'behavioral_issues': behavioral_issues,
            'detected_objects': list(set(detected_objects)),
            'summary': summary
        }

    def parse_bbox(self, bbox_str: str) -> Dict[str, Any]:
        """Parse bounding box string into readable coordinates and dimensions."""
        if pd.isna(bbox_str) or bbox_str == '[0.0, 0.0, 0.0, 0.0]':
            return {
                'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0,
                'width': 0, 'height': 0, 'area': 0,
                'center_x': 0, 'center_y': 0,
                'formatted': 'No detection'
            }
        
        try:
            # Parse the bbox string - it's usually in format "[x1, y1, x2, y2]"
            bbox_str = bbox_str.strip()
            if bbox_str.startswith('[') and bbox_str.endswith(']'):
                coords = ast.literal_eval(bbox_str)
            else:
                coords = [float(x.strip()) for x in bbox_str.split(',')]
            
            x1, y1, x2, y2 = coords[:4]
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            area = width * height
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            return {
                'x1': round(x1, 1), 'y1': round(y1, 1),
                'x2': round(x2, 1), 'y2': round(y2, 1),
                'width': round(width, 1), 'height': round(height, 1),
                'area': round(area, 1),
                'center_x': round(center_x, 1), 'center_y': round(center_y, 1),
                'formatted': f"({round(x1,1)}, {round(y1,1)}) to ({round(x2,1)}, {round(y2,1)})"
            }
        except (ValueError, SyntaxError, IndexError):
            return {
                'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0,
                'width': 0, 'height': 0, 'area': 0,
                'center_x': 0, 'center_y': 0,
                'formatted': 'Invalid coordinates'
            }

    def normalize_head_pose(self, pitch: float, yaw: float, roll: float) -> Dict[str, Any]:
        """Normalize and categorize head pose angles into readable descriptions."""
        try:
            pitch = float(pitch) if not pd.isna(pitch) else 0.0
            yaw = float(yaw) if not pd.isna(yaw) else 0.0  
            roll = float(roll) if not pd.isna(roll) else 0.0
            
            # Handle extremely large values (likely corrupted data)
            if abs(pitch) > 180:
                pitch = pitch % 360
                if pitch > 180:
                    pitch -= 360
            if abs(yaw) > 180:
                yaw = yaw % 360
                if yaw > 180:
                    yaw -= 360
            if abs(roll) > 180:
                roll = roll % 360
                if roll > 180:
                    roll -= 360
            
            # Categorize head pose
            pitch_desc = self._categorize_pitch(pitch)
            yaw_desc = self._categorize_yaw(yaw)
            roll_desc = self._categorize_roll(roll)
            
            # Overall head position description
            primary_direction = self._get_primary_direction(pitch, yaw)
            
            return {
                'pitch': round(pitch, 2),
                'yaw': round(yaw, 2), 
                'roll': round(roll, 2),
                'pitch_description': pitch_desc,
                'yaw_description': yaw_desc,
                'roll_description': roll_desc,
                'primary_direction': primary_direction,
                'formatted': f"{primary_direction} (P:{round(pitch,1)}°, Y:{round(yaw,1)}°, R:{round(roll,1)}°)"
            }
        except (ValueError, TypeError):
            return {
                'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0,
                'pitch_description': 'Unknown',
                'yaw_description': 'Unknown', 
                'roll_description': 'Unknown',
                'primary_direction': 'Unknown',
                'formatted': 'Invalid pose data'
            }

    def _categorize_pitch(self, pitch: float) -> str:
        """Categorize pitch angle into descriptive terms."""
        if pitch > 15:
            return "Looking Down"
        elif pitch < -15:
            return "Looking Up" 
        else:
            return "Level"

    def _categorize_yaw(self, yaw: float) -> str:
        """Categorize yaw angle into descriptive terms."""
        if yaw > 20:
            return "Turned Right"
        elif yaw < -20:
            return "Turned Left"
        else:
            return "Facing Forward"

    def _categorize_roll(self, roll: float) -> str:
        """Categorize roll angle into descriptive terms.""" 
        if abs(roll) > 10:
            direction = "Right" if roll > 0 else "Left"
            return f"Head Tilted {direction}"
        else:
            return "Head Upright"

    def _get_primary_direction(self, pitch: float, yaw: float) -> str:
        """Determine primary head direction from pitch and yaw."""
        if abs(yaw) > abs(pitch):
            if yaw > 20:
                return "Looking Right"
            elif yaw < -20:
                return "Looking Left"
            else:
                return "Looking Forward"
        else:
            if pitch > 15:
                return "Looking Down"
            elif pitch < -15:
                return "Looking Up"
            else:
                return "Looking Forward"

    def categorize_suspicion_level(self, score: int) -> Dict[str, str]:
        """Categorize suspicion score into risk levels."""
        if score == 0:
            return {
                'level': 'Normal',
                'color': 'green',
                'description': 'No suspicious activity detected'
            }
        elif score <= 10:
            return {
                'level': 'Low Risk', 
                'color': 'yellow',
                'description': 'Minor irregularities detected'
            }
        elif score <= 30:
            return {
                'level': 'Medium Risk',
                'color': 'orange', 
                'description': 'Moderate suspicious behavior'
            }
        elif score <= 50:
            return {
                'level': 'High Risk',
                'color': 'red',
                'description': 'Significant violations detected'
            }
        else:
            return {
                'level': 'Critical Risk',
                'color': 'darkred',
                'description': 'Severe violations requiring immediate attention'
            }

    def process_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Process a single CSV file and return a normalized, human-readable DataFrame.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        if df.empty:
            return df
        
        # Create a copy for processing
        processed_df = df.copy()
        
        # Process timestamps
        if 'timestamp' in processed_df.columns:
            timestamp_data = processed_df['timestamp'].apply(self.clean_timestamp)
            timestamp_df = pd.json_normalize(timestamp_data)
            
            # Add timestamp columns
            for col in timestamp_df.columns:
                processed_df[f'timestamp_{col}'] = timestamp_df[col]
        
        # Process event subtypes
        if 'event_subtype' in processed_df.columns:
            event_data = processed_df['event_subtype'].apply(self.parse_event_subtype)
            event_df = pd.json_normalize(event_data)
            
            # Add event analysis columns
            for col in event_df.columns:
                processed_df[f'event_{col}'] = event_df[col]
        
        # Process bounding boxes
        if 'bbox' in processed_df.columns:
            bbox_data = processed_df['bbox'].apply(self.parse_bbox)
            bbox_df = pd.json_normalize(bbox_data)
            
            # Add bbox columns
            for col in bbox_df.columns:
                processed_df[f'bbox_{col}'] = bbox_df[col]
        
        # Process head pose data
        if all(col in processed_df.columns for col in ['head_pose_pitch', 'head_pose_yaw', 'head_pose_roll']):
            pose_data = processed_df.apply(
                lambda row: self.normalize_head_pose(
                    row['head_pose_pitch'], 
                    row['head_pose_yaw'], 
                    row['head_pose_roll']
                ), axis=1
            )
            pose_df = pd.json_normalize(pose_data)
            
            # Add head pose columns
            for col in pose_df.columns:
                processed_df[f'pose_{col}'] = pose_df[col]
        
        # Process gaze direction
        if 'gaze' in processed_df.columns:
            processed_df['gaze_description'] = processed_df['gaze'].map(
                lambda x: self.gaze_directions.get(x, str(x).replace('_', ' ').title())
            )
        
        # Process suspicion score
        if 'suspicion_score' in processed_df.columns:
            suspicion_data = processed_df['suspicion_score'].apply(self.categorize_suspicion_level)
            suspicion_df = pd.json_normalize(suspicion_data)
            
            # Add suspicion level columns
            for col in suspicion_df.columns:
                processed_df[f'suspicion_{col}'] = suspicion_df[col]
        
        # Process confidence score
        if 'confidence' in processed_df.columns:
            processed_df['confidence_percentage'] = (processed_df['confidence'] * 100).round(1)
            processed_df['confidence_level'] = processed_df['confidence'].apply(
                lambda x: 'High' if x > 0.8 else 'Medium' if x > 0.5 else 'Low'
            )
        
        # Add computed fields
        processed_df['session_duration_frame'] = processed_df.groupby('session_id').cumcount() + 1
        
        # Clean up snapshot paths
        if 'snapshot_path' in processed_df.columns:
            processed_df['has_snapshot'] = processed_df['snapshot_path'].notna() & (processed_df['snapshot_path'] != '')
            processed_df['snapshot_filename'] = processed_df['snapshot_path'].apply(
                lambda x: os.path.basename(x) if pd.notna(x) and x else 'No snapshot'
            )
        
        return processed_df

    def create_summary_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a comprehensive summary report from processed data.
        """
        if df.empty:
            return {'status': 'No data available'}
        
        report = {}
        
        # Basic session info
        report['session_info'] = {
            'total_frames': len(df),
            'unique_sessions': df['session_id'].nunique() if 'session_id' in df.columns else 1,
            'unique_students': df['student_id'].nunique() if 'student_id' in df.columns else 1,
            'time_span': self._get_time_span(df),
            'has_snapshots': df['has_snapshot'].sum() if 'has_snapshot' in df.columns else 0
        }
        
        # Suspicion analysis
        if 'suspicion_score' in df.columns:
            report['suspicion_analysis'] = {
                'total_alerts': (df['suspicion_score'] > 0).sum(),
                'max_score': df['suspicion_score'].max(),
                'avg_score': df['suspicion_score'].mean(),
                'score_distribution': df['suspicion_level'].value_counts().to_dict() if 'suspicion_level' in df.columns else {}
            }
        
        # Event type analysis
        if 'event_type' in df.columns:
            report['event_analysis'] = {
                'event_distribution': df['event_type'].value_counts().to_dict(),
                'violation_count': (df['event_type'] == 'SUS').sum(),
                'normal_count': (df['event_type'] == 'NORMAL').sum()
            }
        
        # Object detection analysis
        if 'event_detected_objects' in df.columns:
            detected_objects = []
            for obj_list in df['event_detected_objects'].dropna():
                if isinstance(obj_list, list):
                    detected_objects.extend(obj_list)
            
            report['detection_analysis'] = {
                'unique_objects': list(set(detected_objects)),
                'object_frequency': pd.Series(detected_objects).value_counts().to_dict() if detected_objects else {}
            }
        
        # Gaze analysis
        if 'gaze' in df.columns:
            report['gaze_analysis'] = {
                'gaze_distribution': df['gaze'].value_counts().to_dict(),
                'off_screen_percentage': ((df['gaze'] != 'on_screen').sum() / len(df) * 100).round(2)
            }
        
        return report

    def _get_time_span(self, df: pd.DataFrame) -> str:
        """Calculate the time span of the session."""
        if 'timestamp' not in df.columns or df.empty:
            return 'Unknown'
        
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        duration = end_time - start_time
        
        if duration < 60:
            return f"{duration:.1f} seconds"
        elif duration < 3600:
            return f"{duration/60:.1f} minutes"
        else:
            return f"{duration/3600:.1f} hours"

    def export_processed_data(self, df: pd.DataFrame, output_path: str, format: str = 'excel') -> str:
        """
        Export processed data to Excel or CSV with proper formatting.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format.lower() == 'excel':
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Processed_Data', index=False)
                
                # Summary sheet
                summary = self.create_summary_report(df)
                summary_df = pd.json_normalize(summary, sep='_')
                summary_df.to_excel(writer, sheet_name='Summary_Report', index=False)
                
                # Violation events only
                if 'event_type' in df.columns:
                    violations_df = df[df['event_type'] == 'SUS']
                    violations_df.to_excel(writer, sheet_name='Violations_Only', index=False)
        
        elif format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        
        return output_path


def process_log_file(input_path: str, output_dir: str = None) -> str:
    """
    Convenience function to process a single log file.
    """
    processor = ARAKDataProcessor()
    
    # Generate output path if not provided
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_path), 'processed')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the file
    processed_df = processor.process_csv_file(input_path)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_processed.xlsx")
    
    # Export processed data
    processor.export_processed_data(processed_df, output_path, 'excel')
    
    return output_path


def process_all_logs(logs_dir: str = "logs", output_dir: str = None) -> List[str]:
    """
    Process all CSV log files in the logs directory.
    """
    if output_dir is None:
        output_dir = os.path.join(logs_dir, 'processed')
    
    csv_files = [f for f in os.listdir(logs_dir) if f.endswith('.csv')]
    processed_files = []
    
    for csv_file in csv_files:
        input_path = os.path.join(logs_dir, csv_file)
        try:
            output_path = process_log_file(input_path, output_dir)
            processed_files.append(output_path)
            print(f"Processed: {csv_file} -> {os.path.basename(output_path)}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    return processed_files


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Process A.R.A.K log files for human readability")
    parser.add_argument("--input", "-i", help="Input CSV file path")
    parser.add_argument("--output", "-o", help="Output directory (optional)")
    parser.add_argument("--all", "-a", action="store_true", help="Process all CSV files in logs directory")
    
    args = parser.parse_args()
    
    if args.all:
        processed = process_all_logs(output_dir=args.output)
        print(f"Processed {len(processed)} files")
    elif args.input:
        output_path = process_log_file(args.input, args.output)
        print(f"Processed file saved to: {output_path}")
    else:
        print("Please specify --input file or --all to process all files")