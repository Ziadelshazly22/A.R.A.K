#!/usr/bin/env python3
"""
A.R.A.K Log Data Processing Tool

This script processes A.R.A.K CSV log files to make them more human-readable.
It can be run standalone or integrated into the Streamlit application.

Usage:
    python process_logs.py                    # Process all CSV files in logs/
    python process_logs.py --input file.csv  # Process specific file
    python process_logs.py --help            # Show help
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.data_processor import ARAKDataProcessor, process_log_file, process_all_logs
except ImportError:
    print("❌ Error: Could not import data processor. Make sure you're running from the A.R.A.K project directory.")
    sys.exit(1)


def main():
    """Main entry point for the log processing tool."""
    
    print("🔄 A.R.A.K Log Data Processing Tool")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(
        description="Process A.R.A.K log files for enhanced human readability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           Process all CSV files in logs/
  %(prog)s -i events_session1.csv   Process specific file
  %(prog)s -o processed_logs/       Specify output directory
  %(prog)s --all                    Process all files (explicit)
        """
    )
    
    parser.add_argument(
        "--input", "-i", 
        metavar="FILE",
        help="Input CSV file path to process"
    )
    
    parser.add_argument(
        "--output", "-o", 
        metavar="DIR",
        help="Output directory for processed files (default: logs/processed/)"
    )
    
    parser.add_argument(
        "--all", "-a", 
        action="store_true",
        help="Process all CSV files in logs directory"
    )
    
    parser.add_argument(
        "--logs-dir", 
        metavar="DIR",
        default="logs",
        help="Directory containing log files (default: logs/)"
    )
    
    parser.add_argument(
        "--format", 
        choices=["excel", "csv"],
        default="excel",
        help="Output format (default: excel)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and args.all:
        print("❌ Error: Cannot specify both --input and --all")
        return 1
    
    if not args.input and not args.all:
        # Default behavior: process all files
        args.all = True
    
    # Check if logs directory exists
    if args.all and not os.path.exists(args.logs_dir):
        print(f"❌ Error: Logs directory '{args.logs_dir}' not found.")
        print("💡 Make sure you're running from the A.R.A.K project directory and have generated logs.")
        return 1
    
    # Initialize processor
    try:
        processor = ARAKDataProcessor()
        if args.verbose:
            print("✅ Data processor initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing data processor: {e}")
        return 1
    
    try:
        if args.input:
            # Process single file
            if not os.path.exists(args.input):
                print(f"❌ Error: Input file '{args.input}' not found.")
                return 1
            
            print(f"📄 Processing file: {args.input}")
            
            output_path = process_log_file(args.input, args.output)
            
            print(f"✅ Processing complete!")
            print(f"📂 Output saved to: {output_path}")
            
            # Show summary
            if args.verbose:
                show_file_summary(processor, args.input)
                
        else:
            # Process all files
            print(f"📁 Scanning for CSV files in: {args.logs_dir}")
            
            csv_files = [f for f in os.listdir(args.logs_dir) if f.endswith('.csv')]
            
            if not csv_files:
                print(f"📄 No CSV files found in '{args.logs_dir}'")
                print("💡 Run some monitoring sessions to generate log files first.")
                return 0
            
            print(f"📊 Found {len(csv_files)} CSV file(s) to process:")
            for i, file in enumerate(csv_files, 1):
                print(f"  {i}. {file}")
            
            print("\n🔄 Processing files...")
            
            processed_files = process_all_logs(args.logs_dir, args.output)
            
            print(f"\n✅ Processing complete!")
            print(f"📂 Processed {len(processed_files)} file(s)")
            
            if processed_files:
                print("\n📋 Processed files:")
                for file_path in processed_files:
                    file_name = os.path.basename(file_path)
                    print(f"  ✓ {file_name}")
                
                print(f"\n📁 Output directory: {os.path.dirname(processed_files[0])}")
            
            # Show overall summary
            if args.verbose and processed_files:
                print("\n📊 Processing Summary:")
                total_frames = 0
                total_violations = 0
                
                for csv_file in csv_files:
                    try:
                        input_path = os.path.join(args.logs_dir, csv_file)
                        df = processor.process_csv_file(input_path)
                        summary = processor.create_summary_report(df)
                        
                        session_info = summary.get('session_info', {})
                        event_info = summary.get('event_analysis', {})
                        
                        frames = session_info.get('total_frames', 0)
                        violations = event_info.get('violation_count', 0)
                        
                        total_frames += frames
                        total_violations += violations
                        
                        print(f"  📄 {csv_file}: {frames} frames, {violations} violations")
                        
                    except Exception as e:
                        print(f"  ❌ {csv_file}: Error - {e}")
                
                print(f"\n📈 Total: {total_frames} frames processed, {total_violations} violations detected")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def show_file_summary(processor: ARAKDataProcessor, file_path: str):
    """Show detailed summary for a single file."""
    try:
        print(f"\n📊 Summary for {os.path.basename(file_path)}:")
        print("-" * 40)
        
        df = processor.process_csv_file(file_path)
        summary = processor.create_summary_report(df)
        
        # Session info
        session_info = summary.get('session_info', {})
        print(f"📋 Total frames: {session_info.get('total_frames', 0)}")
        print(f"👥 Students: {session_info.get('unique_students', 0)}")
        print(f"📸 Snapshots: {session_info.get('has_snapshots', 0)}")
        print(f"⏱️ Duration: {session_info.get('time_span', 'Unknown')}")
        
        # Event analysis
        if 'event_analysis' in summary:
            event_info = summary['event_analysis']
            print(f"🚨 Violations: {event_info.get('violation_count', 0)}")
            print(f"✅ Normal events: {event_info.get('normal_count', 0)}")
        
        # Suspicion analysis
        if 'suspicion_analysis' in summary:
            suspicion = summary['suspicion_analysis']
            print(f"⚠️ Total alerts: {suspicion.get('total_alerts', 0)}")
            print(f"📊 Max score: {suspicion.get('max_score', 0)}")
            print(f"📊 Avg score: {suspicion.get('avg_score', 0):.1f}")
        
        # Object detection
        if 'detection_analysis' in summary:
            detection = summary['detection_analysis']
            objects = detection.get('unique_objects', [])
            if objects:
                print(f"🔍 Detected objects: {', '.join(objects)}")
        
    except Exception as e:
        print(f"❌ Error generating summary: {e}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)