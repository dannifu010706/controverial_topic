#!/usr/bin/env python3
"""
CSV to JSON Converter - STORM Format Compatible

This script converts CSV files containing reference articles into the same JSON format
as the STORM article consolidation script. This allows for easy comparison between
generated articles and reference articles.

Usage:
    python csv_to_json.py --csv-file topics.csv --output-dir ./reference_articles
    python csv_to_json.py --csv-file topics.csv --output-dir ./reference_articles --text-format plain
"""

import os
import sys
import re
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

# Import json with error handling for naming conflicts
try:
    import json
    # Verify json module has required functions
    if not hasattr(json, 'dumps') or not hasattr(json, 'dump') or not hasattr(json, 'loads'):
        raise ImportError("json module missing required functions")
except ImportError as e:
    print(f"Error importing json module: {e}")
    print("This might be caused by a local file named 'json.py' conflicting with the standard library.")
    print("Please check for and remove any 'json.py' or 'json.pyc' files in your current directory.")
    sys.exit(1)

def clean_text(text: str, format_type: str = "markdown") -> str:
    """
    Clean and format text for better JSON readability.
    
    Args:
        text: Raw text content
        format_type: "markdown", "plain", or "minimal"
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text)  # Convert to string in case it's not
    
    # Normalize whitespace and line endings
    text = text.strip()
    text = re.sub(r'\r\n', '\n', text)  # Convert Windows line endings
    text = re.sub(r'\r', '\n', text)    # Convert Mac line endings
    
    if format_type == "plain":
        # Remove all markdown formatting for plain text
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # Remove headers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Remove code
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Remove links, keep text
        
    elif format_type == "minimal":
        # Keep structure but clean formatting
        text = re.sub(r'^#{4,}\s*', '### ', text, flags=re.MULTILINE)  # Max 3 levels
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Remove excessive blank lines
        
    # Always clean up excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines to double
    
    return text.strip()

def sanitize_topic_name(topic: str) -> str:
    """
    Sanitize topic name to match STORM article format.
    """
    if not topic or pd.isna(topic):
        return "unnamed_topic"
    
    topic = str(topic).strip()
    topic = topic.replace(" ", "_")
    topic = re.sub(r"[^a-zA-Z0-9_-]", "", topic)
    if not topic:
        topic = "unnamed_topic"
    return topic

def get_topic_name(row: pd.Series) -> str:
    """
    Extract topic name from CSV row, trying multiple possible columns.
    """
    # Based on your CSV structure, these are the topic title columns
    title_columns = [
        'title_in_wiki', 'title_in_britannica', 'norm_title_in_britannica',
        'norm_title_in_wiki1', 'norm_title_in_wiki2_maintable', 'norm_title_in_wiki2_wikitable'
    ]
    
    for col in title_columns:
        if col in row.index and pd.notna(row.get(col)) and str(row.get(col, '')).strip():
            return str(row[col]).strip()
    
    # Fallback to id if no title found
    if 'id' in row.index and pd.notna(row.get('id')):
        return f"topic_{row['id']}"
    
    return f"topic_{row.name if hasattr(row, 'name') else 'unknown'}"

def get_wikipedia_content(row: pd.Series) -> str:
    """Extract Wikipedia FULL TEXT content from row."""
    # Only extract full text content, not introductions
    if 'wiki_fulltext_plain' in row.index and pd.notna(row.get('wiki_fulltext_plain')):
        content = str(row['wiki_fulltext_plain']).strip()
        # Only return if it has substantial content (more than 100 characters)
        if len(content) > 100:
            return content
    
    return ""

def get_britannica_content(row: pd.Series) -> str:
    """Extract Britannica FULL TEXT content from row."""
    # Only extract full text content, not introductions
    if 'brit_fulltext' in row.index and pd.notna(row.get('brit_fulltext')):
        content = str(row['brit_fulltext']).strip()
        # Only return if it has substantial content (more than 100 characters)
        if len(content) > 100:
            return content
    
    return ""

def convert_csv_to_separate_json(csv_file: str, output_dir: str, text_format: str = "markdown", 
                               indent: int = 2) -> Dict[str, Any]:
    """
    Convert CSV file to separate Wikipedia and Britannica JSON files.
    
    Args:
        csv_file: Path to CSV file
        output_dir: Output directory for JSON files
        text_format: Text formatting option
        indent: JSON indentation
    
    Returns:
        Processing summary
    """
    # Load CSV
    try:
        df = pd.read_csv(csv_file)
        logging.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        logging.info(f"Columns: {list(df.columns)}")
        
        # Show column names to help identify content columns
        print("\nCSV Columns found:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize data structures for both sources
    wikipedia_data = {
        'model_name': 'wikipedia',
        'processing_timestamp': datetime.now().isoformat(),
        'source_info': {
            'type': 'reference_csv_wikipedia',
            'csv_file': str(Path(csv_file).absolute()),
            'total_rows': len(df),
            'columns': list(df.columns)
        },
        'topics': {}
    }
    
    britannica_data = {
        'model_name': 'britannica',
        'processing_timestamp': datetime.now().isoformat(),
        'source_info': {
            'type': 'reference_csv_britannica',
            'csv_file': str(Path(csv_file).absolute()),
            'total_rows': len(df),
            'columns': list(df.columns)
        },
        'topics': {}
    }
    
    wiki_successful = 0
    brit_successful = 0
    wiki_failed = 0
    brit_failed = 0
    no_content_rows = 0
    
    for index, row in df.iterrows():
        try:
            # Get topic name
            topic_name_raw = get_topic_name(row)
            topic_name_sanitized = sanitize_topic_name(topic_name_raw)
            
            # Get Wikipedia content
            wiki_content = get_wikipedia_content(row)
            
            # Get Britannica content  
            britannica_content = get_britannica_content(row)
            
            # Track if we found any content for this row
            found_content = False
            
            # Process Wikipedia content
            if wiki_content:
                cleaned_wiki = clean_text(wiki_content, text_format)
                if cleaned_wiki:
                    wikipedia_data['topics'][topic_name_sanitized] = {
                        'topic_name': topic_name_sanitized,
                        'original_topic_name': topic_name_raw,
                        'csv_row': index + 1,
                        'final_text': cleaned_wiki
                    }
                    wiki_successful += 1
                    found_content = True
                else:
                    wiki_failed += 1
            else:
                wiki_failed += 1
            
            # Process Britannica content
            if britannica_content:
                cleaned_brit = clean_text(britannica_content, text_format)
                if cleaned_brit:
                    britannica_data['topics'][topic_name_sanitized] = {
                        'topic_name': topic_name_sanitized,
                        'original_topic_name': topic_name_raw,
                        'csv_row': index + 1,
                        'final_text': cleaned_brit
                    }
                    brit_successful += 1
                    found_content = True
                else:
                    brit_failed += 1
            else:
                brit_failed += 1
            
            if not found_content:
                no_content_rows += 1
                logging.warning(f"Row {index + 1}: No FULL TEXT content found for topic '{topic_name_raw}' (skipping introductions)")
            
            # Progress logging
            if (index + 1) % 50 == 0:
                logging.info(f"Processed {index + 1} rows... Wiki full text: {wiki_successful}, Brit full text: {brit_successful}")
                
        except Exception as e:
            logging.error(f"Error processing row {index + 1}: {e}")
            continue
    
    # Save Wikipedia JSON file
    wiki_file = output_path / "wikipedia.json"
    with open(wiki_file, 'w', encoding='utf-8') as f:
        json.dump(wikipedia_data, f, indent=indent, ensure_ascii=False, separators=(',', ': '))
    
    # Save Britannica JSON file
    brit_file = output_path / "britannica.json"
    with open(brit_file, 'w', encoding='utf-8') as f:
        json.dump(britannica_data, f, indent=indent, ensure_ascii=False, separators=(',', ': '))
    
    # Create processing summary
    summary = {
        'input_csv': str(Path(csv_file).absolute()),
        'output_directory': str(output_path.absolute()),
        'output_files': {
            'wikipedia': str(wiki_file.absolute()),
            'britannica': str(brit_file.absolute())
        },
        'processing_timestamp': datetime.now().isoformat(),
        'text_format': text_format,
        'statistics': {
            'total_rows_in_csv': len(df),
            'wikipedia': {
                'successful_topics': wiki_successful,
                'failed_topics': wiki_failed,
                'success_rate': f"{wiki_successful / len(df) * 100:.1f}%" if len(df) > 0 else "0%"
            },
            'britannica': {
                'successful_topics': brit_successful,
                'failed_topics': brit_failed,
                'success_rate': f"{brit_successful / len(df) * 100:.1f}%" if len(df) > 0 else "0%"
            },
            'rows_with_no_content': no_content_rows
        },
        'csv_columns_found': list(df.columns),
        'content_columns_detected': {
            'wikipedia_columns': ['wiki_fulltext_plain'] if 'wiki_fulltext_plain' in df.columns else [],
            'britannica_columns': ['brit_fulltext'] if 'brit_fulltext' in df.columns else []
        }
    }
    
    # Save summary
    summary_file = output_path / "conversion_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, separators=(',', ': '))
    
    logging.info(f"Wikipedia: {wiki_successful} topics saved to {wiki_file}")
    logging.info(f"Britannica: {brit_successful} topics saved to {brit_file}")
    logging.info(f"Conversion summary saved to: {summary_file}")
    
    return summary

def print_summary(summary: Dict[str, Any]) -> None:
    """Print conversion summary for separate Wikipedia and Britannica files."""
    print("\n" + "="*70)
    print("CSV TO SEPARATE JSON CONVERSION SUMMARY")
    print("="*70)
    
    print(f"Input CSV: {Path(summary['input_csv']).name}")
    print(f"Output Directory: {summary['output_directory']}")
    print(f"Text Format: {summary['text_format']}")
    print(f"Processing Time: {summary['processing_timestamp']}")
    print()
    
    stats = summary['statistics']
    print(f"Total CSV Rows: {stats['total_rows_in_csv']}")
    print(f"Rows with No Content: {stats['rows_with_no_content']}")
    print()
    
    # Wikipedia statistics
    wiki_stats = stats['wikipedia']
    print(f"📚 WIKIPEDIA ({Path(summary['output_files']['wikipedia']).name})")
    print(f"   Successful Topics: {wiki_stats['successful_topics']}")
    print(f"   Failed Topics: {wiki_stats['failed_topics']}")
    print(f"   Success Rate: {wiki_stats['success_rate']}")
    print()
    
    # Britannica statistics
    brit_stats = stats['britannica']
    print(f"📖 BRITANNICA ({Path(summary['output_files']['britannica']).name})")
    print(f"   Successful Topics: {brit_stats['successful_topics']}")
    print(f"   Failed Topics: {brit_stats['failed_topics']}")
    print(f"   Success Rate: {brit_stats['success_rate']}")
    print()
    
    # Column detection
    content_cols = summary['content_columns_detected']
    if content_cols['wikipedia_columns']:
        print(f"Wikipedia Full Text Column: {', '.join(content_cols['wikipedia_columns'])}")
    else:
        print("⚠️  Wikipedia full text column (wiki_fulltext_plain) not found")
    
    if content_cols['britannica_columns']:
        print(f"Britannica Full Text Column: {', '.join(content_cols['britannica_columns'])}")
    else:
        print("⚠️  Britannica full text column (brit_fulltext) not found")
    
    print(f"\nTotal CSV Columns: {len(summary['csv_columns_found'])}")
    print("First few columns:", ", ".join(summary['csv_columns_found'][:5]), "..." if len(summary['csv_columns_found']) > 5 else "")

def main():
    parser = ArgumentParser(description="Convert CSV to separate Wikipedia and Britannica JSON files")
    parser.add_argument(
        "--csv-file",
        type=str,
        required=True,
        help="Path to CSV file containing Wikipedia and Britannica articles"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reference_articles",
        help="Output directory for JSON files (will create wikipedia.json and britannica.json)"
    )
    parser.add_argument(
        "--text-format",
        type=str,
        choices=["markdown", "plain", "minimal"],
        default="markdown",
        help="Text formatting: 'markdown' (keep all), 'plain' (remove formatting), 'minimal' (clean but keep structure)"
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (0 for compact, 2-4 recommended for readability)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Convert CSV to separate JSON files
        logging.info("Starting CSV to separate Wikipedia and Britannica JSON conversion...")
        summary = convert_csv_to_separate_json(
            args.csv_file,
            args.output_dir,
            args.text_format,
            args.indent
        )
        
        # Print summary
        if not args.quiet:
            print_summary(summary)
        
        wiki_count = summary['statistics']['wikipedia']['successful_topics']
        brit_count = summary['statistics']['britannica']['successful_topics']
        
        print(f"\n✅ Successfully created separate JSON files:")
        print(f"   📚 wikipedia.json: {wiki_count} topics")
        print(f"   📖 britannica.json: {brit_count} topics")
        print(f"   📊 Output directory: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())