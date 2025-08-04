#!/usr/bin/env python3
"""
Article Consolidation Script - Simple JSON per Model

This script consolidates all generated STORM articles into simple structured JSON files.
Each JSON file contains all topics for one model with just the final and initial text.

Usage:
    python consolidate_articles.py --input-dir ./results/multiturn_storm --output-dir ./consolidated_articles
    python consolidate_articles.py --input-dir ./results/multiturn_storm --output-dir ./consolidated_articles --models deepseek_deepseek-chat
"""

import os
import sys
import re
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
    if not text:
        return ""
    
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

def process_single_topic(topic_dir: Path, model_name: str, text_format: str = "markdown") -> Optional[Dict[str, Any]]:
    """Process a single topic directory and extract article texts."""
    try:
        topic_data = {
            'topic_name': topic_dir.name,
            'model': model_name
        }
        
        # Check for required files
        final_article_path = topic_dir / "storm_gen_article.txt"
        initial_article_path = topic_dir / "initial_article.txt"
        
        # Process final article
        if final_article_path.exists():
            with open(final_article_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                topic_data['final_text'] = clean_text(raw_text, text_format)
        
        # Process initial article
        if initial_article_path.exists():
            with open(initial_article_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                topic_data['initial_text'] = clean_text(raw_text, text_format)
        
        # Only return if we have at least one article
        if 'final_text' in topic_data or 'initial_text' in topic_data:
            return topic_data
        else:
            return None
        
    except Exception as e:
        logging.error(f"Error processing topic directory {topic_dir}: {e}")
        return None

def consolidate_articles_by_model(input_dir: str, output_dir: str, models: Optional[List[str]] = None, 
                                text_format: str = "markdown", indent: int = 2) -> Dict[str, Any]:
    """
    Consolidate articles and create simple JSON files for each model.
    
    Args:
        input_dir: Path to the multiturn_storm results directory
        output_dir: Directory to save JSON files
        models: List of specific model directories to process
    
    Returns:
        Dictionary with processing summary
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all model directories
    model_dirs = []
    for item in input_path.iterdir():
        if item.is_dir():
            # Skip hidden directories and common non-model directories
            if item.name.startswith('.') or item.name in ['__pycache__', 'logs']:
                continue
            if models is None or item.name in models:
                model_dirs.append(item)
    
    if not model_dirs:
        if models:
            raise ValueError(f"No matching model directories found for: {models}")
        else:
            raise ValueError(f"No model directories found in: {input_dir}")
    
    logging.info(f"Found {len(model_dirs)} model directories to process")
    
    processing_summary = {
        'input_directory': str(input_path.absolute()),
        'output_directory': str(output_path.absolute()),
        'processing_timestamp': datetime.now().isoformat(),
        'total_models': len(model_dirs),
        'models_processed': {},
        'total_topics_across_all_models': 0
    }
    
    # Process each model directory
    for model_dir in model_dirs:
        model_name = model_dir.name
        logging.info(f"Processing model: {model_name}")
        
        # Create simple JSON structure for this model
        json_file = output_path / f"{model_name}.json"
        
        model_data = {
            'model_name': model_name,
            'processing_timestamp': datetime.now().isoformat(),
            'topics': {}
        }
        
        # Find all topic directories within this model directory
        topic_dirs = [d for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        logging.info(f"Found {len(topic_dirs)} topic directories in {model_name}")
        
        if not topic_dirs:
            logging.warning(f"No topic directories found in model directory: {model_name}")
            # Save empty model file
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)
            continue
        
        successful_topics = 0
        failed_topics = 0
        
        # Process each topic
        for topic_dir in topic_dirs:
            try:
                topic_data = process_single_topic(topic_dir, model_name, text_format)
                
                if topic_data:
                    topic_name = topic_data['topic_name']
                    # Remove model name from individual topic data since it's redundant
                    topic_info = {
                        'topic_name': topic_name
                    }
                    
                    if 'final_text' in topic_data:
                        topic_info['final_text'] = topic_data['final_text']
                    
                    if 'initial_text' in topic_data:
                        topic_info['initial_text'] = topic_data['initial_text']
                    
                    model_data['topics'][topic_name] = topic_info
                    successful_topics += 1
                        
                else:
                    failed_topics += 1
                    logging.warning(f"Failed to process topic: {topic_dir.name}")
                        
            except Exception as e:
                failed_topics += 1
                logging.error(f"Error processing topic {topic_dir.name} in model {model_name}: {e}")
                continue
        
        # Save model JSON file with custom formatting
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=indent, ensure_ascii=False, separators=(',', ': '))
        
        logging.info(f"Model {model_name}: {successful_topics} topics saved to {json_file}")
        
        # Update processing summary
        processing_summary['models_processed'][model_name] = {
            'json_file': str(json_file.absolute()),
            'topics_processed': successful_topics,
            'topics_failed': failed_topics
        }
        
        processing_summary['total_topics_across_all_models'] += successful_topics
    
    # Save processing summary with custom formatting
    summary_file = output_path / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(processing_summary, f, indent=2, ensure_ascii=False, separators=(',', ': '))
    
    logging.info(f"Processing summary saved to: {summary_file}")
    
    return processing_summary

def print_summary(summary: Dict[str, Any]) -> None:
    """Print a summary of the processing results."""
    print("\n" + "="*60)
    print("ARTICLE CONSOLIDATION SUMMARY")
    print("="*60)
    
    print(f"Input Directory: {summary['input_directory']}")
    print(f"Output Directory: {summary['output_directory']}")
    print(f"Processing Time: {summary['processing_timestamp']}")
    print(f"Total Models: {summary['total_models']}")
    print(f"Total Topics: {summary['total_topics_across_all_models']}")
    print()
    
    for model_name, model_stats in summary['models_processed'].items():
        print(f"📁 {model_name}.json")
        print(f"   Topics Processed: {model_stats['topics_processed']}")
        if model_stats['topics_failed'] > 0:
            print(f"   Topics Failed: {model_stats['topics_failed']}")
        print()

def diagnose_json_issue():
    """Diagnose potential json module conflicts."""
    print("\n🔍 JSON Module Diagnostics:")
    print("-" * 40)
    
    # Check json module location
    import json
    print(f"JSON module location: {json.__file__}")
    
    # Check available functions
    required_functions = ['dumps', 'dump', 'loads', 'load']
    for func in required_functions:
        has_func = hasattr(json, func)
        print(f"json.{func}: {'✅' if has_func else '❌'}")
    
    # Check for conflicting files in current directory
    current_dir = Path.cwd()
    json_files = list(current_dir.glob("json.py*"))
    if json_files:
        print(f"\n⚠️  Found potential conflicting files:")
        for file in json_files:
            print(f"  - {file}")
        print("These files may be interfering with the standard json module.")
    else:
        print("\n✅ No conflicting json.py files found in current directory.")

def main():
    parser = ArgumentParser(description="Consolidate STORM articles into simple JSON files per model")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./results/multiturn_storm",
        help="Input directory containing model subdirectories with articles"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./consolidated_articles",
        help="Output directory for JSON files (one per model)"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Specific model directories to process (e.g., deepseek_deepseek-chat openai_gpt-4o). If not specified, processes all models."
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
        "--diagnose",
        action="store_true",
        help="Run diagnostics to check for json module conflicts"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    # Run diagnostics if requested
    if args.diagnose:
        diagnose_json_issue()
        return 0
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Consolidate articles
        logging.info("Starting article consolidation (Simple JSON per model)...")
        summary = consolidate_articles_by_model(
            args.input_dir, 
            args.output_dir, 
            args.models,
            args.text_format,
            args.indent
        )
        
        # Print summary
        if not args.quiet:
            print_summary(summary)
        
        print(f"\n✅ Successfully created JSON files in: {args.output_dir}")
        print(f"📊 Total: {summary['total_models']} JSON files, {summary['total_topics_across_all_models']} topics")
        
    except Exception as e:
        logging.error(f"Error during consolidation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())