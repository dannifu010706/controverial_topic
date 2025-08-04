#!/usr/bin/env python3
"""
NER Processing Script for JSON Articles

This script performs Named Entity Recognition on JSON files containing articles
using Stanza. It processes each topic separately and saves results for later
argument analysis.

Usage:
    python ner_processor.py --input-dir ./consolidated_articles --output-dir ./ner_results
    python ner_processor.py --input-dir ./reference_articles --output-dir ./ner_results --sources wikipedia britannica
"""

import os
import json
import sys
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from collections import Counter, defaultdict
import time

# Import stanza with error handling
try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False
    print("Warning: Stanza is not installed. Please install it with: pip install stanza")

def setup_stanza(language='en', use_gpu=False) -> Optional[stanza.Pipeline]:
    """
    Setup Stanza NLP pipeline for NER.
    
    Args:
        language: Language code (default: 'en')
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Stanza pipeline or None if setup fails
    """
    if not STANZA_AVAILABLE:
        raise ImportError("Stanza is required but not installed. Run: pip install stanza")
    
    try:
        # Download language model if not already downloaded
        logging.info(f"Setting up Stanza for language: {language}")
        stanza.download(language, verbose=False)
        
        # Create pipeline with NER processor
        nlp = stanza.Pipeline(
            lang=language,
            processors='tokenize,ner',
            use_gpu=use_gpu,
            verbose=False
        )
        
        logging.info("Stanza pipeline successfully initialized")
        return nlp
        
    except Exception as e:
        logging.error(f"Failed to setup Stanza pipeline: {e}")
        return None

def clean_text_for_ner(text: str) -> str:
    """
    Clean text before NER processing.
    
    Args:
        text: Raw text content
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Basic cleaning
    text = text.strip()
    
    # Remove excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # Remove some problematic characters that might confuse NER
    text = re.sub(r'[^\w\s\.,;:!?\-\(\)\'\"\/]', ' ', text)
    
    return text

def extract_entities_from_text(nlp: stanza.Pipeline, text: str, topic_name: str) -> Dict[str, Any]:
    """
    Extract named entities from text using Stanza.
    
    Args:
        nlp: Stanza pipeline
        text: Text to process
        topic_name: Name of the topic (for logging)
    
    Returns:
        Dictionary containing extracted entities and statistics
    """
    try:
        # Clean text
        cleaned_text = clean_text_for_ner(text)
        
        if not cleaned_text or len(cleaned_text.strip()) < 10:
            logging.warning(f"Topic '{topic_name}': Text too short or empty for NER")
            return {
                "entities": [],
                "entity_counts": {},
                "total_entities": 0,
                "text_length": len(text),
                "processing_status": "skipped_short_text"
            }
        
        # Process with Stanza
        doc = nlp(cleaned_text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.type,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "confidence": getattr(ent, 'confidence', None)  # If available
            })
        
        # Count entities by type
        entity_counts = Counter([ent["label"] for ent in entities])
        
        result = {
            "entities": entities,
            "entity_counts": dict(entity_counts),
            "total_entities": len(entities),
            "text_length": len(text),
            "cleaned_text_length": len(cleaned_text),
            "processing_status": "success"
        }
        
        logging.debug(f"Topic '{topic_name}': Found {len(entities)} entities")
        return result
        
    except Exception as e:
        logging.error(f"Error processing topic '{topic_name}': {e}")
        return {
            "entities": [],
            "entity_counts": {},
            "total_entities": 0,
            "text_length": len(text),
            "processing_status": f"error: {str(e)}"
        }

def process_json_file(file_path: Path, nlp: stanza.Pipeline, source_name: str) -> Dict[str, Any]:
    """
    Process a single JSON file and extract entities from all topics.
    
    Args:
        file_path: Path to JSON file
        nlp: Stanza pipeline
        source_name: Name of the source (e.g., 'wikipedia', 'deepseek')
    
    Returns:
        Dictionary containing NER results for all topics in the file
    """
    logging.info(f"Processing {source_name}: {file_path}")
    
    try:
        # Load JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'topics' not in data:
            logging.error(f"No 'topics' key found in {file_path}")
            return None
        
        # Initialize results structure
        results = {
            "source_name": source_name,
            "source_file": str(file_path.absolute()),
            "processing_timestamp": datetime.now().isoformat(),
            "total_topics": len(data['topics']),
            "topics": {}
        }
        
        # Process each topic
        processed_count = 0
        failed_count = 0
        
        for topic_name, topic_data in data['topics'].items():
            try:
                # Get text content
                text_content = topic_data.get('final_text', '')
                
                if not text_content:
                    logging.warning(f"No final_text found for topic: {topic_name}")
                    continue
                
                # Extract entities
                ner_result = extract_entities_from_text(nlp, text_content, topic_name)
                
                # Add topic metadata
                ner_result.update({
                    "topic_name": topic_name,
                    "original_topic_name": topic_data.get('original_topic_name', topic_name),
                    "csv_row": topic_data.get('csv_row', None)
                })
                
                results['topics'][topic_name] = ner_result
                processed_count += 1
                
                # Progress logging
                if processed_count % 10 == 0:
                    logging.info(f"  Processed {processed_count}/{len(data['topics'])} topics...")
                
            except Exception as e:
                logging.error(f"Failed to process topic '{topic_name}': {e}")
                failed_count += 1
                continue
        
        # Update summary stats
        results.update({
            "topics_processed": processed_count,
            "topics_failed": failed_count,
            "processing_success_rate": f"{processed_count / len(data['topics']) * 100:.1f}%" if data['topics'] else "0%"
        })
        
        logging.info(f"Completed {source_name}: {processed_count} topics processed, {failed_count} failed")
        return results
        
    except Exception as e:
        logging.error(f"Failed to process file {file_path}: {e}")
        return None

def save_ner_results(results: Dict[str, Any], output_dir: Path) -> Path:
    """
    Save NER results to JSON file.
    
    Args:
        results: NER results dictionary
        output_dir: Output directory
    
    Returns:
        Path to saved file
    """
    source_name = results['source_name']
    output_file = output_dir / f"{source_name}_ner_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Saved NER results to: {output_file}")
    return output_file

def generate_summary_statistics(all_results: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Generate summary statistics across all sources.
    
    Args:
        all_results: List of NER results from all sources
        output_dir: Output directory for summary
    """
    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "total_sources": len(all_results),
        "sources_summary": {},
        "overall_statistics": {
            "total_topics_across_sources": 0,
            "total_entities_across_sources": 0,
            "entity_type_distribution": defaultdict(int),
            "average_entities_per_topic": 0,
            "average_text_length": 0
        }
    }
    
    total_entities = 0
    total_text_length = 0
    total_topics = 0
    
    for result in all_results:
        source_name = result['source_name']
        
        # Source-level statistics
        source_entities = 0
        source_text_length = 0
        source_entity_types = defaultdict(int)
        
        for topic_name, topic_data in result['topics'].items():
            topic_entities = topic_data['total_entities']
            topic_text_length = topic_data['text_length']
            
            source_entities += topic_entities
            source_text_length += topic_text_length
            
            # Count entity types
            for entity_type, count in topic_data['entity_counts'].items():
                source_entity_types[entity_type] += count
                summary['overall_statistics']['entity_type_distribution'][entity_type] += count
        
        summary['sources_summary'][source_name] = {
            "topics_processed": result['topics_processed'],
            "topics_failed": result['topics_failed'],
            "total_entities": source_entities,
            "total_text_length": source_text_length,
            "average_entities_per_topic": source_entities / result['topics_processed'] if result['topics_processed'] > 0 else 0,
            "average_text_length": source_text_length / result['topics_processed'] if result['topics_processed'] > 0 else 0,
            "entity_type_distribution": dict(source_entity_types)
        }
        
        total_entities += source_entities
        total_text_length += source_text_length
        total_topics += result['topics_processed']
    
    # Update overall statistics
    summary['overall_statistics'].update({
        "total_topics_across_sources": total_topics,
        "total_entities_across_sources": total_entities,
        "average_entities_per_topic": total_entities / total_topics if total_topics > 0 else 0,
        "average_text_length": total_text_length / total_topics if total_topics > 0 else 0,
        "entity_type_distribution": dict(summary['overall_statistics']['entity_type_distribution'])
    })
    
    # Save summary
    summary_file = output_dir / "ner_processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Summary statistics saved to: {summary_file}")

def find_json_files(input_dir: Path, specific_sources: Optional[List[str]] = None) -> List[Tuple[Path, str]]:
    """
    Find JSON files to process.
    
    Args:
        input_dir: Input directory
        specific_sources: List of specific source names to process
    
    Returns:
        List of (file_path, source_name) tuples
    """
    json_files = []
    
    for file_path in input_dir.glob("*.json"):
        # Skip summary files
        if "summary" in file_path.name.lower():
            continue
        
        # Extract source name from filename
        source_name = file_path.stem
        
        # Filter by specific sources if provided
        if specific_sources and source_name not in specific_sources:
            continue
        
        json_files.append((file_path, source_name))
    
    return json_files

def print_processing_summary(all_results: List[Dict[str, Any]]) -> None:
    """Print a summary of processing results."""
    print("\n" + "="*70)
    print("NER PROCESSING SUMMARY")
    print("="*70)
    
    total_topics = 0
    total_entities = 0
    
    for result in all_results:
        source_name = result['source_name']
        topics_processed = result['topics_processed']
        topics_failed = result['topics_failed']
        
        # Calculate source statistics
        source_entities = sum(topic['total_entities'] for topic in result['topics'].values())
        avg_entities = source_entities / topics_processed if topics_processed > 0 else 0
        
        print(f"\n📊 {source_name.upper()}")
        print(f"   Topics Processed: {topics_processed}")
        if topics_failed > 0:
            print(f"   Topics Failed: {topics_failed}")
        print(f"   Total Entities: {source_entities}")
        print(f"   Avg Entities/Topic: {avg_entities:.1f}")
        
        total_topics += topics_processed
        total_entities += source_entities
    
    print(f"\n🎯 OVERALL TOTALS")
    print(f"   Sources Processed: {len(all_results)}")
    print(f"   Total Topics: {total_topics}")
    print(f"   Total Entities: {total_entities}")
    print(f"   Overall Avg Entities/Topic: {total_entities / total_topics:.1f}" if total_topics > 0 else 0)

def main():
    parser = ArgumentParser(description="Perform NER on JSON article files using Stanza")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing JSON files to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ner_results",
        help="Output directory for NER results"
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        help="Specific source names to process (e.g., wikipedia britannica deepseek). If not specified, processes all JSON files."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for Stanza (default: en)"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration for Stanza (requires CUDA)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Setup Stanza
        logging.info("Initializing Stanza NLP pipeline...")
        nlp = setup_stanza(language=args.language, use_gpu=args.use_gpu)
        if nlp is None:
            return 1
        
        # Find JSON files to process
        json_files = find_json_files(input_dir, args.sources)
        
        if not json_files:
            logging.error(f"No JSON files found in {input_dir}")
            if args.sources:
                logging.error(f"Specifically looking for: {args.sources}")
            return 1
        
        logging.info(f"Found {len(json_files)} JSON files to process")
        
        # Process each file
        all_results = []
        start_time = time.time()
        
        for i, (file_path, source_name) in enumerate(json_files):
            logging.info(f"Processing file {i+1}/{len(json_files)}: {source_name}")
            
            # Process the file
            results = process_json_file(file_path, nlp, source_name)
            
            if results:
                # Save results
                save_ner_results(results, output_dir)
                all_results.append(results)
            else:
                logging.error(f"Failed to process {source_name}")
        
        # Generate summary statistics
        if all_results:
            generate_summary_statistics(all_results, output_dir)
            
            # Print summary
            if not args.quiet:
                print_processing_summary(all_results)
            
            processing_time = time.time() - start_time
            print(f"\n✅ NER processing completed successfully!")
            print(f"📊 Processed {len(all_results)} sources in {processing_time:.1f} seconds")
            print(f"📁 Results saved to: {output_dir}")
        else:
            logging.error("No files were successfully processed")
            return 1
        
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())