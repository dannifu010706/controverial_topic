import os
import re
import logging
import requests
import json
import csv
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

try:
    import aisuite as ai
    AISUITE_AVAILABLE = True
except ImportError:
    AISUITE_AVAILABLE = False

def count_words(text):
    """
    Count words in text, handling None/empty values.
    """
    if not text or pd.isna(text):
        return 0
    word_count = len(str(text).split())
    return word_count

def sanitize_topic(topic):
    """
    Sanitize the topic name for use in file names.
    Remove or replace characters that are not allowed in file names.
    """
    topic = topic.replace(" ", "_")
    topic = re.sub(r"[^a-zA-Z0-9_-]", "", topic)
    if not topic:
        topic = "unnamed_topic"
    return topic

def load_api_key_from_toml(toml_file_path="secrets.toml"):
    """
    Load API key from TOML file if it exists.
    """
    if os.path.exists(toml_file_path):
        try:
            import tomli
            with open(toml_file_path, "rb") as f:
                data = tomli.load(f)
                for key, value in data.items():
                    if key not in os.environ:
                        os.environ[key] = str(value)
        except ImportError:
            with open(toml_file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key not in os.environ:
                            os.environ[key] = value

class UnifiedModel:
    """
    Unified model interface supporting multiple providers through aisuite.
    """
    def __init__(self, provider="deepseek", model="deepseek-chat", temperature=1.0, top_p=0.9, max_tokens=None):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        if provider == "deepseek":
            self.full_model = f"deepseek:{model}"
        elif provider == "openai":
            self.full_model = f"openai:{model}"
        elif provider == "anthropic":
            self.full_model = f"anthropic:{model}"
        elif provider == "cohere":
            self.full_model = f"cohere:{model}"
        elif provider == "mistral":
            self.full_model = f"mistral:{model}"
        elif provider == "gemini":
            self.full_model = f"google:{model}"
        elif provider == "groq":
            self.full_model = f"groq:{model}"
        else:
            self.full_model = f"{provider}:{model}"
        if AISUITE_AVAILABLE:
            self.client = ai.Client()
        else:
            if provider not in ["deepseek", "groq"]:
                raise ValueError(f"Provider '{provider}' requires aisuite. Please install: pip install aisuite")
            self.client = None
    
    def generate(self, prompt, max_tokens=None):
        """
        Generate text using the specified provider.
        """
        if AISUITE_AVAILABLE and self.client:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.full_model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens
            )
            return response.choices[0].message.content
        else:
            if self.provider == "deepseek":
                return self._deepseek_generate(prompt, max_tokens)
            elif self.provider == "groq":
                return self._groq_generate(prompt, max_tokens)
            else:
                raise ValueError(f"Provider '{self.provider}' not available without aisuite")
    
    def _deepseek_generate(self, prompt, max_tokens=None):
        """
        Direct DeepSeek API call (fallback when aisuite not available).
        """
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek provider")
        api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if max_tokens is not None or self.max_tokens is not None:
            data["max_tokens"] = max_tokens if max_tokens is not None else self.max_tokens
        response = requests.post(
            f"{api_base}/v1/chat/completions",
            headers=headers,
            json=data
        )
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _groq_generate(self, prompt, max_tokens=None):
        """
        Direct Groq API call (fallback when aisuite not available).
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for Groq provider")
        api_base = os.getenv("GROQ_API_BASE", "https://api.groq.com").rstrip("/")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if max_tokens is not None or self.max_tokens is not None:
            data["max_tokens"] = max_tokens if max_tokens is not None else self.max_tokens
        response = requests.post(
            f"{api_base}/v1/chat/completions",
            headers=headers,
            json=data
        )
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
        result = response.json()
        return result["choices"][0]["message"]["content"]

class MultiTurnSTORMRunner:
    """
    Multi-turn STORM runner that generates comprehensive articles in two phases:
    1. Generate initial article with many sections
    2. Expand each section individually
    """
    def __init__(self, output_dir, provider="deepseek", model="deepseek-chat", temperature=1.0, top_p=0.9):
        self.output_dir = output_dir
        self.provider = provider
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.article_gen_lm = UnifiedModel(
            provider=provider, 
            model=model, 
            temperature=temperature, 
            top_p=top_p, 
            max_tokens=None
        )
    
    def generate_article(self, topic):
        """
        Generate comprehensive article using multi-turn approach:
        Turn 1: Generate initial article with many sections
        Turn 2: Expand each section individually
        """
        self.logger.info(f"=== MULTI-TURN ARTICLE GENERATION FOR: {topic} ===")
        
        # TURN 1: Generate initial comprehensive article with many sections
        self.logger.info("TURN 1: Generating initial comprehensive article with many sections...")
        initial_article = self._generate_initial_comprehensive_article(topic)
        initial_words = count_words(initial_article)
        self.logger.info(f"Initial article generated: {initial_words} words")
        
        # TURN 2: Expand each section individually
        self.logger.info("TURN 2: Expanding each section individually...")
        expanded_article = self._expand_all_sections(topic, initial_article)
        final_words = count_words(expanded_article)
        
        self.logger.info(f"=== GENERATION COMPLETE ===")
        self.logger.info(f"Initial: {initial_words} words → Final: {final_words} words")
        self.logger.info(f"Expansion added: {final_words - initial_words} words")
        
        return expanded_article, initial_article
    
    def _generate_initial_comprehensive_article(self, topic):
        """
        Turn 1: Generate initial article with comprehensive section coverage.
        """
        initial_prompt = f"""Write a comprehensive encyclopedia-style article about "{topic.replace('_', ' ').title()}".

Requirements for comprehensive coverage:
1. Start with the main title "# {topic.replace('_', ' ').title()}"
2. Include a substantial introduction that defines the topic and explains its significance
3. Create MANY major sections covering ALL important aspects of this topic
4. Use ONLY ## level headings for sections - no subsections
5. Include detailed content for each section
6. Use an encyclopedic tone throughout
7. Do NOT include any conclusion, summary, or final thoughts sections
8. Do NOT add sections about future prospects or outlook
9. Focus on informative content

IMPORTANT: This is the initial comprehensive coverage phase. Create as many relevant major sections as needed to thoroughly cover the topic from all important angles. Each section should be substantial enough to provide good foundational information, but they will be expanded in a second phase.

Topic: {topic.replace('_', ' ')}
Write the complete initial comprehensive article:"""
        
        initial_article = self.article_gen_lm.generate(initial_prompt)
        
        # Log section analysis
        sections = self._parse_article_sections(initial_article)
        self.logger.info(f"Initial article has {len(sections)} sections:")
        for i, (title, content) in enumerate(sections):
            section_words = count_words(content)
            self.logger.info(f"  {i+1}. {title} ({section_words} words)")
        
        return initial_article
    
    def _expand_all_sections(self, topic, initial_article):
        """
        Turn 2: Expand each section individually by sending it back to the model.
        """
        sections = self._parse_article_sections(initial_article)
        self.logger.info(f"Expanding {len(sections)} sections individually...")
        
        # Extract title and introduction (content before first section)
        article_parts = initial_article.split('##', 1)
        if len(article_parts) == 2:
            title_and_intro = article_parts[0].strip()
        else:
            title_and_intro = initial_article.strip()
        
        expanded_sections = []
        
        for i, (section_title, section_content) in enumerate(sections):
            self.logger.info(f"Expanding section {i+1}/{len(sections)}: '{section_title}'")
            
            original_words = count_words(section_content)
            expanded_section = self._expand_single_section(topic, section_title, section_content)
            expanded_words = count_words(expanded_section)
            
            self.logger.info(f"  Section '{section_title}': {original_words} → {expanded_words} words (+{expanded_words - original_words})")
            expanded_sections.append(expanded_section)
        
        # Reconstruct the full article
        full_expanded_article = title_and_intro
        if not full_expanded_article.endswith('\n'):
            full_expanded_article += '\n'
        
        for section in expanded_sections:
            full_expanded_article += f"\n{section}\n"
        
        return full_expanded_article.strip()
    
    def _expand_single_section(self, topic, section_title, section_content):
        """
        Expand a single section by sending it back to the model for detailed expansion.
        """
        expansion_prompt = f"""You are expanding a section from an encyclopedia article about "{topic.replace('_', ' ')}". 

Current section that needs expansion:
## {section_title}
{section_content}

Your task is to significantly expand this section with much more detail, depth, and comprehensive coverage:

Requirements:
1. Keep the same section heading: ## {section_title}
2. Greatly expand the content with more specific details, examples, explanations, and context
3. Add more depth while maintaining the encyclopedic tone
4. Include more specific information and details
5. Make the section much more comprehensive and informative
6. Write in detailed paragraphs - do NOT create subsections
7. Do NOT add any conclusion or summary statements at the end
8. Focus purely on expanding the informational content related to this specific aspect of {topic.replace('_', ' ')}

Write the complete expanded section with significantly more detail and depth:"""
        
        expanded_section = self.article_gen_lm.generate(expansion_prompt)
        return expanded_section.strip()
    
    def _parse_article_sections(self, article):
        """
        Parse article into sections with their content.
        Returns list of (title, content) tuples.
        """
        sections = []
        lines = article.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Check if this is a section header
            match = re.match(r'^##\s+(.+)', line)
            if match:
                # Save previous section if exists
                if current_section:
                    sections.append((current_section, '\n'.join(current_content)))
                # Start new section
                current_section = match.group(1).strip()
                current_content = []
            else:
                # Add to current section content
                if current_section:
                    current_content.append(line)
        
        # Don't forget the last section
        if current_section:
            sections.append((current_section, '\n'.join(current_content)))
        
        return sections
    
    def run_single_topic(self, topic):
        """
        Generate article for a single topic using multi-turn approach.
        """
        # Create organized directory structure: output_dir/model_name/topic
        model_dir = Path(self.output_dir) / f"{self.provider}_{self.model}"
        topic_dir = model_dir / topic
        topic_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting multi-turn article generation for topic: {topic}")
        self.logger.info(f"Model: {self.provider}_{self.model}")
        self.logger.info(f"Output directory: {topic_dir}")
        
        # Generate article using multi-turn approach
        final_article, initial_article = self.generate_article(topic)
        
        # Save both versions
        final_article_file = topic_dir / "storm_gen_article.txt"
        with open(final_article_file, "w", encoding="utf-8") as f:
            f.write(final_article)
        self.logger.info(f"Final article saved to: {final_article_file}")
        
        initial_article_file = topic_dir / "initial_article.txt"
        with open(initial_article_file, "w", encoding="utf-8") as f:
            f.write(initial_article)
        self.logger.info(f"Initial article saved to: {initial_article_file}")
        
        # Save generation info
        generation_info_file = topic_dir / "generation_info.txt"
        with open(generation_info_file, "w", encoding="utf-8") as f:
            f.write(f"Topic: {topic}\n")
            f.write(f"Generation Method: Multi-Turn (Initial + Section Expansion)\n\n")
            
            initial_words = count_words(initial_article)
            final_words = count_words(final_article)
            f.write(f"Initial article length: {initial_words} words\n")
            f.write(f"Final article length: {final_words} words\n")
            f.write(f"Words added through expansion: {final_words - initial_words} words\n\n")
            
            # Section analysis
            initial_sections = self._parse_article_sections(initial_article)
            final_sections = self._parse_article_sections(final_article)
            f.write(f"Number of sections: {len(initial_sections)}\n\n")
            
            f.write("Section expansion details:\n")
            for i, ((init_title, init_content), (final_title, final_content)) in enumerate(zip(initial_sections, final_sections)):
                init_words = count_words(init_content)
                final_words = count_words(final_content)
                f.write(f"  {i+1}. {init_title}: {init_words} → {final_words} words (+{final_words - init_words})\n")
        
        return topic_dir
    
    def run_from_csv(self, csv_file, max_topics=None, start_topic=None):
        """
        Generate articles for topics in CSV file using multi-turn approach.
        
        Args:
            csv_file: Path to CSV file containing topics
            max_topics: Maximum number of topics to process
            start_topic: Starting topic number (1-based index)
        """
        self.logger.info(f"Loading topics from CSV: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # Only need topic title columns - we're not using reference lengths anymore
        title_columns = ['title_in_wiki', 'title_in_britannica']
        available_title_columns = [col for col in title_columns if col in df.columns]
        if not available_title_columns:
            available_cols = list(df.columns)
            raise ValueError(f"Missing required title columns. Need at least one of: {title_columns}. Available columns: {available_cols}")
        
        total_topics = len(df)
        
        # Apply start_topic filter first
        if start_topic is not None and start_topic > 0:
            if start_topic > total_topics:
                raise ValueError(f"Start topic {start_topic} is greater than total topics {total_topics} in CSV")
            
            # Convert to 0-based index for pandas
            start_index = start_topic - 1
            df = df.iloc[start_index:]
            self.logger.info(f"Starting from topic {start_topic} (row {start_topic} in CSV)")
            self.logger.info(f"Remaining topics after start filter: {len(df)}")
        
        # Apply max_topics limit after start_topic filter
        if max_topics is not None and max_topics > 0:
            df = df.head(max_topics)
            self.logger.info(f"Limited to processing {len(df)} topics (--max-topics={max_topics})")
        
        # Final processing info
        processing_start = start_topic if start_topic else 1
        processing_end = processing_start + len(df) - 1
        self.logger.info(f"Processing topics {processing_start} to {processing_end} out of {total_topics} total topics")
        
        results = []
        
        for index, row in df.iterrows():
            try:
                topic = None
                if pd.notna(row.get('title_in_wiki')) and str(row.get('title_in_wiki', '')).strip():
                    topic = str(row['title_in_wiki']).strip()
                elif pd.notna(row.get('title_in_britannica')) and str(row.get('title_in_britannica', '')).strip():
                    topic = str(row['title_in_britannica']).strip()
                
                if not topic:
                    self.logger.warning(f"Row {index + 1}: No valid topic name found, skipping")
                    continue
                
                sanitized_topic = sanitize_topic(topic)
                
                # Show current progress with original CSV row numbers
                current_position = len(results) + 1
                total_to_process = len(df)
                original_row_num = index + 1
                
                self.logger.info(f"Processing topic {current_position}/{total_to_process} (CSV row {original_row_num}): {topic}")
                
                result_dir = self.run_single_topic(topic=sanitized_topic)
                
                results.append({
                    'row_index': index,
                    'topic': topic,
                    'sanitized_topic': sanitized_topic,
                    'output_dir': result_dir
                })
                
            except Exception as e:
                self.logger.error(f"Failed to process topic '{topic if 'topic' in locals() else 'unknown'}' (CSV row {index + 1}): {str(e)}")
                continue
        
        self.logger.info(f"Completed processing {len(results)} topics successfully")
        
        if results:
            # Calculate article length statistics
            final_lengths = []
            initial_lengths = []
            
            for r in results:
                final_article_path = r['output_dir'] / "storm_gen_article.txt"
                initial_article_path = r['output_dir'] / "initial_article.txt"
                
                if final_article_path.exists():
                    with open(final_article_path, "r", encoding="utf-8") as f:
                        final_content = f.read()
                        final_lengths.append(count_words(final_content))
                
                if initial_article_path.exists():
                    with open(initial_article_path, "r", encoding="utf-8") as f:
                        initial_content = f.read()
                        initial_lengths.append(count_words(initial_content))
            
            # Save summary in model-specific directory
            model_dir = Path(self.output_dir) / f"{self.provider}_{self.model}"
            summary_file = model_dir / "generation_summary.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("Multi-Turn STORM Article Generation Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Model: {self.provider}_{self.model}\n")
                f.write(f"Generation Method: Multi-Turn (Initial + Section Expansion)\n")
                f.write(f"Total topics processed: {len(results)}\n")
                f.write(f"Total topics in CSV: {total_topics}\n")
                if start_topic:
                    f.write(f"Started from topic: {start_topic}\n")
                if max_topics:
                    f.write(f"Limited to: {max_topics} topics\n")
                f.write(f"Topic range processed: {processing_start} to {processing_end}\n")
                f.write(f"Success rate: {len(results)/len(df)*100:.1f}%\n\n")
                
                # Article length statistics
                if final_lengths:
                    avg_final = sum(final_lengths) / len(final_lengths)
                    min_final = min(final_lengths)
                    max_final = max(final_lengths)
                    f.write(f"Final article length statistics:\n")
                    f.write(f"  Average: {avg_final:.0f} words\n")
                    f.write(f"  Min: {min_final} words\n")
                    f.write(f"  Max: {max_final} words\n\n")
                
                if initial_lengths:
                    avg_initial = sum(initial_lengths) / len(initial_lengths)
                    min_initial = min(initial_lengths)
                    max_initial = max(initial_lengths)
                    f.write(f"Initial article length statistics:\n")
                    f.write(f"  Average: {avg_initial:.0f} words\n")
                    f.write(f"  Min: {min_initial} words\n")
                    f.write(f"  Max: {max_initial} words\n\n")
                
                if final_lengths and initial_lengths:
                    expansion_ratios = [final/initial for final, initial in zip(final_lengths, initial_lengths) if initial > 0]
                    if expansion_ratios:
                        avg_expansion = sum(expansion_ratios) / len(expansion_ratios)
                        f.write(f"Expansion statistics:\n")
                        f.write(f"  Average expansion ratio: {avg_expansion:.1f}x\n")
                        f.write(f"  Average words added: {avg_final - avg_initial:.0f} words\n\n")
                
                f.write("Individual topics processed:\n")
                for r in results:
                    f.write(f"- CSV Row {r['row_index'] + 1}: {r['topic']}\n")
            
            self.logger.info(f"Summary report saved to: {summary_file}")
        
        return results

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    load_api_key_from_toml("secrets.toml")
    logger = logging.getLogger(__name__)

    if args.provider not in ["deepseek", "groq"] and not AISUITE_AVAILABLE:
        raise ValueError(
            f"Provider '{args.provider}' requires aisuite. Please install: pip install aisuite"
        )

    # API key validation for each provider
    if args.provider == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError(
            "DEEPSEEK_API_KEY environment variable is not set. Please set it in your secrets.toml file."
        )
    elif args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. Please set it in your secrets.toml file."
        )
    elif args.provider == 'anthropic' and not os.getenv('ANTHROPIC_API_KEY'):
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. Please set it in your secrets.toml file."
        )
    elif args.provider == "groq" and not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. Please set it in your secrets.toml file."
        )

    runner = MultiTurnSTORMRunner(
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p
    )

    if not args.csv_file:
        raise ValueError("CSV file is required. Use --csv-file to specify the path to your CSV file.")
    
    try:
        results = runner.run_from_csv(
            args.csv_file, 
            max_topics=args.max_topics,
            start_topic=args.start_topic
        )
        logger.info(f"Successfully processed {len(results)} topics from CSV file.")
    except Exception as e:
        logger.exception(f"An error occurred processing CSV: {str(e)}")
        raise

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/multiturn_storm",
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["deepseek", "openai", "anthropic", "cohere", "mistral", "gemini", "groq"],
        default="deepseek",
        help="AI provider to use. Groq and DeepSeek support direct API fallback when aisuite is not available.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="Model to use. Examples: deepseek-chat, gpt-4o, claude-3-5-sonnet-20240620, command-r-plus, llama-3.1-70b-versatile (Groq), etc.",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        required=True,
        help="Path to CSV file containing topics. Requires at least one of: title_in_wiki, title_in_britannica",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=None,
        help="Maximum number of topics to process. If not specified, processes all topics in CSV (default behavior).",
    )
    parser.add_argument(
        "--start-topic",
        type=int,
        default=None,
        help="Starting topic number (1-based index). For example, --start-topic=87 will start processing from the 87th topic in the CSV file.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature to use."
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter."
    )

    args = parser.parse_args()
    main(args)