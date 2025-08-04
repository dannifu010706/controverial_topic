"""
Simplified STORM Wiki pipeline powered by DeepSeek models without web search.
You need to set up the following environment variables to run this script:
    - DEEPSEEK_API_KEY: DeepSeek API key
    - DEEPSEEK_API_BASE: DeepSeek API base URL (default is https://api.deepseek.com)

Output will be structured as below
args.output_dir/
    topic_name/  # topic_name will follow convention of underscore-connected topic name w/o space and slash
        direct_gen_outline.txt          # Outline directly generated with LLM's parametric knowledge
        storm_gen_article.txt           # Final article generated
        storm_gen_article_polished.txt  # Polished final article (if args.do_polish_article is True)
"""

import os
import re
import logging
from argparse import ArgumentParser

from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiLMConfigs,
)
from knowledge_storm.lm import DeepSeekModel
from knowledge_storm.utils import load_api_key
from knowledge_storm.storm_wiki.modules.outline_generation import NaiveOutlineGen
from knowledge_storm.storm_wiki.modules.storm_dataclass import StormArticle


def sanitize_topic(topic):
    """
    Sanitize the topic name for use in file names.
    Remove or replace characters that are not allowed in file names.
    """
    # Replace spaces with underscores
    topic = topic.replace(" ", "_")

    # Remove any character that isn't alphanumeric, underscore, or hyphen
    topic = re.sub(r"[^a-zA-Z0-9_-]", "", topic)

    # Ensure the topic isn't empty after sanitization
    if not topic:
        topic = "unnamed_topic"

    return topic


class SimplifiedSTORMWikiRunner:
    """
    Simplified STORM Wiki Runner that generates articles using only LLM parametric knowledge.
    """
    
    def __init__(self, engine_args, lm_configs):
        self.engine_args = engine_args
        self.lm_configs = lm_configs
        self.logger = logging.getLogger(__name__)
        
        # Initialize modules
        self.naive_outline_gen = NaiveOutlineGen()
        
    def run(
        self,
        topic: str,
        do_generate_outline: bool = True,
        do_generate_article: bool = True,
        do_polish_article: bool = True,
    ):
        """
        Generate article for the topic using only LLM parametric knowledge.
        """
        import dspy
        from knowledge_storm.utils import ArticleTextProcessing
        
        output_dir = self.engine_args.output_dir
        os.makedirs(f"{output_dir}/{topic}", exist_ok=True)
        
        self.logger.info(f"Starting article generation for topic: {topic}")
        
        # Step 1: Generate outline using LLM parametric knowledge
        if do_generate_outline:
            self.logger.info("Generating outline...")
            with dspy.settings.context(lm=self.lm_configs.outline_gen_lm):
                outline_result = self.naive_outline_gen(topic=topic)
                outline = outline_result.outline
            
            # Save outline
            with open(f"{output_dir}/{topic}/direct_gen_outline.txt", "w", encoding="utf-8") as f:
                f.write(outline)
            self.logger.info("Outline generation completed.")
        else:
            # Load existing outline
            with open(f"{output_dir}/{topic}/direct_gen_outline.txt", "r", encoding="utf-8") as f:
                outline = f.read()
            self.logger.info("Loaded existing outline.")
        
        # Step 2: Generate article
        if do_generate_article:
            self.logger.info("Generating article...")
            article = self._generate_article_from_outline(topic, outline)
            
            # Save article
            with open(f"{output_dir}/{topic}/storm_gen_article.txt", "w", encoding="utf-8") as f:
                f.write(article)
            self.logger.info("Article generation completed.")
        else:
            # Load existing article
            with open(f"{output_dir}/{topic}/storm_gen_article.txt", "r", encoding="utf-8") as f:
                article = f.read()
            self.logger.info("Loaded existing article.")
        
        # Step 3: Polish article
        if do_polish_article:
            self.logger.info("Polishing article...")
            polished_article = self._polish_article(topic, article)
            
            # Save polished article
            with open(f"{output_dir}/{topic}/storm_gen_article_polished.txt", "w", encoding="utf-8") as f:
                f.write(polished_article)
            self.logger.info("Article polishing completed.")
    
    def _generate_article_from_outline(self, topic: str, outline: str) -> str:
        """
        Generate article based on outline using LLM parametric knowledge.
        """
        import dspy
        from knowledge_storm.utils import ArticleTextProcessing
        import copy
        
        # Create StormArticle object from outline
        try:
            article_with_outline = StormArticle.from_outline_str(topic=topic, outline_str=outline)
        except:
            # If outline parsing fails, create a simple article structure
            self.logger.warning("Failed to parse outline, generating simple article.")
            return self._generate_simple_article(topic)
        
        sections_to_write = article_with_outline.get_first_level_section_names()
        article = copy.deepcopy(article_with_outline)
        
        if len(sections_to_write) == 0:
            self.logger.warning(f"No sections found in outline for {topic}. Generating simple article.")
            return self._generate_simple_article(topic)
        
        # Generate content for each section
        section_gen = SimplifiedSectionGen(engine=self.lm_configs.article_gen_lm)
        
        for section_title in sections_to_write:
            # Skip introduction and conclusion sections
            if (section_title.lower().strip() in ["introduction", "intro"] or
                section_title.lower().strip().startswith("conclusion") or
                section_title.lower().strip().startswith("summary")):
                continue
            
            self.logger.info(f"Generating section: {section_title}")
            
            # Get section outline
            try:
                section_outline_list = article_with_outline.get_outline_as_list(
                    root_section_name=section_title, add_hashtags=True
                )
                section_outline = "\n".join(section_outline_list)
            except:
                section_outline = ""
            
            # Generate section content
            section_content = section_gen(
                topic=topic,
                outline=section_outline,
                section=section_title
            ).section
            
            # Update article
            article.update_section(
                parent_section_name=topic,
                current_section_content=section_content,
                current_section_info_list=[]
            )
        
        article.post_processing()
        return article.to_string()
    
    def _generate_simple_article(self, topic: str) -> str:
        """
        Generate a simple article when outline parsing fails.
        """
        import dspy
        
        simple_article_gen = dspy.Predict(SimpleArticleGeneration)
        
        with dspy.settings.context(lm=self.lm_configs.article_gen_lm):
            result = simple_article_gen(topic=topic)
            return result.article
    
    def _polish_article(self, topic: str, article: str) -> str:
        """
        Polish the article by improving expression and adding summary.
        """
        import dspy
        
        article_polisher = dspy.Predict(PolishArticle)
        
        with dspy.settings.context(lm=self.lm_configs.article_polish_lm):
            result = article_polisher(topic=topic, article=article)
            return result.polished_article
    
    def post_run(self):
        """
        Post-processing after running the pipeline.
        """
        self.logger.info("Post-processing completed.")
    
    def summary(self):
        """
        Print summary of the generation process.
        """
        self.logger.info("Article generation pipeline completed successfully.")


class SimplifiedSectionGen:
    """
    Generate article sections using only LLM parametric knowledge.
    """
    
    def __init__(self, engine):
        import dspy
        self.write_section = dspy.Predict(WriteSectionFromKnowledge)
        self.engine = engine
    
    def __call__(self, topic: str, outline: str, section: str):
        import dspy
        from knowledge_storm.utils import ArticleTextProcessing
        
        with dspy.settings.context(lm=self.engine):
            section_content = ArticleTextProcessing.clean_up_section(
                self.write_section(
                    topic=topic,
                    outline=outline,
                    section=section
                ).output
            )
        
        class Prediction:
            def __init__(self, section):
                self.section = section
        
        return Prediction(section_content)


# DSPy signatures
class WriteSectionFromKnowledge:
    """Write a Wikipedia section based on LLM's parametric knowledge."""
    
    def __init__(self):
        pass
    
    def __call__(self, topic, outline, section):
        class Output:
            def __init__(self, output_text):
                self.output = output_text
        
        # This is a placeholder - in actual implementation, this would be handled by DSPy
        # For now, we'll create a simple implementation
        content = f"# {section}\n\nThis section about {section} in the context of {topic} would be generated based on the outline:\n{outline}"
        return Output(content)


class SimpleArticleGeneration:
    """Generate a simple Wikipedia-style article based on LLM's parametric knowledge."""
    
    def __init__(self):
        pass
    
    def __call__(self, topic):
        class Output:
            def __init__(self, article_text):
                self.article = article_text
        
        # This is a placeholder - in actual implementation, this would be handled by DSPy
        article = f"# {topic}\n\n{topic} is a notable subject that encompasses various aspects and characteristics. This article provides an overview based on available knowledge.\n\n## Overview\n\nContent about {topic} would be generated here.\n\n## Characteristics\n\nDetailed characteristics and features would be described in this section."
        return Output(article)


class PolishArticle:
    """Polish and improve a Wikipedia-style article."""
    
    def __init__(self):
        pass
    
    def __call__(self, topic, article):
        class Output:
            def __init__(self, polished_text):
                self.polished_article = polished_text
        
        # This is a placeholder - in actual implementation, this would be handled by DSPy
        polished = f"# {topic}\n\n**Summary**: This article provides a comprehensive overview of {topic}.\n\n{article}\n\n## Conclusion\n\nThis concludes the article about {topic}."
        return Output(polished)


def main(args):
    load_api_key(toml_file_path="secrets.toml")
    lm_configs = STORMWikiLMConfigs()

    logger = logging.getLogger(__name__)

    # Ensure DEEPSEEK_API_KEY is set
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError(
            "DEEPSEEK_API_KEY environment variable is not set. Please set it in your secrets.toml file."
        )

    deepseek_kwargs = {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "api_base": os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    # DeepSeek offers two main models: 'deepseek-chat' for general tasks and 'deepseek-coder' for coding tasks
    # Users can choose the appropriate model based on their needs
    outline_gen_lm = DeepSeekModel(model=args.model, max_tokens=400, **deepseek_kwargs)
    article_gen_lm = DeepSeekModel(model=args.model, max_tokens=700, **deepseek_kwargs)
    article_polish_lm = DeepSeekModel(
        model=args.model, max_tokens=4000, **deepseek_kwargs
    )

    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_thread_num=args.max_thread_num,
    )

    runner = SimplifiedSTORMWikiRunner(engine_args, lm_configs)

    topic = input("Topic: ")
    sanitized_topic = sanitize_topic(topic)

    try:
        runner.run(
            topic=sanitized_topic,
            do_generate_outline=args.do_generate_outline,
            do_generate_article=args.do_generate_article,
            do_polish_article=args.do_polish_article,
        )
        runner.post_run()
        runner.summary()
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    parser = ArgumentParser()
    # global arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/simplified_storm",
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--max-thread-num",
        type=int,
        default=1,
        help="Maximum number of threads to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["deepseek-chat", "deepseek-coder"],
        default="deepseek-chat",
        help='DeepSeek model to use. "deepseek-chat" for general tasks, "deepseek-coder" for coding tasks.',
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature to use."
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter."
    )
    # stage of the pipeline
    parser.add_argument(
        "--do-generate-outline",
        action="store_true",
        help="If True, generate an outline for the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-generate-article",
        action="store_true",
        help="If True, generate an article for the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-polish-article",
        action="store_true",
        help="If True, polish the article by adding a summarization section.",
    )

    args = parser.parse_args()
    
    # If no steps specified, default to execute all
    if not any([args.do_generate_outline, args.do_generate_article, args.do_polish_article]):
        args.do_generate_outline = True
        args.do_generate_article = True
        args.do_polish_article = True

    main(args)