#!/usr/bin/env python3
"""Generate science journalism articles from topic-segmented transcripts."""

import json
import sys
from pathlib import Path

from src.agents.article_generation.agent import ArticleWriterAgent
from src.config import ArticleGenerationConfig, Config


def process_topic_file(
    topic_file: Path,
    agent: ArticleWriterAgent,
    article_config: ArticleGenerationConfig,
    config: Config,
    output_dir: Path,
) -> bool:
    """Process a single topic file and generate an article.

    Args:
        topic_file: Path to the topic JSON file.
        agent: Initialized ArticleWriterAgent instance.
        article_config: Article generation configuration.
        config: Main configuration object for accessing allowed styles.
        output_dir: Base output directory for generated articles.

    Returns:
        True if successful, False otherwise.
    """
    channel_name = topic_file.parent.name
    topic_slug = topic_file.stem

    try:
        # Read topic file
        with open(topic_file, encoding="utf-8") as f:
            topic_data = json.load(f)

        # Extract required fields
        source_text = topic_data.get("source_text")
        if not source_text:
            print("    ✗ Missing 'source_text' field, skipping")
            return False

        # Build source metadata
        source_metadata = topic_data.get("source_metadata", {})
        source_metadata["topic_slug"] = topic_data.get("topic_slug", topic_slug)
        source_metadata["topic_title"] = topic_data.get("topic_title", "Unknown Topic")

        # Get style mode and target length (from topic file or defaults)
        style_mode = topic_data.get("style_mode", article_config.default_style_mode)
        target_length_words = topic_data.get(
            "target_length_words",
            article_config.default_target_length_words,
        )

        # Validate style_mode against allowed styles from config
        allowed_styles = config.get_allowed_article_styles()
        if style_mode not in allowed_styles:
            print(f"    ✗ Invalid style_mode '{style_mode}'. Allowed styles: {allowed_styles}, skipping")
            return False

        # Generate article
        result = agent.generate_article(
            source_text=source_text,
            source_metadata=source_metadata,
            style_mode=style_mode,
            target_length_words=target_length_words,
        )

        if result.success:
            # Save output
            output_channel_dir = output_dir / channel_name
            output_channel_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_channel_dir / f"{topic_slug}.json"

            # Convert to dict for JSON serialization
            output_data = {
                "success": result.success,
                "article": result.article.model_dump() if result.article else None,
                "metadata": result.metadata.model_dump() if result.metadata else None,
                "error": result.error,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"    ✓ Article generated: {output_file}")
            return True

        print(f"    ✗ Generation failed: {result.error}")
        return False

    except Exception as e:
        print(f"    ✗ Unexpected error: {e}")
        return False


def main() -> int:
    """Main pipeline function.

    Returns:
        0 if all articles generated successfully, 1 if any failed.
    """
    print("=== Article Generation Pipeline ===\n")

    # Load configuration
    config_path = Path("config/config.yaml")
    print(f"Loading configuration from {config_path}")
    try:
        config = Config(config_path)
        article_config = config.get_article_generation_config()
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return 1

    print("✓ Configuration loaded")
    print(f"  Writer LLM: {article_config.writer_llm.model}")
    print(f"  Max retries: {article_config.max_retries}")
    print(f"  Default style: {article_config.default_style_mode}")
    print()

    # Initialize agent
    print("Initializing Article Writer Agent...")
    try:
        agent = ArticleWriterAgent(
            llm_config=article_config.writer_llm,
            config=config,
        )
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        return 1
    print("✓ Agent initialized\n")

    # Scan for topic files
    topics_dir = config.getDataTranscriptsTopicsDir()
    print(f"Scanning for topic files in {topics_dir}...")

    topic_files: list[Path] = [
        topic_file for channel_dir in topics_dir.iterdir() if channel_dir.is_dir() for topic_file in channel_dir.glob("*.json")
    ]

    if not topic_files:
        print(f"✗ No topic files found in {topics_dir}")
        return 1

    print(f"✓ Found {len(topic_files)} topic files\n")

    # Setup output directory
    output_dir = config.getDataOutputArticlesDir()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Process each topic file
    success_count = 0
    failure_count = 0

    for idx, topic_file in enumerate(topic_files, 1):
        channel_name = topic_file.parent.name
        topic_slug = topic_file.stem

        print(f"[{idx}/{len(topic_files)}] Processing: {channel_name}/{topic_slug}")

        success = process_topic_file(topic_file, agent, article_config, config, output_dir)
        if success:
            success_count += 1
        else:
            failure_count += 1

        print()

    # Summary
    print("=== Generation Complete ===")
    print(f"Success: {success_count}")
    print(f"Failures: {failure_count}")
    print(f"Total: {len(topic_files)}")

    return 1 if failure_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
