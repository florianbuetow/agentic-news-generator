#!/usr/bin/env python3
"""Generate reviewed articles from article input bundles."""

import logging
import sys
import time
from pathlib import Path

from src.agents.article_generation.agent import build_chief_editor_orchestrator
from src.agents.article_generation.bundle_loader import bundle_to_source_metadata, load_bundle
from src.agents.article_generation.chief_editor.orchestrator import ChiefEditorOrchestrator
from src.agents.article_generation.llm_client import LiteLLMClient
from src.config import Config

logger = logging.getLogger(__name__)


def process_bundle(
    *,
    bundle_dir: Path,
    orchestrator: ChiefEditorOrchestrator,
    config: Config,
) -> bool:
    """Process a single article input bundle."""
    slug = bundle_dir.name

    try:
        started_at = time.perf_counter()
        logger.info("Loading bundle: %s", bundle_dir)
        bundle = load_bundle(bundle_dir)
        source_metadata = bundle_to_source_metadata(bundle)

        article_config = config.get_article_generation_config()
        style_mode = article_config.default_style_mode

        allowed_styles = config.get_allowed_article_styles()
        if style_mode not in allowed_styles:
            raise ValueError(f"Invalid style_mode '{style_mode}'. Allowed styles: {allowed_styles}")

        logger.info(
            "Contract: slug=%s style_mode=%s source_chars=%d",
            slug,
            style_mode,
            len(bundle.source_text),
        )

        result = orchestrator.generate_article(
            source_text=bundle.source_text,
            source_metadata=source_metadata,
            style_mode=style_mode,
            reader_preference="",
        )

        elapsed_seconds = time.perf_counter() - started_at
        if result.success:
            logger.info("Article generated: %s (%.1fs)", slug, elapsed_seconds)
            return True

        logger.error("Generation failed: %s — %s (%.1fs)", slug, result.error, elapsed_seconds)
        return False

    except Exception as exc:
        logger.exception("Unexpected error processing bundle %s: %s", slug, exc)
        return False


def _run_preflight_check(*, config: Config) -> None:
    """Verify LLM API connectivity before starting the pipeline.

    Raises:
        ConnectionError: If the LLM API endpoint is unreachable.
        ValueError: If no api_base is configured.
    """
    article_config = config.get_article_generation_config()
    api_base = article_config.agents.writer_llm.api_base
    if api_base is None:
        raise ValueError("Writer LLM api_base is not configured — cannot run pre-flight check")
    llm_client = LiteLLMClient()
    llm_client.check_connectivity(api_base=api_base, timeout_seconds=10)


def main() -> int:
    """Run article generation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== Article Generation Pipeline ===")

    config_path = Path("config/config.yaml")
    logger.info("Loading configuration from %s", config_path)
    try:
        config = Config(config_path)
        article_config = config.get_article_generation_config()
    except Exception as exc:
        logger.error("Failed to load configuration: %s", exc)
        return 1

    logger.info(
        "Configuration loaded: writer_model=%s editor_max_rounds=%d default_style=%s",
        article_config.agents.writer_llm.model,
        article_config.editor.editor_max_rounds,
        article_config.default_style_mode,
    )

    logger.info("Running pre-flight connectivity check...")
    try:
        _run_preflight_check(config=config)
    except ConnectionError as exc:
        logger.error("Pre-flight check FAILED: %s", exc)
        return 1
    logger.info("Pre-flight check passed")

    logger.info("Initializing Chief Editor Orchestrator...")
    try:
        orchestrator = build_chief_editor_orchestrator(config=config)
    except Exception as exc:
        logger.error("Failed to initialize orchestrator: %s", exc)
        return 1
    logger.info("Orchestrator initialized")

    articles_input_dir = config.getDataArticlesInputDir()
    logger.info("Scanning for article bundles in %s", articles_input_dir)

    if not articles_input_dir.exists():
        logger.error("Articles input directory does not exist: %s", articles_input_dir)
        return 1

    bundle_dirs = sorted(
        bundle_dir for bundle_dir in articles_input_dir.iterdir() if bundle_dir.is_dir() and not bundle_dir.name.startswith(".")
    )

    if len(bundle_dirs) == 0:
        logger.error("No article bundles found in %s", articles_input_dir)
        return 1

    logger.info("Found %d article bundles", len(bundle_dirs))

    success_count = 0
    failure_count = 0

    for idx, bundle_dir in enumerate(bundle_dirs, start=1):
        logger.info("[%d/%d] Processing bundle: %s", idx, len(bundle_dirs), bundle_dir.name)
        success = process_bundle(
            bundle_dir=bundle_dir,
            orchestrator=orchestrator,
            config=config,
        )
        if success:
            success_count += 1
        else:
            failure_count += 1

    logger.info(
        "=== Generation Complete === success=%d failures=%d total=%d",
        success_count,
        failure_count,
        len(bundle_dirs),
    )

    return 1 if failure_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
