"""Output and run-artifact writer for article-generation orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.agents.article_generation.models import ArticleGenerationResult

logger = logging.getLogger(__name__)


class OutputHandler:
    """Writes canonical output and initializes per-run artifact directories."""

    def __init__(
        self,
        *,
        final_articles_dir: Path,
        run_artifacts_dir: Path,
    ) -> None:
        self._final_articles_dir = final_articles_dir
        self._run_artifacts_dir = run_artifacts_dir

    def initialize_run_artifacts_dir(
        self,
        *,
        channel_name: str,
        slug: str,
        source_file: str,
        style_mode: str,
    ) -> Path:
        """Create run artifacts directory and return its path."""
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        digest_source = source_file + slug + style_mode
        suffix = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:8]
        run_id = f"{timestamp}_{suffix}"
        artifacts_dir = self._run_artifacts_dir / channel_name / slug / run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized run artifacts dir: %s", artifacts_dir)
        return artifacts_dir

    def write_canonical_output(self, *, channel_name: str, slug: str, result: ArticleGenerationResult) -> Path:
        """Write canonical output JSON for downstream systems."""
        output_dir = self._final_articles_dir / channel_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{slug}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result.model_dump(), handle, indent=2, ensure_ascii=False)
        logger.info("Canonical output written: %s (success=%s)", output_path, result.success)
        return output_path
