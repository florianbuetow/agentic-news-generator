"""Output and run-artifact writer for article-generation orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.agents.article_generation.models import (
    ArticleGenerationResult,
    ArticleResponse,
    ArticleReviewRaw,
    ArticleReviewResult,
    ConcernMappingResult,
    EditorReport,
    Verdict,
    WriterFeedback,
)

logger = logging.getLogger(__name__)


class OutputHandler:
    """Writes canonical output and per-run artifacts."""

    def __init__(
        self,
        *,
        final_articles_dir: Path,
        run_artifacts_dir: Path,
        save_intermediate_results: bool,
    ) -> None:
        self._final_articles_dir = final_articles_dir
        self._run_artifacts_dir = run_artifacts_dir
        self._save_intermediate_results = save_intermediate_results

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

    def write_writer_draft(self, *, artifacts_dir: Path, iteration: int, article: ArticleResponse) -> None:
        """Write writer draft artifacts for an iteration."""
        if not self._save_intermediate_results:
            return

        json_path = artifacts_dir / f"iter{iteration}_writer_draft.json"
        md_path = artifacts_dir / f"iter{iteration}_writer_draft.md"
        self._write_json(path=json_path, payload=article.model_dump())
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(article.article_body)

    def write_article_review_raw(self, *, artifacts_dir: Path, iteration: int, review_raw: ArticleReviewRaw) -> None:
        """Write raw article-review markdown output."""
        if not self._save_intermediate_results:
            return

        raw_path = artifacts_dir / f"iter{iteration}_article_review_raw.md"
        with open(raw_path, "w", encoding="utf-8") as handle:
            handle.write(review_raw.markdown_bullets)

    def write_article_review_parsed(self, *, artifacts_dir: Path, iteration: int, review: ArticleReviewResult) -> None:
        """Write parsed article-review concerns."""
        if not self._save_intermediate_results:
            return

        review_path = artifacts_dir / f"iter{iteration}_article_review.json"
        self._write_json(path=review_path, payload=review.model_dump())

    def write_concern_mapping(self, *, artifacts_dir: Path, iteration: int, mapping: ConcernMappingResult) -> None:
        """Write concern mapping output."""
        if not self._save_intermediate_results:
            return

        mapping_path = artifacts_dir / f"iter{iteration}_concern_mapping.json"
        self._write_json(path=mapping_path, payload=mapping.model_dump())

    def write_verdicts(self, *, artifacts_dir: Path, iteration: int, verdicts: list[Verdict]) -> None:
        """Write specialist verdicts for an iteration."""
        if not self._save_intermediate_results:
            return

        verdicts_path = artifacts_dir / f"iter{iteration}_verdicts.json"
        payload: list[object] = [verdict.model_dump() for verdict in verdicts]
        self._write_json(path=verdicts_path, payload=payload)

    def write_feedback(self, *, artifacts_dir: Path, iteration: int, feedback: WriterFeedback) -> None:
        """Write writer feedback for an iteration."""
        if not self._save_intermediate_results:
            return

        feedback_path = artifacts_dir / f"iter{iteration}_feedback.json"
        self._write_json(path=feedback_path, payload=feedback.model_dump())

    def write_final_artifacts(self, *, artifacts_dir: Path, result: ArticleGenerationResult, editor_report: EditorReport) -> None:
        """Write final run artifacts."""
        editor_report_path = artifacts_dir / "editor_report.json"
        article_result_path = artifacts_dir / "article_result.json"

        self._write_json(path=editor_report_path, payload=editor_report.model_dump())
        self._write_json(path=article_result_path, payload=result.model_dump())

        if result.article is not None:
            article_md_path = artifacts_dir / "article.md"
            with open(article_md_path, "w", encoding="utf-8") as handle:
                handle.write(result.article.article_body)

    def write_canonical_output(self, *, channel_name: str, slug: str, result: ArticleGenerationResult) -> Path:
        """Write canonical output JSON for downstream systems."""
        output_dir = self._final_articles_dir / channel_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{slug}.json"
        self._write_json(path=output_path, payload=result.model_dump())
        logger.info("Canonical output written: %s (success=%s)", output_path, result.success)
        return output_path

    def _write_json(self, *, path: Path, payload: dict[str, object] | list[object]) -> None:
        """Write JSON payload with deterministic formatting."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
