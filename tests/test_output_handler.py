"""Tests for article-generation output handler."""

import json
from datetime import UTC, datetime
from pathlib import Path

from src.agents.article_generation.chief_editor.output_handler import OutputHandler
from src.agents.article_generation.models import (
    ArticleGenerationResult,
    ArticleMetadata,
    ArticleResponse,
    Concern,
    ConcernMapping,
    EditorReport,
    IterationReport,
    Verdict,
    WriterFeedback,
)


def _build_article() -> ArticleResponse:
    return ArticleResponse(
        headline="headline",
        alternative_headline="alt",
        article_body="body",
        description="desc",
    )


def _build_editor_report(article: ArticleResponse) -> EditorReport:
    concern = Concern(concern_id=1, excerpt="x", review_note="note")
    mapping = ConcernMapping(
        concern_id=1,
        concern_type="unsupported_fact",
        selected_agent="opinion",
        confidence="high",
        reason="reason",
    )
    verdict = Verdict(
        concern_id=1,
        misleading=True,
        status="REWRITE",
        rationale="rationale",
        suggested_fix="fix",
        evidence=None,
        citations=None,
    )
    feedback = WriterFeedback(
        iteration=1,
        rating=4,
        passed=False,
        reasoning="needs changes",
        improvement_suggestions=[],
        todo_list=["fix"],
        verdicts=[verdict],
    )
    iteration = IterationReport(
        iteration_number=1,
        concerns=[concern],
        mappings=[mapping],
        verdicts=[verdict],
        feedback_to_writer=feedback,
        article_draft=article,
    )
    return EditorReport(
        iterations=[iteration],
        total_iterations=1,
        final_status="SUCCESS",
        blocking_concerns=None,
    )


def test_initialize_run_artifacts_dir_creates_directory(tmp_path: Path) -> None:
    """initialize_run_artifacts_dir creates the directory and returns its path."""
    output_handler = OutputHandler(
        final_articles_dir=tmp_path / "final",
        run_artifacts_dir=tmp_path / "runs",
    )

    artifacts_dir = output_handler.initialize_run_artifacts_dir(
        channel_name="channel",
        slug="topic",
        source_file="source.json",
        style_mode="SCIAM_MAGAZINE",
    )
    assert artifacts_dir.exists()
    assert "channel" in str(artifacts_dir)
    assert "topic" in str(artifacts_dir)


def test_write_canonical_output_writes_json(tmp_path: Path) -> None:
    """write_canonical_output writes valid JSON with result data."""
    output_handler = OutputHandler(
        final_articles_dir=tmp_path / "final",
        run_artifacts_dir=tmp_path / "runs",
    )

    article = _build_article()
    metadata = ArticleMetadata(
        source_file="source.json",
        channel_name="channel",
        video_id="video",
        article_title="title",
        slug="topic",
        publish_date=None,
        references=[],
        style_mode="SCIAM_MAGAZINE",
        generated_at=datetime.now(UTC).isoformat(),
    )
    editor_report = _build_editor_report(article)
    result = ArticleGenerationResult(
        success=True,
        article=article,
        metadata=metadata,
        editor_report=editor_report,
        artifacts_dir=str(tmp_path / "runs"),
        error=None,
    )

    canonical_path = output_handler.write_canonical_output(channel_name="channel", slug="topic", result=result)
    assert canonical_path.exists()

    with open(canonical_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["success"] is True
    assert payload["article"]["headline"] == "headline"
