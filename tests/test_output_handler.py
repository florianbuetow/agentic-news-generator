"""Tests for article-generation output handler."""

import json
from datetime import UTC, datetime
from pathlib import Path

from src.agents.article_generation.chief_editor.output_handler import OutputHandler
from src.agents.article_generation.models import (
    ArticleGenerationResult,
    ArticleMetadata,
    ArticleResponse,
    ArticleReviewRaw,
    ArticleReviewResult,
    Concern,
    ConcernMapping,
    ConcernMappingResult,
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


def _build_verdict() -> Verdict:
    return Verdict(
        concern_id=1,
        misleading=True,
        status="REWRITE",
        rationale="rationale",
        suggested_fix="fix",
        evidence=None,
        citations=None,
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
    verdict = _build_verdict()
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


def test_output_handler_writes_artifacts_and_canonical_output(tmp_path: Path) -> None:
    """Output handler writes required artifacts and canonical JSON output."""
    output_handler = OutputHandler(
        final_articles_dir=tmp_path / "final",
        run_artifacts_dir=tmp_path / "runs",
        save_intermediate_results=True,
    )

    artifacts_dir = output_handler.initialize_run_artifacts_dir(
        channel_name="channel",
        slug="topic",
        source_file="source.json",
        style_mode="SCIAM_MAGAZINE",
    )
    assert artifacts_dir.exists()

    article = _build_article()
    output_handler.write_writer_draft(artifacts_dir=artifacts_dir, iteration=1, article=article)

    review_raw = ArticleReviewRaw(markdown_bullets="- bullet")
    review_result = ArticleReviewResult(concerns=[Concern(concern_id=1, excerpt="a", review_note="b")])
    mapping_result = ConcernMappingResult(
        mappings=[
            ConcernMapping(
                concern_id=1,
                concern_type="unsupported_fact",
                selected_agent="opinion",
                confidence="high",
                reason="reason",
            )
        ]
    )
    verdict = _build_verdict()
    feedback = WriterFeedback(
        iteration=1,
        rating=2,
        passed=False,
        reasoning="reason",
        improvement_suggestions=[],
        todo_list=["fix"],
        verdicts=[verdict],
    )

    output_handler.write_article_review_raw(artifacts_dir=artifacts_dir, iteration=1, review_raw=review_raw)
    output_handler.write_article_review_parsed(artifacts_dir=artifacts_dir, iteration=1, review=review_result)
    output_handler.write_concern_mapping(artifacts_dir=artifacts_dir, iteration=1, mapping=mapping_result)
    output_handler.write_verdicts(artifacts_dir=artifacts_dir, iteration=1, verdicts=[verdict])
    output_handler.write_feedback(artifacts_dir=artifacts_dir, iteration=1, feedback=feedback)

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
        artifacts_dir=str(artifacts_dir),
        error=None,
    )

    output_handler.write_final_artifacts(artifacts_dir=artifacts_dir, result=result, editor_report=editor_report)
    canonical_path = output_handler.write_canonical_output(channel_name="channel", slug="topic", result=result)

    assert (artifacts_dir / "iter1_writer_draft.json").exists()
    assert (artifacts_dir / "iter1_writer_draft.md").exists()
    assert (artifacts_dir / "iter1_article_review_raw.md").exists()
    assert (artifacts_dir / "iter1_article_review.json").exists()
    assert (artifacts_dir / "iter1_concern_mapping.json").exists()
    assert (artifacts_dir / "iter1_verdicts.json").exists()
    assert (artifacts_dir / "iter1_feedback.json").exists()
    assert (artifacts_dir / "editor_report.json").exists()
    assert (artifacts_dir / "article_result.json").exists()
    assert (artifacts_dir / "article.md").exists()
    assert canonical_path.exists()

    with open(canonical_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["success"] is True
