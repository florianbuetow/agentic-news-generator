"""Chief editor orchestrator for multi-agent article generation."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from src.agents.article_generation.base import (
    ArticleReviewAgentProtocol,
    ConcernMappingAgentProtocol,
    SpecialistAgentProtocol,
    WriterAgentProtocol,
)
from src.agents.article_generation.chief_editor.bullet_parser import ArticleReviewBulletParser
from src.agents.article_generation.chief_editor.institutional_memory import InstitutionalMemoryStore
from src.agents.article_generation.chief_editor.output_handler import OutputHandler
from src.agents.article_generation.chief_editor.verbose_context_logger import VerboseContextLogger
from src.agents.article_generation.models import (
    AgentResult,
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
from src.config import Config

logger = logging.getLogger(__name__)


class ChiefEditorOrchestrator:
    """Orchestrates draft generation, editorial review, and revisions."""

    def __init__(
        self,
        *,
        config: Config,
        writer_agent: WriterAgentProtocol,
        article_review_agent: ArticleReviewAgentProtocol,
        concern_mapping_agent: ConcernMappingAgentProtocol,
        fact_check_agent: SpecialistAgentProtocol,
        evidence_finding_agent: SpecialistAgentProtocol,
        opinion_agent: SpecialistAgentProtocol,
        attribution_agent: SpecialistAgentProtocol,
        style_review_agent: SpecialistAgentProtocol,
        bullet_parser: ArticleReviewBulletParser,
        institutional_memory: InstitutionalMemoryStore,
        output_handler: OutputHandler,
    ) -> None:
        self._config = config
        self._writer_agent = writer_agent
        self._article_review_agent = article_review_agent
        self._concern_mapping_agent = concern_mapping_agent
        self._fact_check_agent = fact_check_agent
        self._evidence_finding_agent = evidence_finding_agent
        self._opinion_agent = opinion_agent
        self._attribution_agent = attribution_agent
        self._style_review_agent = style_review_agent
        self._bullet_parser = bullet_parser
        self._institutional_memory = institutional_memory
        self._output_handler = output_handler

    def generate_article(
        self,
        *,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_mode: str,
        reader_preference: str,
    ) -> ArticleGenerationResult:
        """Run full orchestration loop and return canonical result."""
        channel_name = self._require_metadata_value(source_metadata, "channel_name")
        slug = self._require_metadata_value(source_metadata, "slug")
        source_file = self._require_metadata_value(source_metadata, "source_file")
        run_label = f"{channel_name}/{slug}"

        artifacts_dir = self._output_handler.initialize_run_artifacts_dir(
            channel_name=channel_name,
            slug=slug,
            source_file=source_file,
            style_mode=style_mode,
        )

        ctx = VerboseContextLogger(artifacts_dir=artifacts_dir)

        # Log source inputs
        ctx.log(agent_name="source", step="context", content=source_text, fmt="txt")
        ctx.log(agent_name="source", step="metadata", content=dict(source_metadata), fmt="json")
        ctx.log(agent_name="source", step="run_params", content={"style_mode": style_mode, "reader_preference": reader_preference}, fmt="json")

        iteration_reports: list[IterationReport] = []

        self._progress(run_label=run_label, message="Generating initial writer draft")
        writer_result = self._writer_agent.generate(
            source_text=source_text,
            source_metadata=source_metadata,
            style_mode=style_mode,
            reader_preference=reader_preference,
        )
        ctx.log(agent_name="writer", step="prompt", content=writer_result.prompt, fmt="md")
        ctx.log(agent_name="writer", step="output", content=writer_result.output.model_dump(), fmt="json")
        ctx.log(agent_name="writer", step="output", content=writer_result.output.article_body, fmt="md")
        current_article = writer_result.output
        self._progress(run_label=run_label, message=f"Initial draft stored in {artifacts_dir}")

        editor_max_rounds = self._config.get_article_editor_max_rounds()
        self._progress(run_label=run_label, message=f"Starting editor loop with max rounds={editor_max_rounds}")

        for iteration in range(1, editor_max_rounds + 1):
            self._progress(run_label=run_label, message=f"Round {iteration}: Article review")
            review_result = self._article_review_agent.review(
                article=current_article,
                source_text=source_text,
                source_metadata=source_metadata,
            )
            ctx.log(agent_name="article_review", step="prompt", content=review_result.prompt, fmt="md")
            ctx.log(agent_name="article_review", step="output", content=review_result.output.markdown_bullets, fmt="md")
            review_raw = review_result.output

            parsed_review = self._bullet_parser.parse(markdown_bullets=review_raw.markdown_bullets)
            ctx.log(agent_name="article_review", step="parsed", content=parsed_review.model_dump(), fmt="json")
            self._progress(
                run_label=run_label,
                message=f"Round {iteration}: Parsed {len(parsed_review.concerns)} concerns",
            )

            if len(parsed_review.concerns) == 0:
                editor_report = EditorReport(
                    iterations=iteration_reports,
                    total_iterations=iteration,
                    final_status="SUCCESS",
                    blocking_concerns=None,
                )
                result = self._build_success_result(
                    article=current_article,
                    source_metadata=source_metadata,
                    style_mode=style_mode,
                    editor_report=editor_report,
                    artifacts_dir=artifacts_dir,
                )
                self._log_final_artifacts(ctx=ctx, result=result, editor_report=editor_report)
                self._output_handler.write_canonical_output(channel_name=channel_name, slug=slug, result=result)
                self._progress(run_label=run_label, message=f"Completed successfully in round {iteration}")
                return result

            self._progress(run_label=run_label, message=f"Round {iteration}: Concern mapping")
            mapping_agent_result = self._concern_mapping_agent.map_concerns(
                style_requirements=style_mode,
                source_text=source_text,
                generated_article_json=current_article.model_dump_json(),
                concerns=parsed_review.concerns,
            )
            ctx.log(agent_name="concern_mapping", step="prompt", content=mapping_agent_result.prompt, fmt="md")
            ctx.log(agent_name="concern_mapping", step="output", content=mapping_agent_result.output.model_dump(), fmt="json")
            mapping_result = mapping_agent_result.output
            self._progress(
                run_label=run_label,
                message=f"Round {iteration}: Evaluating {len(mapping_result.mappings)} mapped concerns",
            )

            verdicts: list[Verdict] = []
            for mapping in mapping_result.mappings:
                self._progress(
                    run_label=run_label,
                    message=f"Round {iteration}: Specialist evaluation for concern #{mapping.concern_id} ({mapping.selected_agent})",
                )
                concern = self._find_concern(concerns=parsed_review.concerns, concern_id=mapping.concern_id)
                specialist_result = self._evaluate_mapping(
                    mapping=mapping,
                    concern=concern,
                    article=current_article,
                    source_text=source_text,
                    source_metadata=source_metadata,
                    style_requirements=style_mode,
                )
                ctx.log(agent_name=mapping.selected_agent, step="prompt", content=specialist_result.prompt, fmt="md")
                ctx.log(agent_name=mapping.selected_agent, step="output", content=specialist_result.output.model_dump(), fmt="json")
                verdicts.append(specialist_result.output)

            is_passed = all(not verdict.misleading for verdict in verdicts)
            if is_passed:
                self._progress(run_label=run_label, message=f"Round {iteration}: All specialist verdicts passed")
                iteration_reports.append(
                    IterationReport(
                        iteration_number=iteration,
                        concerns=parsed_review.concerns,
                        mappings=mapping_result.mappings,
                        verdicts=verdicts,
                        feedback_to_writer=None,
                        article_draft=current_article,
                    )
                )
                editor_report = EditorReport(
                    iterations=iteration_reports,
                    total_iterations=iteration,
                    final_status="SUCCESS",
                    blocking_concerns=None,
                )
                result = self._build_success_result(
                    article=current_article,
                    source_metadata=source_metadata,
                    style_mode=style_mode,
                    editor_report=editor_report,
                    artifacts_dir=artifacts_dir,
                )
                self._log_final_artifacts(ctx=ctx, result=result, editor_report=editor_report)
                self._output_handler.write_canonical_output(channel_name=channel_name, slug=slug, result=result)
                self._progress(run_label=run_label, message=f"Completed successfully in round {iteration}")
                return result

            if iteration == editor_max_rounds:
                blocking_concerns = self._blocking_concerns(concerns=parsed_review.concerns, verdicts=verdicts)
                iteration_reports.append(
                    IterationReport(
                        iteration_number=iteration,
                        concerns=parsed_review.concerns,
                        mappings=mapping_result.mappings,
                        verdicts=verdicts,
                        feedback_to_writer=None,
                        article_draft=current_article,
                    )
                )
                editor_report = EditorReport(
                    iterations=iteration_reports,
                    total_iterations=iteration,
                    final_status="FAILED",
                    blocking_concerns=blocking_concerns,
                )
                result = ArticleGenerationResult(
                    success=False,
                    article=current_article,
                    metadata=self._build_metadata(
                        source_metadata=source_metadata,
                        style_mode=style_mode,
                    ),
                    editor_report=editor_report,
                    artifacts_dir=str(artifacts_dir),
                    error="Unresolved misleading concerns after editor_max_rounds",
                )
                self._log_final_artifacts(ctx=ctx, result=result, editor_report=editor_report)
                self._output_handler.write_canonical_output(channel_name=channel_name, slug=slug, result=result)
                self._progress(run_label=run_label, message=f"Failed after {editor_max_rounds} rounds with unresolved concerns")
                return result

            self._progress(run_label=run_label, message=f"Round {iteration}: Compiling writer feedback")
            feedback = self._compile_feedback(iteration=iteration, verdicts=verdicts)
            ctx.log(agent_name="feedback", step="output", content=feedback.model_dump(), fmt="json")

            iteration_reports.append(
                IterationReport(
                    iteration_number=iteration,
                    concerns=parsed_review.concerns,
                    mappings=mapping_result.mappings,
                    verdicts=verdicts,
                    feedback_to_writer=feedback,
                    article_draft=current_article,
                )
            )

            revision_context = self._build_revision_context(
                source_text=source_text,
                source_metadata=source_metadata,
                style_mode=style_mode,
                reader_preference=reader_preference,
                current_article=current_article,
            )
            self._progress(run_label=run_label, message=f"Round {iteration}: Requesting writer revision")
            revision_result = self._writer_agent.revise(context=revision_context, feedback=feedback)
            ctx.log(agent_name="writer", step="prompt", content=revision_result.prompt, fmt="md")
            ctx.log(agent_name="writer", step="output", content=revision_result.output.model_dump(), fmt="json")
            ctx.log(agent_name="writer", step="output", content=revision_result.output.article_body, fmt="md")
            current_article = revision_result.output
            self._progress(run_label=run_label, message=f"Round {iteration}: Revision draft stored")

        raise RuntimeError("Orchestration loop exited without returning a result")

    def _log_final_artifacts(
        self,
        *,
        ctx: VerboseContextLogger,
        result: ArticleGenerationResult,
        editor_report: EditorReport,
    ) -> None:
        """Write final artifacts using the verbose context logger."""
        if result.article is not None:
            ctx.log_final(agent_name="final", step="article", content=result.article.article_body, fmt="md")
        ctx.log_final(agent_name="final", step="article_result", content=result.model_dump(), fmt="json")
        ctx.log_final(agent_name="final", step="editor_report", content=editor_report.model_dump(), fmt="json")

    def _progress(self, *, run_label: str, message: str) -> None:
        """Emit orchestration progress updates."""
        logger.info("[%s] %s", run_label, message)

    def _evaluate_mapping(
        self,
        *,
        mapping: ConcernMapping,
        concern: Concern,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_requirements: str,
    ) -> AgentResult[Verdict]:
        """Route a concern to the selected specialist and return its full AgentResult."""
        agents = {
            "fact_check": self._fact_check_agent,
            "evidence_finding": self._evidence_finding_agent,
            "opinion": self._opinion_agent,
            "attribution": self._attribution_agent,
            "style_review": self._style_review_agent,
        }
        agent = agents.get(mapping.selected_agent)
        if agent is None:
            raise ValueError(f"Unknown specialist agent: {mapping.selected_agent}")
        return agent.evaluate(
            concern=concern,
            article=article,
            source_text=source_text,
            source_metadata=source_metadata,
            style_requirements=style_requirements,
        )

    def _compile_feedback(self, *, iteration: int, verdicts: list[Verdict]) -> WriterFeedback:
        """Compile deterministic feedback without additional model calls."""
        sorted_verdicts = sorted(verdicts, key=lambda verdict: verdict.concern_id)

        required_todos = [
            verdict.suggested_fix
            for verdict in sorted_verdicts
            if verdict.status in {"REWRITE", "REMOVE"} and verdict.suggested_fix is not None
        ]

        improvement_suggestions: list[str] = []
        for verdict in sorted_verdicts:
            if verdict.status == "KEEP" and len(improvement_suggestions) < 5:
                improvement_suggestions.append(verdict.rationale)

        rating_raw = 10 - 2 * len(required_todos) - len(improvement_suggestions)
        rating = max(1, min(10, rating_raw))

        if len(required_todos) == 0:
            reasoning = "No required rewrites or removals were identified."
        else:
            todo_lines = [f"{index}. {todo}" for index, todo in enumerate(required_todos, start=1)]
            reasoning = "Required changes:\n" + "\n".join(todo_lines)

        return WriterFeedback(
            iteration=iteration,
            rating=rating,
            passed=False,
            reasoning=reasoning,
            improvement_suggestions=improvement_suggestions,
            todo_list=required_todos,
            verdicts=sorted_verdicts,
        )

    def _build_revision_context(
        self,
        *,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_mode: str,
        reader_preference: str,
        current_article: ArticleResponse,
    ) -> str:
        """Build revision context payload for writer retries."""
        payload = {
            "style_mode": style_mode,
            "reader_preference": reader_preference,
            "source_text": source_text,
            "source_metadata": source_metadata,
            "current_article": current_article.model_dump(),
        }
        return json.dumps(payload, ensure_ascii=False)

    def _build_success_result(
        self,
        *,
        article: ArticleResponse,
        source_metadata: dict[str, str | None],
        style_mode: str,
        editor_report: EditorReport,
        artifacts_dir: Path,
    ) -> ArticleGenerationResult:
        """Build successful generation result."""
        metadata = self._build_metadata(
            source_metadata=source_metadata,
            style_mode=style_mode,
        )
        return ArticleGenerationResult(
            success=True,
            article=article,
            metadata=metadata,
            editor_report=editor_report,
            artifacts_dir=str(artifacts_dir),
            error=None,
        )

    def _build_metadata(
        self,
        *,
        source_metadata: dict[str, str | None],
        style_mode: str,
    ) -> ArticleMetadata:
        """Build article metadata from source context."""
        publish_date = source_metadata.get("publish_date")
        references_raw = source_metadata.get("references")
        references: list[dict[str, str]] = json.loads(references_raw) if references_raw is not None else []
        return ArticleMetadata(
            source_file=self._require_metadata_value(source_metadata, "source_file"),
            channel_name=self._require_metadata_value(source_metadata, "channel_name"),
            video_id=self._require_metadata_value(source_metadata, "video_id"),
            article_title=self._require_metadata_value(source_metadata, "article_title"),
            slug=self._require_metadata_value(source_metadata, "slug"),
            publish_date=publish_date,
            references=references,
            style_mode=cast(Literal["NATURE_NEWS", "SCIAM_MAGAZINE"], style_mode),
            generated_at=datetime.now(UTC).isoformat(),
        )

    def _require_metadata_value(self, metadata: dict[str, str | None], key: str) -> str:
        """Require a non-null metadata field."""
        value = metadata.get(key)
        if value is None:
            raise ValueError(f"Missing required metadata field: {key}")
        return value

    def _find_concern(self, *, concerns: list[Concern], concern_id: int) -> Concern:
        """Find concern by id in parsed concern list."""
        for concern in concerns:
            if concern.concern_id == concern_id:
                return concern
        raise ValueError(f"Concern id from mapping not found: {concern_id}")

    def _blocking_concerns(self, *, concerns: list[Concern], verdicts: list[Verdict]) -> list[Concern]:
        """Return concerns still marked as misleading."""
        misleading_ids = {verdict.concern_id for verdict in verdicts if verdict.misleading}
        return [concern for concern in concerns if concern.concern_id in misleading_ids]
