"""Integration test for the real article-generation orchestrator flow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.agents.article_generation.article_review.agent import ArticleReviewAgent
from src.agents.article_generation.bundle_loader import bundle_to_source_metadata, load_bundle
from src.agents.article_generation.chief_editor.bullet_parser import ArticleReviewBulletParser
from src.agents.article_generation.chief_editor.institutional_memory import InstitutionalMemoryStore
from src.agents.article_generation.chief_editor.orchestrator import ChiefEditorOrchestrator
from src.agents.article_generation.chief_editor.output_handler import OutputHandler
from src.agents.article_generation.concern_mapping.agent import ConcernMappingAgent
from src.agents.article_generation.models import ArticleGenerationResult
from src.agents.article_generation.prompts.loader import PromptLoader
from src.agents.article_generation.specialists.attribution.agent import AttributionAgent
from src.agents.article_generation.specialists.evidence_finding.agent import EvidenceFindingAgent
from src.agents.article_generation.specialists.fact_check.agent import FactCheckAgent
from src.agents.article_generation.specialists.opinion.agent import OpinionAgent
from src.agents.article_generation.specialists.style_review.agent import StyleReviewAgent
from src.agents.article_generation.writer.agent import WriterAgent
from src.config import Config, LLMConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_BUNDLE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "article_generation" / "sample_bundle"

pytestmark = pytest.mark.e2e

_UNSUPPORTED_LINE = "That shift could let smaller labs compete head-to-head with the biggest tech companies."
_ALTERNATIVE_HEADLINE = "A deployment-focused case for using lighter models when cost and latency matter"
_DESCRIPTION = "A source-backed argument for choosing smaller AI models when deployment constraints matter more than maximum breadth."
_REWRITE_FIX = (
    "Remove the sentence about smaller labs competing head-to-head with the biggest companies, "
    "or narrow it to the source's practical workflow claim."
)
_INITIAL_ARTICLE_BODY = (
    "Smaller language models can be enough for targeted internal tools when teams care about latency "
    "and hardware cost. "
    "In the source discussion, the speaker argues that a seven-billion-parameter model can often run "
    "on a single GPU and respond faster than a much larger system.\\n\\n"
    "The speaker frames that as a practical tradeoff rather than a blanket rule. Larger models still "
    "perform better on some broad reasoning tasks, so teams are urged to benchmark their own workflows "
    "instead of assuming the biggest model always wins.\\n\\n"
    f"{_UNSUPPORTED_LINE}"
)
_REVISED_ARTICLE_BODY = (
    "Smaller language models can be enough for targeted internal tools when teams care about latency "
    "and hardware cost. "
    "In the source discussion, the speaker argues that a seven-billion-parameter model can often run "
    "on a single GPU and respond faster than a much larger system.\\n\\n"
    "The speaker frames that as a practical tradeoff rather than a blanket rule. Larger models still "
    "perform better on some broad reasoning tasks, so teams are urged to benchmark their own workflows "
    "instead of assuming the biggest model always wins.\\n\\n"
    "That framing stays narrow: the source says smaller models can be sufficient for targeted workflows "
    "when latency and hardware budgets matter."
)


class _ScriptedLLMClient:
    """Deterministic fake LLM client for exercising the real agent pipeline."""

    def __init__(self) -> None:
        self.routes: list[str] = []

    def complete(self, *, llm_config: LLMConfig, messages: list[dict[str, str]]) -> str:
        prompt = messages[0]["content"]

        if "<persona>" in prompt and "SOURCE_TEXT:" in prompt and "SOURCE_METADATA:" in prompt:
            self.routes.append("writer_generate")
            return json.dumps(
                {
                    "headline": "Smaller Models Can Be Enough for Focused AI Workloads",
                    "alternative_headline": _ALTERNATIVE_HEADLINE,
                    "article_body": _INITIAL_ARTICLE_BODY,
                    "description": _DESCRIPTION,
                }
            )

        if prompt.startswith("Review the generated article against the source text"):
            self.routes.append("article_review")
            if _UNSUPPORTED_LINE in prompt:
                return (
                    '- "That shift could let smaller labs compete head-to-head with the biggest tech companies." '
                    "— the source only says smaller models can be enough for targeted workflows with tighter latency and hardware budgets, "
                    "not that smaller labs will compete head-to-head with the biggest companies."
                )
            return ""

        if prompt.startswith("You are the Concern-Mapping Agent."):
            self.routes.append("concern_mapping")
            return json.dumps(
                [
                    {
                        "concern_id": 1,
                        "concern_type": "scope_expansion",
                        "selected_agent": "opinion",
                        "confidence": "high",
                        "reason": "The concern is an interpretive leap about competitive impact that is not stated in the source.",
                    }
                ]
            )

        if prompt.startswith("You are the Opinion Agent."):
            self.routes.append("opinion")
            return json.dumps(
                {
                    "concern_id": 1,
                    "misleading": True,
                    "status": "REWRITE",
                    "rationale": (
                        "The article broadens the source from a narrow deployment claim into a sweeping "
                        "competitive conclusion about smaller labs."
                    ),
                    "suggested_fix": _REWRITE_FIX,
                    "evidence": "The source limits the claim to targeted workflows where latency and hardware budgets matter.",
                    "citations": None,
                }
            )

        if prompt.startswith("You are revising your previous article after editorial review."):
            self.routes.append("writer_revise")
            return json.dumps(
                {
                    "headline": "Smaller Models Can Be Enough for Focused AI Workloads",
                    "alternative_headline": _ALTERNATIVE_HEADLINE,
                    "article_body": _REVISED_ARTICLE_BODY,
                    "description": _DESCRIPTION,
                }
            )

        raise AssertionError(f"Unexpected prompt route. First 120 chars: {prompt[:120]!r}")


class _UnusedKnowledgeBaseRetriever:
    def search(self, *, query: str, top_k: int, timeout_seconds: int) -> list[dict[str, str]]:
        return []


class _UnusedPerplexityClient:
    def search(self, *, query: str, model: str, timeout_seconds: int) -> dict[str, object]:
        return {"citations": [], "choices": []}


def _write_test_config(tmp_path: Path) -> Config:
    llm = {
        "model": "test-model",
        "api_base": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
        "context_window": 32768,
        "max_tokens": 2048,
        "temperature": 0.3,
        "context_window_threshold": 95,
        "max_retries": 0,
        "retry_delay": 0.1,
        "timeout_seconds": 30,
    }

    prompt_root = PROJECT_ROOT / "prompts" / "article_editor"

    config_data: dict[str, object] = {
        "paths": {
            "data_dir": str(tmp_path),
            "data_models_dir": str(tmp_path / "models"),
            "data_downloads_dir": str(tmp_path / "downloads"),
            "data_downloads_videos_dir": str(tmp_path / "downloads" / "videos"),
            "data_downloads_transcripts_dir": str(tmp_path / "downloads" / "transcripts"),
            "data_downloads_transcripts_hallucinations_dir": str(tmp_path / "downloads" / "transcripts-hallucinations"),
            "data_downloads_transcripts_cleaned_dir": str(tmp_path / "downloads" / "transcripts_cleaned"),
            "data_transcripts_topics_dir": str(tmp_path / "downloads" / "transcripts-topics"),
            "data_downloads_audio_dir": str(tmp_path / "downloads" / "audio"),
            "data_downloads_metadata_dir": str(tmp_path / "downloads" / "metadata"),
            "data_output_dir": str(tmp_path / "output"),
            "data_input_dir": str(tmp_path / "input"),
            "data_temp_dir": str(tmp_path / "temp"),
            "data_archive_dir": str(tmp_path / "archive"),
            "data_archive_videos_dir": str(tmp_path / "archive" / "videos"),
            "data_logs_dir": str(tmp_path / "logs"),
            "data_output_articles_dir": str(tmp_path / "output" / "articles"),
            "data_articles_input_dir": str(tmp_path / "articles" / "input"),
            "reports_dir": str(tmp_path / "reports"),
            "data_article_generation_output_dir": str(tmp_path / "output" / "articles"),
            "data_article_generation_artifacts_dir": str(tmp_path / "output" / "article_editor_runs"),
            "data_article_generation_kb_dir": str(tmp_path / "knowledgebase"),
            "data_article_generation_kb_index_dir": str(tmp_path / "knowledgebase_index"),
            "data_article_generation_institutional_memory_dir": str(tmp_path / "institutional_memory"),
            "data_article_generation_prompts_dir": str(prompt_root),
            "data_topic_detection_output_dir": str(tmp_path / "output" / "topics"),
            "data_topic_detection_taxonomies_dir": str(tmp_path / "input" / "taxonomies"),
            "data_topic_detection_taxonomy_cache_dir": str(tmp_path / "input" / "taxonomies" / "cache"),
            "data_hallucination_detection_output_dir": str(tmp_path / "downloads" / "transcripts-hallucinations"),
            "data_article_compiler_input_dir": str(tmp_path / "input" / "newspaper" / "articles"),
            "data_article_compiler_output_file": str(tmp_path / "input" / "newspaper" / "articles.js"),
        },
        "channels": [],
        "defaults": {
            "encoding_name": "o200k_base",
            "repetition_min_k": 1,
            "repetition_min_repetitions": 5,
            "detect_min_k": 3,
        },
        "article_generation": {
            "editor": {
                "editor_max_rounds": 3,
                "prompts": {
                    "writer_prompt_file": "writer.md",
                    "revision_prompt_file": "revision.md",
                    "article_review_prompt_file": "article_review.md",
                    "concern_mapping_prompt_file": "concern_mapping.md",
                    "specialists_dir": "specialists",
                    "fact_check_prompt_file": "fact_check.md",
                    "evidence_finding_prompt_file": "evidence_finding.md",
                    "opinion_prompt_file": "opinion.md",
                    "attribution_prompt_file": "attribution.md",
                    "style_review_prompt_file": "style_review.md",
                },
            },
            "agents": {
                "writer": {"agent_name": "default", "llm": llm},
                "article_review": {"agent_name": "default", "llm": llm},
                "concern_mapping": {"agent_name": "default", "llm": llm},
                "specialists": {
                    "fact_check": {"agent_name": "default", "llm": llm},
                    "evidence_finding": {"agent_name": "default", "llm": llm},
                    "opinion": {"agent_name": "default", "llm": llm},
                    "attribution": {"agent_name": "default", "llm": llm},
                    "style_review": {"agent_name": "default", "llm": llm},
                },
            },
            "knowledge_base": {
                "chunk_size_tokens": 128,
                "chunk_overlap_tokens": 16,
                "timeout_seconds": 10,
                "embedding": {
                    "provider": "lmstudio",
                    "model_name": "embed-model",
                    "api_base": "http://127.0.0.1:1234/v1",
                    "api_key": "lm-studio",
                    "timeout_seconds": 10,
                },
            },
            "perplexity": {
                "api_base": "https://api.perplexity.ai",
                "api_key": "test-key",
                "model": "sonar",
                "timeout_seconds": 10,
            },
            "institutional_memory": {
                "fact_checking_subdir": "fact_checking",
                "evidence_finding_subdir": "evidence_finding",
            },
            "allowed_styles": ["NATURE_NEWS", "SCIAM_MAGAZINE"],
            "default_style_mode": "SCIAM_MAGAZINE",
        },
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_data, handle)
    return Config(config_path)


def _build_real_orchestrator(*, tmp_path: Path, llm_client: _ScriptedLLMClient) -> ChiefEditorOrchestrator:
    config = _write_test_config(tmp_path)
    article_config = config.get_article_generation_config()
    prompts_config = article_config.editor.prompts
    prompt_loader = PromptLoader(root_dir=config.getArticleGenerationPromptsDir())
    institutional_memory = InstitutionalMemoryStore(
        data_dir=config.getArticleGenerationInstitutionalMemoryDir(),
        fact_checking_subdir=article_config.institutional_memory.fact_checking_subdir,
        evidence_finding_subdir=article_config.institutional_memory.evidence_finding_subdir,
    )
    output_handler = OutputHandler(
        final_articles_dir=config.getArticleGenerationOutputDir(),
        run_artifacts_dir=config.getArticleGenerationArtifactsDir(),
    )
    writer_llm = article_config.agents.writer.llm
    specialists_config = article_config.agents.specialists

    return ChiefEditorOrchestrator(
        config=config,
        writer_agent=WriterAgent(
            llm_config=writer_llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            writer_prompt_file=prompts_config.writer_prompt_file,
            revision_prompt_file=prompts_config.revision_prompt_file,
        ),
        article_review_agent=ArticleReviewAgent(
            llm_config=article_config.agents.article_review.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            prompt_file=prompts_config.article_review_prompt_file,
        ),
        concern_mapping_agent=ConcernMappingAgent(
            llm_config=article_config.agents.concern_mapping.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            prompt_file=prompts_config.concern_mapping_prompt_file,
        ),
        fact_check_agent=FactCheckAgent(
            llm_config=specialists_config.fact_check.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            specialists_dir=prompts_config.specialists_dir,
            prompt_file=prompts_config.fact_check_prompt_file,
            knowledge_base_retriever=_UnusedKnowledgeBaseRetriever(),
            institutional_memory=institutional_memory,
            kb_index_version="test-index",
            kb_timeout_seconds=article_config.knowledge_base.timeout_seconds,
        ),
        evidence_finding_agent=EvidenceFindingAgent(
            llm_config=specialists_config.evidence_finding.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            specialists_dir=prompts_config.specialists_dir,
            prompt_file=prompts_config.evidence_finding_prompt_file,
            perplexity_client=_UnusedPerplexityClient(),
            perplexity_model=article_config.perplexity.model,
            institutional_memory=institutional_memory,
        ),
        opinion_agent=OpinionAgent(
            llm_config=specialists_config.opinion.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            specialists_dir=prompts_config.specialists_dir,
            prompt_file=prompts_config.opinion_prompt_file,
        ),
        attribution_agent=AttributionAgent(
            llm_config=specialists_config.attribution.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            specialists_dir=prompts_config.specialists_dir,
            prompt_file=prompts_config.attribution_prompt_file,
        ),
        style_review_agent=StyleReviewAgent(
            llm_config=specialists_config.style_review.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            specialists_dir=prompts_config.specialists_dir,
            prompt_file=prompts_config.style_review_prompt_file,
        ),
        bullet_parser=ArticleReviewBulletParser(),
        institutional_memory=institutional_memory,
        output_handler=output_handler,
    )


class TestArticleGenerationIntegration:
    @pytest.mark.timeout(120)
    def test_real_agents_complete_revision_loop_with_fixture_bundle(self, tmp_path: Path) -> None:
        bundle = load_bundle(FIXTURE_BUNDLE_DIR)
        source_metadata = bundle_to_source_metadata(bundle)
        llm_client = _ScriptedLLMClient()
        orchestrator = _build_real_orchestrator(tmp_path=tmp_path, llm_client=llm_client)

        result = orchestrator.generate_article(
            source_text=bundle.source_text,
            source_metadata=source_metadata,
            style_mode="SCIAM_MAGAZINE",
            reader_preference="focus on deployment tradeoffs",
        )

        assert isinstance(result, ArticleGenerationResult)
        assert result.success is True
        assert result.error is None
        assert result.article is not None
        assert _UNSUPPORTED_LINE not in result.article.article_body
        assert "targeted workflows" in result.article.article_body
        assert result.metadata is not None
        assert result.metadata.article_title == bundle.manifest.article_title
        assert result.editor_report is not None
        assert result.editor_report.final_status == "SUCCESS"
        assert result.editor_report.total_iterations == 2
        assert len(result.editor_report.iterations) == 1
        first_iteration = result.editor_report.iterations[0]
        assert first_iteration.feedback_to_writer is not None
        assert first_iteration.feedback_to_writer.todo_list == [_REWRITE_FIX]
        assert first_iteration.verdicts[0].status == "REWRITE"
        assert result.artifacts_dir is not None
        assert Path(result.artifacts_dir).exists()
        assert llm_client.routes == [
            "writer_generate",
            "article_review",
            "concern_mapping",
            "opinion",
            "writer_revise",
            "article_review",
        ]
