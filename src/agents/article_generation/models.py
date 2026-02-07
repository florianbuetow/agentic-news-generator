"""Pydantic models for the multi-agent article generation system."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ArticleResponse(BaseModel):
    """Generated article content."""

    headline: str
    alternative_headline: str
    article_body: str
    description: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleMetadata(BaseModel):
    """Article source and generation metadata."""

    source_file: str
    channel_name: str
    video_id: str
    article_title: str
    slug: str
    publish_date: str | None
    references: list[dict[str, str]]
    style_mode: Literal["NATURE_NEWS", "SCIAM_MAGAZINE"]
    generated_at: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleReviewRaw(BaseModel):
    """Raw output from the article-review agent."""

    markdown_bullets: str

    model_config = ConfigDict(frozen=True, extra="forbid")


ConcernType = Literal[
    "unsupported_fact",
    "inferred_fact",
    "scope_expansion",
    "editorializing",
    "structured_addition",
    "attribution_gap",
    "certainty_inflation",
    "truncation_completion",
]


class Concern(BaseModel):
    """A single concern identified in the article."""

    concern_id: int
    excerpt: str
    review_note: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleReviewResult(BaseModel):
    """Structured concerns parsed from article-review output."""

    concerns: list[Concern]

    model_config = ConfigDict(frozen=True, extra="forbid")


class ConcernMapping(BaseModel):
    """Mapping from concern to specialist agent."""

    concern_id: int
    concern_type: ConcernType
    selected_agent: Literal["fact_check", "evidence_finding", "opinion", "attribution", "style_review"]
    confidence: Literal["high", "medium", "low"]
    reason: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class ConcernMappingResult(BaseModel):
    """Output from concern-mapping agent."""

    mappings: list[ConcernMapping]

    model_config = ConfigDict(frozen=True, extra="forbid")


class Verdict(BaseModel):
    """Verdict returned by a specialist agent."""

    concern_id: int
    misleading: bool
    status: Literal["KEEP", "REWRITE", "REMOVE"]
    rationale: str
    suggested_fix: str | None
    evidence: str | None
    citations: list[str] | None

    model_config = ConfigDict(frozen=True, extra="forbid")


class FactCheckRecord(BaseModel):
    """Persisted fact-check record for institutional memory."""

    timestamp: str
    article_id: str
    concern_id: int
    prompt: str
    query: str
    normalized_query: str
    model_name: str
    kb_index_version: str
    cache_key_hash: str
    kb_response: str
    verdict: Verdict

    model_config = ConfigDict(frozen=True, extra="forbid")


class EvidenceRecord(BaseModel):
    """Persisted evidence-finding record for institutional memory."""

    timestamp: str
    article_id: str
    concern_id: int
    prompt: str
    query: str
    normalized_query: str
    model_name: str
    cache_key_hash: str
    perplexity_response: str
    citations: list[str]
    verdict: Verdict

    model_config = ConfigDict(frozen=True, extra="forbid")


class WriterFeedback(BaseModel):
    """Compiled feedback sent to the writer in a revision round."""

    iteration: int
    rating: int
    passed: bool
    reasoning: str
    improvement_suggestions: list[str]
    todo_list: list[str]
    verdicts: list[Verdict]

    model_config = ConfigDict(frozen=True, extra="forbid")


class IterationReport(BaseModel):
    """One complete editorial iteration."""

    iteration_number: int
    concerns: list[Concern]
    mappings: list[ConcernMapping]
    verdicts: list[Verdict]
    feedback_to_writer: WriterFeedback | None
    article_draft: ArticleResponse

    model_config = ConfigDict(frozen=True, extra="forbid")


class EditorReport(BaseModel):
    """Complete multi-iteration editor report."""

    iterations: list[IterationReport]
    total_iterations: int
    final_status: Literal["SUCCESS", "FAILED"]
    blocking_concerns: list[Concern] | None

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleGenerationResult(BaseModel):
    """Complete article generation result."""

    success: bool
    article: ArticleResponse | None
    metadata: ArticleMetadata | None
    editor_report: EditorReport | None
    artifacts_dir: str | None
    error: str | None

    model_config = ConfigDict(frozen=True, extra="forbid")
