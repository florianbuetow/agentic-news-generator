# Article Generator Multi-Agent System Design

**Created:** 2025-02-06
**Last updated:** 2026-02-06
**Status:** Approved for Implementation (Revised)

## 0. Requirements Alignment & Explicit Resolutions

### 0.1 Sources of truth (precedence)

1. Repository development rules (AGENTS.md)
2. Editor system requirements: `docs/article-editor-agent-requirements.md`
3. Existing pipeline reality: `scripts/generate-articles.py` and existing `config/config.yaml` patterns
4. This document (must be consistent with the above)

### 0.2 Explicit resolutions (decision-complete)

- **Assignment Agent:** Not implemented. The system uses the existing writer prompt contract’s `OPTIONAL_ANGLE` field (string) when provided.
- **Concern mapping:** Exactly one specialist per concern (no `secondary_agent`).
- **Article-Review output format:** Markdown bullet list (verbatim requirements). The orchestrator parses bullets into structured `Concern` objects.
- **Canonical output contract:** Exactly one JSON output per topic at `data/output/articles/<channel>/<topic_slug>.json`. Intermediate artifacts (iterations, raw agent outputs, editor report, etc.) are stored in a separate run artifacts directory.
- **Error handling:** Fail-fast per topic (article); batch pipeline continues processing remaining topics and exits non-zero if any topics failed.
- **Retries (two separate concepts):**
  - `editor_max_rounds` controls editor→writer feedback rounds.
  - Per-agent `max_retries`/`retry_delay` control LLM transport/parse retries within a single agent call and do not affect editorial rounds.
- **Configuration:** No environment variables. All config values are explicit in YAML. Missing required values raise errors (no implicit defaults).
- **Perplexity integration:** Use Perplexity’s OpenAI-compatible HTTPS API via a thin HTTP client (no assumption of a Python SDK).
- **External evidence usage:** External evidence may only be introduced as clearly labeled footnotes; it must never be silently injected as new main-text facts.
- **Timeout naming:** Use `timeout_seconds` consistently for timeouts in config and code (no rename to `timeout`).
- **`thinking_mode`:** Not supported by this implementation and intentionally omitted from config.

## 1. System Overview

The **Article Generator** is a multi-agent editorial system that produces high-quality science journalism articles from source text. The system mimics a real news organization with specialized agents for writing, reviewing, and fact-checking.

### 1.1 Core Agents

| Agent | Role | External Dependencies |
|-------|------|----------------------|
| **Chief Editor** (Orchestrator) | Coordinates entire editorial process, compiles prompts, manages iteration loop | None |
| **Writer Agent** | Generates article drafts from source text | None |
| **Article-Review Agent** | Identifies unsupported additions in drafts | None |
| **Concern-Mapping Agent** | Routes concerns to appropriate specialists | None |
| **Fact-Check Agent** | Validates facts against knowledge base | Knowledge Base (RAG) |
| **Evidence-Finding Agent** | Searches web for supporting/refuting evidence | Perplexity (OpenAI-compatible HTTPS API) |
| **Opinion Agent** | Evaluates fair interpretation vs misleading | None |
| **Attribution Agent** | Verifies proper source attribution | None |
| **Style-Review Agent** | Checks style compliance | None |

### 1.2 High-Level Flow

```
Source Text + Source Metadata + Style Settings (+ OPTIONAL_ANGLE if provided)
            │
            ▼
    ┌───────────────────┐
    │   Chief Editor    │
    │   (Orchestrator)  │
    └────────┬──────────┘
             │
             ▼
    ┌───────────────────┐
    │   Writer Agent    │
    │ (generate draft)  │
    └────────┬──────────┘
             │
             ▼
    ┌─────────────────────────────────────────────┐
    │         REVIEW LOOP (up to N iterations)    │
    │                                             │
    │  ┌─────────────────┐                        │
    │  │ Article-Review  │ → Markdown bullets     │
    │  │     Agent       │                        │
    │  └────────┬────────┘                        │
    │           │                                 │
    │           ▼                                 │
    │  ┌─────────────────┐                        │
    │  │ Bullet Parser   │ → Concerns (structured)│
    │  └────────┬────────┘                        │
    │           │                                 │
    │           ▼                                 │
    │  ┌─────────────────┐                        │
    │  │ Concern-Mapping │ → Specialist           │
    │  │     Agent       │   assignments          │
    │  └────────┬────────┘                        │
    │           │                                 │
    │           ▼                                 │
    │  ┌─────────────────┐                        │
    │  │   Specialists   │ → Verdicts             │
    │  │  (sequential)   │                        │
    │  │  - Fact-Check   │                        │
    │  │  - Evidence     │                        │
    │  │  - Opinion      │                        │
    │  │  - Attribution  │                        │
    │  │  - Style        │                        │
    │  └────────┬────────┘                        │
    │           │                                 │
    │           ▼                                 │
    │  ┌─────────────────┐                        │
    │  │ Compile Feedback│ → Writer               │
    │  └────────┬────────┘   (if issues remain)  │
    │           │                                 │
    │           ▼                                 │
    │  ┌─────────────────┐                        │
    │  │  Writer Agent   │ → Revised draft        │
    │  │   (revision)    │                        │
    │  └─────────────────┘                        │
    │                                             │
    └─────────────────────────────────────────────┘
             │
             ▼
    ┌───────────────────┐
    │   Final Article   │
    │  + Editor Report  │
    └───────────────────┘
```

## 2. Directory Structure

```
src/agents/article_generation/
├── __init__.py
├── models.py                      # Shared Pydantic models (writer + editor)
│
├── writer/
│   ├── __init__.py
│   ├── agent.py                   # WriterAgent class
│
├── chief_editor/
│   ├── __init__.py
│   ├── orchestrator.py            # ChiefEditorOrchestrator class
│
├── article_review/
│   ├── __init__.py
│   ├── agent.py                   # ArticleReviewAgent class
│
├── concern_mapping/
│   ├── __init__.py
│   ├── agent.py                   # ConcernMappingAgent class
│
├── specialists/
│   ├── __init__.py
│   ├── base.py                    # BaseSpecialistAgent class
│   ├── fact_check/
│   │   ├── __init__.py
│   │   ├── agent.py               # FactCheckAgent class
│   ├── evidence_finding/
│   │   ├── __init__.py
│   │   ├── agent.py               # EvidenceFindingAgent class
│   ├── opinion/
│   │   ├── __init__.py
│   │   ├── agent.py               # OpinionAgent class
│   ├── attribution/
│   │   ├── __init__.py
│   │   ├── agent.py               # AttributionAgent class
│   └── style_review/
│       ├── __init__.py
│       ├── agent.py               # StyleReviewAgent class
│
├── prompts/
│   ├── __init__.py                # Prompt loader utilities (reads from prompts/)
│   └── loader.py
│
└── knowledge_base/
    ├── __init__.py
    ├── indexer.py                 # Haystack + FAISS indexing
    └── retriever.py               # RAG retrieval interface
```

Prompt templates are stored under `prompts/article_editor/`:

```
prompts/article_editor/
├── writer.md
├── revision.md
├── article_review.md
├── concern_mapping.md
└── specialists/
    ├── fact_check.md
    ├── evidence_finding.md
    ├── opinion.md
    ├── attribution.md
    └── style_review.md
```

## 3. Configuration Structure

All configuration lives in `config/config.yaml` (no environment variables). Missing required values must raise errors (no implicit defaults).

The multi-agent editor system is configured under a single `article_generation` section.

Configuration file handling:

- `config/config.yaml.template` contains non-secret placeholders.
- `config/config.yaml` is gitignored and contains real values, including secrets (e.g., Perplexity API key).
- Any required secret value that is missing or empty must raise an error at startup.

```yaml
article_generation:
  editor:
    editor_max_rounds: 3

    # Canonical output: one JSON file per topic (existing pipeline contract).
    output:
      final_articles_dir: ./data/output/articles
      run_artifacts_dir: ./data/output/article_editor_runs
      save_intermediate_results: true

    prompts:
      root_dir: ./prompts/article_editor
      writer_prompt_file: writer.md
      revision_prompt_file: revision.md
      article_review_prompt_file: article_review.md
      concern_mapping_prompt_file: concern_mapping.md
      specialists_dir: specialists

  agents:
    writer_llm:
      model: your-model-name
      api_base: http://127.0.0.1:1234/v1
      api_key: lm-studio
      context_window: 32768
      max_tokens: 8192
      temperature: 0.7
      context_window_threshold: 90
      max_retries: 1
      retry_delay: 2.0
      timeout_seconds: 180

    article_review_llm:
      model: your-model-name
      api_base: http://127.0.0.1:1234/v1
      api_key: lm-studio
      context_window: 32768
      max_tokens: 2048
      temperature: 0.3
      context_window_threshold: 90
      max_retries: 1
      retry_delay: 2.0
      timeout_seconds: 90

    concern_mapping_llm:
      model: your-model-name
      api_base: http://127.0.0.1:1234/v1
      api_key: lm-studio
      context_window: 32768
      max_tokens: 2048
      temperature: 0.3
      context_window_threshold: 90
      max_retries: 1
      retry_delay: 2.0
      timeout_seconds: 60

    specialists:
      fact_check_llm:
        model: your-model-name
        api_base: http://127.0.0.1:1234/v1
        api_key: lm-studio
        context_window: 32768
        max_tokens: 2048
        temperature: 0.2
        context_window_threshold: 90
        max_retries: 1
        retry_delay: 2.0
        timeout_seconds: 60

      evidence_finding_llm:
        model: your-model-name
        api_base: http://127.0.0.1:1234/v1
        api_key: lm-studio
        context_window: 32768
        max_tokens: 2048
        temperature: 0.2
        context_window_threshold: 90
        max_retries: 1
        retry_delay: 2.0
        timeout_seconds: 60

      opinion_llm:
        model: your-model-name
        api_base: http://127.0.0.1:1234/v1
        api_key: lm-studio
        context_window: 32768
        max_tokens: 2048
        temperature: 0.3
        context_window_threshold: 90
        max_retries: 1
        retry_delay: 2.0
        timeout_seconds: 60

      attribution_llm:
        model: your-model-name
        api_base: http://127.0.0.1:1234/v1
        api_key: lm-studio
        context_window: 32768
        max_tokens: 2048
        temperature: 0.2
        context_window_threshold: 90
        max_retries: 1
        retry_delay: 2.0
        timeout_seconds: 60

      style_review_llm:
        model: your-model-name
        api_base: http://127.0.0.1:1234/v1
        api_key: lm-studio
        context_window: 32768
        max_tokens: 2048
        temperature: 0.3
        context_window_threshold: 90
        max_retries: 1
        retry_delay: 2.0
        timeout_seconds: 60

  knowledge_base:
    data_dir: ./data/knowledgebase
    index_dir: ./data/knowledgebase_index
    chunk_size_tokens: 512
    chunk_overlap_tokens: 50
    timeout_seconds: 30
    embedding:
      provider: lmstudio
      model_name: text-embedding-bge-large-en-v1.5
      api_base: http://127.0.0.1:1234/v1
      api_key: lm-studio
      timeout_seconds: 30

  perplexity:
    api_base: https://api.perplexity.ai
    api_key: "<REQUIRED_NON_EMPTY_STRING>"
    model: sonar
    timeout_seconds: 45

  institutional_memory:
    data_dir: ./data/institutional_memory
    fact_checking_subdir: fact_checking
    evidence_finding_subdir: evidence_finding

  allowed_styles:
    - "NATURE_NEWS"
    - "SCIAM_MAGAZINE"
  default_style_mode: "SCIAM_MAGAZINE"
  default_target_length_words: "900-1200"
```

### 3.1 Input Contract (Topic JSON)

The pipeline processes topic JSON files (one per topic) and must provide explicit inputs to the orchestrator.

Required fields in each topic JSON:

- `source_text`: string (non-empty)
- `source_metadata`: object with these required keys (all non-empty strings unless noted):
  - `source_file`
  - `channel_name`
  - `video_id`
  - `video_title`
  - `publish_date` (string; may be `null` only if explicitly present as null)
  - `topic_slug`
  - `topic_title`

Optional fields in each topic JSON:

- `style_mode`: `"NATURE_NEWS"` or `"SCIAM_MAGAZINE"` (if missing, the pipeline must use `article_generation.default_style_mode`)
- `target_length_words`: string (if missing, the pipeline must use `article_generation.default_target_length_words`)
- `optional_angle`: string or `null` (if missing, treated as `null`)

If any required field/key is missing, the topic must fail-fast with a clear error message.

## 4. Data Models

All models in `src/agents/article_generation/models.py`:

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict

# === Article & Metadata ===

class ArticleResponse(BaseModel):
    """Generated article content."""
    headline: str
    alternativeHeadline: str
    articleBody: str  # Markdown formatted
    description: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleMetadata(BaseModel):
    """Article source and generation metadata."""
    source_file: str  # Path to topic JSON input file (string)
    channel_name: str
    video_id: str
    video_title: str
    publish_date: str | None = None
    topic_slug: str
    topic_title: str
    style_mode: Literal["NATURE_NEWS", "SCIAM_MAGAZINE"]
    target_length_words: str
    generated_at: str  # ISO 8601 UTC timestamp

    model_config = ConfigDict(frozen=True, extra="forbid")


# === Article Review (raw + parsed) ===

class ArticleReviewRaw(BaseModel):
    """Raw output from Article-Review Agent (markdown bullets)."""
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
    excerpt: str  # Problematic text from article (quoted in review bullets)
    review_note: str  # Review explanation / mismatch note

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleReviewResult(BaseModel):
    """Parsed, structured concerns derived from Article-Review markdown bullets."""
    concerns: list[Concern]

    model_config = ConfigDict(frozen=True, extra="forbid")


# === Concern Mapping ===

class ConcernMapping(BaseModel):
    """Mapping of a concern to a specialist agent."""
    concern_id: int
    concern_type: ConcernType
    selected_agent: Literal["fact_check", "evidence_finding", "opinion", "attribution", "style_review"]
    confidence: Literal["high", "medium", "low"]
    reason: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class ConcernMappingResult(BaseModel):
    """Output from Concern-Mapping Agent."""
    mappings: list[ConcernMapping]

    model_config = ConfigDict(frozen=True, extra="forbid")


# === Specialist Verdicts ===

class Verdict(BaseModel):
    """Verdict from a specialist agent."""
    concern_id: int
    misleading: bool
    status: Literal["KEEP", "REWRITE", "REMOVE"]
    rationale: str
    suggested_fix: str | None = None
    evidence: str | None = None  # For fact-check/evidence-finding (short blurb)
    citations: list[str] | None = None  # URLs or citation strings provided by pipeline (never invented)

    model_config = ConfigDict(frozen=True, extra="forbid")


# === Institutional Memory ===

class FactCheckRecord(BaseModel):
    """Persisted fact-check for institutional memory."""
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
    """Persisted evidence search for institutional memory."""
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


# === Feedback ===

class WriterFeedback(BaseModel):
    """Compiled feedback for writer revision (matches retry template contract)."""
    iteration: int
    rating: int  # 1-10
    passed: bool
    reasoning: str
    improvement_suggestions: list[str]
    todo_list: list[str]
    verdicts: list[Verdict]

    model_config = ConfigDict(frozen=True, extra="forbid")


# === Editor Report ===

class IterationReport(BaseModel):
    """Report for a single editorial iteration."""
    iteration_number: int
    concerns: list[Concern]
    mappings: list[ConcernMapping]
    verdicts: list[Verdict]
    feedback_to_writer: WriterFeedback | None
    article_draft: ArticleResponse

    model_config = ConfigDict(frozen=True, extra="forbid")


class EditorReport(BaseModel):
    """Complete editorial report."""
    iterations: list[IterationReport]
    total_iterations: int
    final_status: Literal["SUCCESS", "FAILED"]
    blocking_concerns: list[Concern] | None = None  # If failed

    model_config = ConfigDict(frozen=True, extra="forbid")


# === Final Result ===

class ArticleGenerationResult(BaseModel):
    """Complete result of article generation."""
    success: bool
    article: ArticleResponse | None = None
    metadata: ArticleMetadata | None = None
    editor_report: EditorReport | None = None
    artifacts_dir: str | None = None
    error: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")
```

## 5. Agent Base Class

All agents are dependency-injected and testable in isolation. Agents do not perform file I/O directly; persistence is handled by dedicated components (output handler, institutional memory store).

All LLM calls go through an injected `LLMClient` adapter (backed by LiteLLM in production, mocked in tests).

```python
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar

from pydantic import BaseModel

from src.config import Config, LLMConfig
from src.util.token_validator import validate_token_usage


class LLMClient(Protocol):
    """Protocol for LLM interactions (enables mocking)."""
    def complete(self, *, llm_config: LLMConfig, messages: list[dict]) -> str: ...


class KnowledgeBaseRetriever(Protocol):
    """Protocol for KB retrieval (enables mocking)."""
    def search(self, *, query: str, top_k: int, timeout_seconds: int) -> list[dict]: ...


class PerplexityClient(Protocol):
    """Protocol for Perplexity API (enables mocking).

    Implementation uses Perplexity's OpenAI-compatible HTTPS API.
    """
    def search(self, *, query: str, model: str, timeout_seconds: int) -> dict: ...


T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC):
    """Base class for all agents with common functionality."""

    def __init__(self, *, llm_config: LLMConfig, config: Config, llm_client: LLMClient) -> None:
        self._llm_config = llm_config
        self._config = config
        self._llm_client = llm_client

    def _validate_tokens(self, messages: list[dict]) -> int:
        """Validate token usage before LLM call."""
        return validate_token_usage(
            messages=messages,
            context_window=self._llm_config.context_window,
            threshold=self._llm_config.context_window_threshold,
            encoding_name=self._config.getEncodingName(),
        )

    def _call_llm(self, messages: list[dict]) -> str:
        """Call LLM with standard error handling. Fails fast on errors."""
        self._validate_tokens(messages)
        return self._llm_client.complete(llm_config=self._llm_config, messages=messages)

    def _parse_json_response(self, response: str, model_class: type[T]) -> T:
        """Parse and validate JSON response with markdown fence stripping."""
        text = response.strip()
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            if text.startswith("```json"):
                text = text[7:]
            else:
                text = text[3:]
            # Remove closing fence
            text = text.rsplit("```", 1)[0].strip()
        return model_class.model_validate_json(text)


class BaseSpecialistAgent(BaseAgent):
    """Base for specialist agents that produce verdicts."""

    @abstractmethod
    def evaluate(
        self,
        concern: "Concern",
        article: "ArticleResponse",
        source_text: str,
        style_requirements: str,
    ) -> "Verdict":
        """Evaluate a concern and return a verdict."""
        ...
```

## 6. Chief Editor Orchestrator

The orchestrator is the single entry point for multi-agent article generation + editorial review. It preserves the existing writer prompt contract (`STYLE_MODE`, `TARGET_LENGTH_WORDS`, `OPTIONAL_ANGLE`, `SOURCE_TEXT`, `OPTIONAL_METADATA`) and adds a downstream editorial loop.

```python
class ChiefEditorOrchestrator:
    """Orchestrates the complete article generation and editorial review process."""

    def __init__(
        self,
        *,
        config: Config,
        writer_agent: WriterAgent,
        article_review_agent: ArticleReviewAgent,
        concern_mapping_agent: ConcernMappingAgent,
        fact_check_agent: FactCheckAgent,
        evidence_finding_agent: EvidenceFindingAgent,
        opinion_agent: OpinionAgent,
        attribution_agent: AttributionAgent,
        style_review_agent: StyleReviewAgent,
        bullet_parser: "ArticleReviewBulletParser",
        institutional_memory: "InstitutionalMemoryStore",
        output_handler: "OutputHandler",
    ) -> None:
        ...

    def generate_article(
        self,
        *,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_mode: str,
        target_length_words: str,
        optional_angle: str | None,
    ) -> ArticleGenerationResult:
        """Main entry point for article generation."""
        ...
```

### 6.1 Orchestration Flow

```
generate_article(source_text, metadata, style_mode, target_length_words, optional_angle?)
│
├─► [0] Setup run artifacts directory
│   └─► run_id = <UTC timestamp YYYYMMDDTHHMMSSZ> + "_" + <8 hex of sha256(source_file + topic_slug + style_mode + target_length_words)>
│   └─► artifacts_dir = <run_artifacts_dir>/<channel>/<topic_slug>/<run_id>/
│
├─► [1] Initial Draft (Writer)
│   └─► WriterAgent.generate(source_text, metadata, style_mode, target_length_words, optional_angle?)
│       └─► Returns: ArticleResponse
│   └─► Save: iter1_writer_draft.json + iter1_writer_draft.md
│
└─► [2] Review Loop (round = 1..editor_max_rounds)
    │
    ├─► [2a] Article Review (raw markdown bullets)
    │   └─► ArticleReviewAgent.review(article, source_text, metadata)
    │       └─► Returns: ArticleReviewRaw.markdown_bullets
    │   └─► Save: iter{n}_article_review_raw.md
    │
    ├─► [2b] Parse review bullets
    │   └─► BulletParser.parse(markdown_bullets) -> ArticleReviewResult (list[Concern])
    │   └─► Save: iter{n}_article_review.json
    │
    ├─► [2c] Check: No concerns?
    │   └─► YES: Return SUCCESS with article + report
    │
    ├─► [2d] Concern Mapping
    │   └─► ConcernMappingAgent.map(concerns, style_requirements, source_text, article)
    │       └─► Returns: list[ConcernMapping]
    │   └─► Save: iter{n}_concern_mapping.json
    │
    ├─► [2e] Run Specialists (sequential, one per concern)
    │   └─► For each mapping:
    │       └─► specialist = get_specialist(mapping.selected_agent)
    │       └─► verdict = specialist.evaluate(concern, article, source_text)
    │       └─► If fact_check/evidence_finding: check institutional_memory cache; persist on cache miss
    │   └─► Save: iter{n}_verdicts.json (+ per-agent JSON files, optional)
    │
    ├─► [2f] Check: passed?
    │   └─► YES: Return SUCCESS
    │
    ├─► [2g] Check: Last round?
    │   └─► YES: Return FAILED with blocking_concerns + best article
    │
    ├─► [2h] Compile Feedback (deterministic)
    │   ├─► required_todos = [v.suggested_fix for v in verdicts if v.status in ("REWRITE","REMOVE") and v.suggested_fix is not None]
    │   ├─► optional_suggestions = [v.rationale for v in verdicts if v.status == "KEEP"]
    │   ├─► passed = false
    │   ├─► rating = clamp(1, 10, 10 - 2*len(required_todos) - 1*len(optional_suggestions))
    │   └─► reasoning = summary of required_todos (deterministic order)
    │   └─► Save: iter{n}_feedback.json
    │
    └─► [2i] Writer Revision
        └─► WriterAgent.revise(article, feedback, source_text, metadata, style_mode, target_length_words, optional_angle?)
            └─► Returns: ArticleResponse (revised)
        └─► Save: iter{n+1}_writer_draft.json + iter{n+1}_writer_draft.md
        └─► Continue loop...

Final:
└─► Save: article_result.json (complete nested result)
└─► Save: editor_report.json (full editorial report)
└─► Save: article.md (final rendered article)
└─► Return: ArticleGenerationResult
```

### 6.1.1 Bullet Parsing Rules (Article-Review → Concern)

The Article-Review Agent returns a markdown bullet list. The bullet parser converts it into a list of `Concern` objects deterministically:

- **Bullet detection:** A new bullet starts on a line beginning with `- ` or `* `. Multi-line bullets continue until the next bullet start.
- **concern_id:** Assigned sequentially starting at `1` in bullet order.
- **excerpt extraction:**
  - If the bullet contains a quoted substring between `“` and `”`, use the first such substring as `excerpt`.
  - Else, if the bullet contains a quoted substring between `"` and `"`, use the first such substring as `excerpt`.
  - Else, set `excerpt` to the full bullet text.
- **review_note:** The full bullet text (verbatim).
- **Invalid output handling:** If the raw markdown is non-empty but no bullets are detected, raise an error (fail-fast for the topic).

### 6.1.2 Feedback Computation Rules (Deterministic)

Writer feedback is compiled without additional LLM calls:

- **required_todos:** `suggested_fix` for verdicts with `status` in `("REWRITE", "REMOVE")` (drop entries where `suggested_fix` is `null`).
- **improvement_suggestions:** Up to 5 items, in ascending `concern_id` order, using `rationale` from `KEEP` verdicts.
- **passed:** Always `false` when feedback is sent (feedback is only sent when not passed).
- **rating:** `clamp(1, 10, 10 - 2*len(required_todos) - 1*len(improvement_suggestions))`.
- **reasoning:** Deterministic summary listing the required_todos in order.

### 6.2 Footnotes & External Evidence Rules

- Footnotes are written in Markdown using `[^n]` references in `articleBody` and a terminal `## Footnotes` section.
- Only citations returned by the pipeline (specialist verdict `citations`) may appear in footnotes. Writers must not invent URLs/DOIs/papers.
- Evidence-Finding may support:
  - **Contradiction:** Main text must be rewritten/removed to avoid misleading claims. A footnote may summarize the contradiction and provide citations.
  - **Support:** Main text must not add new factual claims beyond the source text. Supporting evidence may be added only as a footnote that provides context and cites sources.

## 7. Output Structure

### 7.1 Canonical Output (Downstream Contract)

Exactly one JSON file per topic is written to:

`data/output/articles/<channel>/<topic_slug>.json`

This file is the canonical contract for downstream systems. It contains the final article, metadata, and editor report; intermediate files are never required for consumption.

Canonical JSON shape:

```json
{
  "success": true,
  "article": { "headline": "...", "alternativeHeadline": "...", "articleBody": "...", "description": "..." },
  "metadata": { "...": "..." },
  "editor_report": { "...": "..." },
  "artifacts_dir": "data/output/article_editor_runs/<channel>/<topic_slug>/<run_id>",
  "error": null
}
```

On failure, the same file is still written with `success=false`, `error` populated, and `article`/`metadata`/`editor_report` either `null` or best-effort (explicitly documented in the implementation).

### 7.2 Run Artifacts (Intermediate Outputs)

All intermediate outputs are written under a per-run directory:

`data/output/article_editor_runs/<channel>/<topic_slug>/<run_id>/`

Required artifacts:

```
data/output/article_editor_runs/<channel>/<topic_slug>/<run_id>/
├── iter1_writer_draft.json
├── iter1_writer_draft.md
├── iter1_article_review_raw.md
├── iter1_article_review.json
├── iter1_concern_mapping.json
├── iter1_verdicts.json
├── iter1_feedback.json
├── iter2_writer_draft.json
├── iter2_writer_draft.md
├── ...
├── editor_report.json
├── article_result.json
└── article.md
```

The `run_id` is derived deterministically as described in §6.1 (no implicit defaults).

### 7.3 Institutional Memory (Persistence + Retrieval)

```
data/institutional_memory/
├── fact_checking/
│   └── {date}/
│       └── {cache_key_hash}.json
└── evidence_finding/
    └── {date}/
        └── {cache_key_hash}.json
```

Retrieval and dedup policy (must be implemented exactly):

- **Cache key:** `(agent_name, normalized_query, model_name, kb_index_version)` for KB; `(agent_name, normalized_query, model_name)` for Perplexity.
- **cache_key_hash:** `sha256("|".join(cache_key_fields)).hexdigest()[:16]` (hex, lower-case)
- **Lookup timing:** Before making any KB/Perplexity call.
- **Cache hit behavior:** Reuse the previous record’s `verdict`, `evidence`, and `citations` and skip the external call.
- **Cache miss behavior:** Perform the external call and persist a new record.

### 7.4 Knowledge Base

```
data/knowledgebase/
├── *.txt
└── *.md
```

Source documents for the RAG system. Indexed on startup using Haystack + FAISS.

## 8. External Dependencies

### 8.1 Knowledge Base (Fact-Check Agent)

- **Library:** Haystack
- **Index:** FAISS (persisted under `knowledge_base.index_dir`, loaded on startup)
- **Embeddings:** LM Studio (same model as topic detection: `text-embedding-bge-large-en-v1.5`)
- **Documents:** `data/knowledgebase/*.txt` and `*.md`
- **Chunking:** 512 tokens, 50 token overlap

Index lifecycle (explicit):

- If `knowledge_base.index_dir` exists and matches the current embedding model settings, load it.
- Otherwise, rebuild the index from `knowledge_base.data_dir` and write it to `knowledge_base.index_dir`.

### 8.2 Perplexity API (Evidence-Finding Agent)

- **Integration:** Perplexity OpenAI-compatible HTTPS API (`perplexity.api_base`)
- **Model:** configured in `perplexity.model` (explicit)
- **Timeout:** configured in `perplexity.timeout_seconds` (explicit)
- **Returns:** search response + a list of citations (URLs or citation strings)
- **Pipeline rule:** citations must be persisted and passed through to `Verdict.citations` so the writer can format footnotes; citations must never be invented by the writer

## 9. Error Handling

**Fail-fast per topic:** For a single topic/article run, all errors surface immediately. No swallowing exceptions.

- LLM errors → Raise immediately
- Validation errors → Raise immediately
- API timeouts → Raise immediately
- Any agent failure → Abort current topic and return error

The orchestrator catches top-level exceptions and returns `ArticleGenerationResult` with `success=False` and `error` message.

**Batch behavior:** The batch pipeline continues processing remaining topics. At the end it exits with code `1` if any topic failed, otherwise `0`.

## 10. Testing Strategy

### 10.1 Test Structure

```
tests/
├── unit/
│   └── agents/
│       └── article_generation/
│           ├── test_writer_agent.py
│           ├── test_article_review_agent.py
│           ├── test_article_review_bullet_parser.py
│           ├── test_concern_mapping_agent.py
│           ├── test_fact_check_agent.py
│           ├── test_evidence_finding_agent.py
│           ├── test_opinion_agent.py
│           ├── test_attribution_agent.py
│           ├── test_style_review_agent.py
│           ├── test_chief_editor_orchestrator.py
│           ├── test_output_handler.py
│           └── test_institutional_memory_store.py
│
│       └── knowledge_base/
│           └── test_knowledge_base.py
│
├── integration/
│   └── agents/
│       └── article_generation/
│           ├── test_writer_agent_integration.py
│           ├── test_fact_check_with_kb_integration.py
│           ├── test_evidence_finding_with_perplexity_integration.py  # opt-in (requires API key)
│           ├── test_writer_to_review_integration.py
│           ├── test_review_to_specialists_integration.py
│           └── test_full_pipeline_integration.py
│
└── fixtures/
    └── article_generation/
        ├── sample_transcript.txt
        ├── sample_article_draft.json
        ├── sample_concerns.json
        ├── sample_verdicts.json
        └── mock_llm_responses/
```

### 10.2 Test Levels

| Level | What's Tested | LLM | Agents | Speed |
|-------|--------------|-----|--------|-------|
| Unit | Single agent logic | Mocked | Single | Fast |
| Integration (single) | Agent + real LLM | Real | Single | Medium |
| Integration (multi) | Agent data exchange | Mocked | 2-3 | Fast |
| Integration (full) | Orchestrator + mocked agents | Mocked | All (mocked) | Fast |
| Integration (online) | Perplexity + real key | Real | Single | Slow |
| E2E | Full pipeline | Real | All | Slow |

### 10.3 Design for Testability

- All agents accept dependencies via constructor (dependency injection)
- Protocols defined for all external services (LLMClient, KnowledgeBaseRetriever, PerplexityClient)
- Each agent can be instantiated and tested independently
- Orchestrator accepts all agents as constructor parameters (can inject mocks)
- Bullet parsing, rating computation, and output path derivation are deterministic and unit-tested

## 11. Implementation Phases

### Phase 1: Foundation
**Deliverables:**
- Base agent class with dependency injection
- All Pydantic models (writer + editor system)
- Config updates (single `article_generation` section; breaking change, no compatibility)
- Prompt templates in `prompts/article_editor/` + prompt loader
- Deterministic bullet parser for Article-Review markdown output
- Output handler that writes canonical output JSON + run artifacts directory
- Institutional memory store (persistence + lookup key + cache hit policy)
- Add `timeout_seconds` to per-agent LLM config (explicit; required)

**Tests:** Unit tests for models, config loading, bullet parser, output handler, institutional memory store

**Checkpoint:** Can load config, parse review bullets, and write canonical output + run artifacts for a mocked run

---

### Phase 2: Core Agents
**Deliverables:**
- Writer Agent (refactor existing; prompt contract preserved)
- Article-Review Agent (outputs markdown bullets)
- Concern-Mapping Agent (outputs strict JSON mappings; one specialist per concern)

**Tests:** Unit tests + single-agent integration tests for each

**Checkpoint:** Writer produces valid JSON; Article-Review produces markdown bullets; Mapping produces valid JSON

---

### Phase 3: Specialist Agents (No External Dependencies)
**Deliverables:**
- Opinion Agent
- Attribution Agent
- Style-Review Agent

**Tests:** Unit tests + single-agent integration tests for each

**Checkpoint:** Each specialist produces valid verdicts

---

### Phase 4: Infrastructure
**Deliverables:**
- Knowledge Base (Haystack + FAISS indexer/retriever)
- Perplexity OpenAI-compatible HTTP client wrapper
- Institutional memory persistence

**Tests:** Unit tests + offline integration tests (KB local fixtures)

**Checkpoint:** KB indexes/retrieves; Perplexity client returns citations when configured; cache lookup works

---

### Phase 5: Specialist Agents (With External Dependencies)
**Deliverables:**
- Fact-Check Agent (uses Knowledge Base)
- Evidence-Finding Agent (uses Perplexity)

**Tests:** Unit tests + opt-in online integration tests with real Perplexity API key

**Checkpoint:** Specialists correctly use external services and persist to institutional memory

---

### Phase 6: Orchestration
**Deliverables:**
- Chief Editor Orchestrator
- Writer feedback compilation (deterministic rating/pass/reasoning + TODO list)
- Full integration

**Tests:** Orchestrator with mocked agents, full integration tests

**Checkpoint:** Complete flow works end-to-end

---

### Phase 7: Pipeline Integration
**Deliverables:**
- Update `scripts/generate-articles.py` to use ChiefEditorOrchestrator
- Preserve canonical output path contract (`data/output/articles/<channel>/<topic_slug>.json`)
- Extend canonical output JSON to include `editor_report` + `artifacts_dir` (breaking schema change; no compatibility)
- E2E tests

**Tests:** E2E tests processing real transcripts

**Checkpoint:** Pipeline processes transcripts into reviewed articles

## 12. Acceptance Criteria

### 12.1 Functional Requirements

- [ ] System processes topic JSON inputs (source_text + source_metadata) from the existing pipeline
- [ ] Writer receives `OPTIONAL_ANGLE` when provided (no Assignment Agent)
- [ ] Writer Agent generates articles matching specified style
- [ ] Article-Review Agent outputs markdown bullet list of unsupported additions (verbatim requirements)
- [ ] Bullet parser deterministically converts bullets into `Concern` objects
- [ ] Concern-Mapping Agent maps each concern to exactly one specialist agent
- [ ] All specialist agents produce valid verdicts
- [ ] Fact-Check Agent queries knowledge base
- [ ] Evidence-Finding Agent queries Perplexity and returns citations for footnotes
- [ ] Chief Editor runs up to `editor_max_rounds` and stops early on pass
- [ ] Per-topic fail-fast; batch continues and reports success/failure counts
- [ ] Run artifacts are saved (intermediate JSON + MD, editor_report.json, article.md)
- [ ] Institutional memory persists and is consulted before external calls
- [ ] External evidence is only introduced as footnotes (never silently injected into main text)
- [ ] Canonical output JSON includes final article + editor report + artifacts_dir

### 12.2 Non-Functional Requirements

- [ ] All agents testable in isolation (dependency injection)
- [ ] All paths configurable via config.yaml
- [ ] All timeouts configurable per-agent and per-service via `timeout_seconds`
- [ ] Each phase testable before proceeding to next
- [ ] No hardcoded values anywhere
- [ ] No swallowed errors

### 12.3 Configuration Requirements

- [ ] Single `article_generation` section in config.yaml for the editor system
- [ ] Individual LLM config per agent, each with `timeout_seconds`
- [ ] Knowledge base config with embedding settings
- [ ] Perplexity API config
- [ ] Institutional memory paths
- [ ] Output paths
- [ ] Prompt root directory configured (no prompts embedded as Python constants)
- [ ] No environment-variable-based configuration

## 13. References

- Requirements: `docs/article-editor-agent-requirements.md`
- Existing patterns: `src/agents/topic_segmentation/`
- Config template: `config/config.yaml.template`
