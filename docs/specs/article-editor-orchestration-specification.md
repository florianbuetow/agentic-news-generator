# Article Editor Orchestration — Behavioral Specification

## Objective

The Chief Editor Orchestrator coordinates a multi-agent editorial pipeline that generates science journalism articles from source text and iteratively reviews them for unsupported additions. The orchestrator does not perform editorial judgment itself — it delegates to specialized agents and compiles their outputs into deterministic writer feedback. The pipeline produces a canonical output JSON per article and preserves all intermediate artifacts for auditability.

## Existing Implementation

The orchestrator is fully implemented in `src/agents/article_generation/chief_editor/orchestrator.py`. This spec documents the behavioral contract the implementation satisfies and identifies areas where test coverage must be strengthened.

Supporting components that are also fully implemented:
- `src/agents/article_generation/base.py` — Agent protocols and base classes
- `src/agents/article_generation/models.py` — All Pydantic data models
- `src/agents/article_generation/chief_editor/bullet_parser.py` — Markdown bullet parsing
- `src/agents/article_generation/chief_editor/output_handler.py` — Canonical + artifacts output
- `src/agents/article_generation/chief_editor/institutional_memory.py` — Cache store
- `src/agents/article_generation/chief_editor/verbose_context_logger.py` — Artifact sequencing
- `src/agents/article_generation/agent.py` — Agent factory (`build_chief_editor_orchestrator`)

## User Stories & Acceptance Criteria

### US-1: As a pipeline operator, I want to generate an article from a source text bundle, so that I get a reviewed, high-quality article as structured JSON output.

**AC-1.1:** Given a valid article input bundle (source_text, source_metadata with channel_name/slug/source_file, style_mode, reader_preference), when `generate_article()` is called, then the orchestrator returns an `ArticleGenerationResult` with `success=True` or `success=False`.

**AC-1.2:** Given a valid input bundle, when the writer agent produces a draft and the article review agent finds zero concerns, then the orchestrator returns `ArticleGenerationResult` with `success=True`, a non-null `article`, and `editor_report.final_status == "SUCCESS"` after exactly 1 review iteration.

**AC-1.3:** Given a valid input bundle, when the writer agent produces a draft and specialists find no misleading concerns (all `verdict.misleading == False`), then the orchestrator returns `success=True` with `final_status == "SUCCESS"` after the current iteration.

**AC-1.4:** Given a valid input bundle, when specialists find misleading concerns and the round is not the last, then the orchestrator compiles feedback and requests a writer revision before starting the next iteration.

**AC-1.5:** Given a valid input bundle, when misleading concerns persist after `editor_max_rounds` iterations, then the orchestrator returns `ArticleGenerationResult` with `success=False`, `error` containing "Unresolved misleading concerns after editor_max_rounds", the best available `article`, and `editor_report.blocking_concerns` listing the unresolved concerns.

---

### US-2: As a pipeline operator, I want the orchestrator to follow a deterministic editorial review loop, so that article quality is systematically validated.

**AC-2.1:** The editorial loop executes the following steps in order for each iteration:
1. Article review agent produces markdown bullets
2. Bullet parser converts bullets to structured `Concern[]`
3. If no concerns: return SUCCESS
4. Concern mapping agent maps each concern to exactly one specialist
5. Each mapped concern is evaluated sequentially by its assigned specialist
6. If no verdicts are misleading: return SUCCESS
7. If last round and misleading verdicts remain: return FAILED
8. Compile deterministic `WriterFeedback` from verdicts
9. Writer agent revises the article using the feedback
10. Loop continues to next iteration

**AC-2.2:** The loop runs for at most `editor_max_rounds` iterations (read from `config.get_article_editor_max_rounds()`). The loop counter starts at 1 and increments by 1 each iteration.

**AC-2.3:** Specialist agents are dispatched sequentially (not in parallel), one per mapped concern, in the order returned by the concern mapping agent.

**AC-2.4:** The orchestrator routes concerns to specialists via a lookup map:
- `"fact_check"` → fact check agent
- `"evidence_finding"` → evidence finding agent
- `"opinion"` → opinion agent
- `"attribution"` → attribution agent
- `"style_review"` → style review agent

If `mapping.selected_agent` is not one of these five values, the orchestrator raises `ValueError`.

---

### US-3: As a pipeline operator, I want deterministic writer feedback compiled from specialist verdicts, so that the writer receives actionable revision instructions without additional LLM calls.

**AC-3.1:** Feedback compilation uses no LLM calls. All fields are derived deterministically from the verdict list.

**AC-3.2:** Verdicts are sorted by `concern_id` (ascending) before compilation.

**AC-3.3:** `todo_list` contains `suggested_fix` from verdicts where `status` is `"REWRITE"` or `"REMOVE"` and `suggested_fix` is not `None`. Verdicts with `suggested_fix == None` are excluded from the todo list.

**AC-3.4:** `improvement_suggestions` contains the `rationale` from verdicts where `status == "KEEP"`, limited to the first 5 items in concern_id order.

**AC-3.5:** `rating` is computed as `clamp(1, 10, 10 - 2 * len(todo_list) - len(improvement_suggestions))`.

**AC-3.6:** `passed` is always `False` (feedback is only compiled when the article has not passed).

**AC-3.7:** `reasoning` is a deterministic summary:
- If `todo_list` is empty: `"No required rewrites or removals were identified."`
- Otherwise: `"Required changes:\n"` followed by a numbered list of todo items.

**AC-3.8:** The `WriterFeedback` object includes the full sorted `verdicts` list so the writer can reference citations for footnotes.

---

### US-4: As a pipeline operator, I want all intermediate artifacts preserved per run, so that I can audit and debug the editorial process.

**AC-4.1:** Before the loop starts, the orchestrator initializes a run artifacts directory at `{run_artifacts_dir}/{channel_name}/{slug}/{run_id}/`, where `run_id` is `{UTC_timestamp}_{8_char_sha256_suffix}`. The suffix is derived from `sha256(source_file + slug + style_mode)`.

**AC-4.2:** The following source artifacts are logged before the loop:
- `000_source_context.txt` — the full source text
- `001_source_metadata.json` — source metadata dict
- `002_source_run_params.json` — `{style_mode, reader_preference}`

**AC-4.3:** For each iteration, the following artifacts are logged in sequence:
- Writer draft prompt (`.md`) and output (`.json` and `.md`)
- Article review prompt (`.md`) and output (`.md`)
- Parsed review concerns (`.json`)
- Concern mapping prompt (`.md`) and output (`.json`)
- Per-specialist prompt (`.md`) and verdict (`.json`)
- Compiled feedback (`.json`)

**AC-4.4:** Final artifacts are logged with sequence numbers >= 900:
- `900_final_article.md` — the final article body
- `901_final_article_result.json` — the complete `ArticleGenerationResult`
- `902_final_editor_report.json` — the complete `EditorReport`

**AC-4.5:** Artifact files are sequentially numbered with a 3-digit prefix (e.g., `003_writer_prompt.md`). Sequence numbers increment monotonically within a run. Numbers below 900 are for iteration artifacts; 900+ are reserved for final artifacts.

---

### US-5: As a pipeline operator, I want a canonical output file written for every article, so that downstream systems have a single stable contract to consume.

**AC-5.1:** After the loop completes (success or failure), the orchestrator writes a canonical output file at `{final_articles_dir}/{channel_name}/{slug}.json`.

**AC-5.2:** The canonical output is a JSON serialization of `ArticleGenerationResult` containing:
- `success` (bool)
- `article` (`ArticleResponse` or null on failure)
- `metadata` (`ArticleMetadata` with `generated_at` as ISO 8601 UTC)
- `editor_report` (`EditorReport` with all `IterationReport` entries)
- `artifacts_dir` (string path to the run artifacts directory)
- `error` (string or null)

**AC-5.3:** On failure, the canonical output still includes the best available article (the last revision), metadata, and the complete editor report. `error` is non-null.

---

### US-6: As a pipeline operator, I want the orchestrator to validate required metadata fields upfront, so that failures happen early with clear error messages.

**AC-6.1:** The orchestrator requires the following metadata fields to be non-null strings: `channel_name`, `slug`, `source_file`. If any are missing or null, `ValueError` is raised before any agent calls.

**AC-6.2:** Building `ArticleMetadata` additionally requires: `video_id`, `article_title`. If missing, `ValueError` is raised.

**AC-6.3:** `publish_date` is optional (may be null). `references` is optional (may be absent; defaults to empty list when missing; parsed from JSON string when present).

---

### US-7: As a developer, I want the orchestrator to accept agents via protocol-based dependency injection, so that I can swap mock and real agent implementations.

**AC-7.1:** The orchestrator constructor accepts all agents as keyword-only arguments typed to their respective protocols:
- `writer_agent: WriterAgentProtocol`
- `article_review_agent: ArticleReviewAgentProtocol`
- `concern_mapping_agent: ConcernMappingAgentProtocol`
- `fact_check_agent: SpecialistAgentProtocol`
- `evidence_finding_agent: SpecialistAgentProtocol`
- `opinion_agent: SpecialistAgentProtocol`
- `attribution_agent: SpecialistAgentProtocol`
- `style_review_agent: SpecialistAgentProtocol`

**AC-7.2:** The orchestrator also accepts:
- `bullet_parser: ArticleReviewBulletParser`
- `institutional_memory: InstitutionalMemoryStore`
- `output_handler: OutputHandler`
- `config: Config`

**AC-7.3:** The agent factory (`build_chief_editor_orchestrator` in `agent.py`) reads `agent_name` from each agent's config section. When `agent_name == "mock"`, it instantiates the mock implementation. When `agent_name == "default"`, it instantiates the real LLM-backed implementation. Any other value raises `ValueError`.

**AC-7.4:** The agent factory is the only place where concrete agent classes are imported and instantiated. The orchestrator only depends on protocols, never on concrete agent classes.

---

### US-8: As a pipeline operator, I want the editorial loop to handle agent errors as fail-fast per article, so that one article failure does not block other articles in a batch.

**AC-8.1:** If any agent raises an exception during `generate_article()`, the exception propagates to the caller. The orchestrator does not catch or swallow agent errors.

**AC-8.2:** The batch pipeline (`scripts/generate-articles.py`) catches per-article exceptions, logs them, continues processing remaining bundles, and exits with code 1 if any articles failed.

**AC-8.3:** If the bullet parser receives non-empty markdown with no parseable bullets, it raises `ValueError` (fail-fast, not silent empty list).

## Constraints

### Technical
- All agents are called via protocol interfaces, never concrete implementations
- No LLM calls are made by the orchestrator itself — only by the delegated agents
- Feedback compilation is deterministic (no randomness, no LLM calls)
- The orchestrator is stateless between `generate_article()` calls — all state is contained within a single invocation
- All file I/O goes through `OutputHandler` and `VerboseContextLogger` — the orchestrator does not write files directly

### Configuration
- `editor_max_rounds` is read from `config.get_article_editor_max_rounds()`
- Output paths are read from config and passed to `OutputHandler` at construction time
- Agent selection (`"mock"` vs `"default"`) is read from each agent's config section
- No environment variables are used

## Edge Cases

### EC-1: Zero concerns on first review
The orchestrator returns SUCCESS immediately after the first review without entering the specialist dispatch or feedback path. The `EditorReport` contains zero `IterationReport` entries and `total_iterations == 1`.

### EC-2: Concerns found but none misleading
When specialists evaluate all concerns and none have `verdict.misleading == True`, the orchestrator returns SUCCESS. The `IterationReport` contains concerns, mappings, and verdicts, but `feedback_to_writer` is `None`.

### EC-3: All concerns have status KEEP with misleading=False
Same as EC-2 — passes. Even though KEEP verdicts produce improvement suggestions, they do not block success.

### EC-4: Concern mapping returns concern_id not in parsed concerns
The orchestrator raises `ValueError` with message `"Concern id from mapping not found: {id}"`.

### EC-5: Specialist returns verdict with suggested_fix=None and status=REWRITE
The todo_list excludes this verdict (because `suggested_fix is None`), but the verdict is still included in the full `verdicts` list within `WriterFeedback`. The rating formula still counts it as KEEP since it has no todo entry.

### EC-6: editor_max_rounds = 1
The loop runs exactly once. If misleading concerns are found, the orchestrator returns FAILED immediately without requesting a revision (since `iteration == editor_max_rounds`).

### EC-7: Writer revision produces an article with new concerns
The next iteration re-reviews the revised article from scratch. Previous iteration concerns are not carried forward — each iteration starts with a fresh review.

### EC-8: Empty source_text
The orchestrator does not validate source_text content. If the writer agent can handle empty text, the loop proceeds normally. Content validation is the writer agent's responsibility.

### EC-9: Missing metadata fields
`ValueError` is raised before any agent calls. The error message identifies the specific missing field.

## Non-Goals

- **The orchestrator does not perform editorial judgment.** It delegates all quality assessment to the review agent, mapping agent, and specialist agents.
- **The orchestrator does not modify article content.** Only the writer agent produces and revises article text.
- **The orchestrator does not run specialists in parallel.** Sequential dispatch is intentional for deterministic artifact ordering and institutional memory consistency.
- **The orchestrator does not implement retry logic for failed LLM calls.** Retries are handled within each agent's `_call_llm` method via `LLMConfig.max_retries`.
- **The orchestrator does not manage the knowledge base or Perplexity connections.** These are injected into the relevant specialist agents at construction time by the agent factory.

## Data Contracts

### Input: `generate_article()` parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source_text` | `str` | Yes | Full transcript/source text |
| `source_metadata` | `dict[str, str \| None]` | Yes | Must contain `channel_name`, `slug`, `source_file`, `video_id`, `article_title`; may contain `publish_date`, `references` (JSON string) |
| `style_mode` | `str` | Yes | `"NATURE_NEWS"` or `"SCIAM_MAGAZINE"` |
| `reader_preference` | `str` | Yes | Free-text reader angle or empty string |

### Output: `ArticleGenerationResult`

| Field | Type | Success | Failure |
|-------|------|---------|---------|
| `success` | `bool` | `True` | `False` |
| `article` | `ArticleResponse \| None` | Non-null | Best available (non-null) |
| `metadata` | `ArticleMetadata \| None` | Non-null | Non-null |
| `editor_report` | `EditorReport \| None` | Non-null | Non-null |
| `artifacts_dir` | `str \| None` | Non-null | Non-null |
| `error` | `str \| None` | `None` | Non-null error message |

### Agent Protocol Contracts

| Protocol | Method | Input | Output |
|----------|--------|-------|--------|
| `WriterAgentProtocol` | `generate()` | source_text, source_metadata, style_mode, reader_preference | `AgentResult[ArticleResponse]` |
| `WriterAgentProtocol` | `revise()` | context (JSON string), feedback (WriterFeedback) | `AgentResult[ArticleResponse]` |
| `ArticleReviewAgentProtocol` | `review()` | article, source_text, source_metadata | `AgentResult[ArticleReviewRaw]` |
| `ConcernMappingAgentProtocol` | `map_concerns()` | style_requirements, source_text, generated_article_json, concerns | `AgentResult[ConcernMappingResult]` |
| `SpecialistAgentProtocol` | `evaluate()` | concern, article, source_text, source_metadata, style_requirements | `AgentResult[Verdict]` |

### Bullet Parser Contract

| Input | Output | Error |
|-------|--------|-------|
| Empty string | `ArticleReviewResult(concerns=[])` | None |
| Non-empty with parseable bullets | `ArticleReviewResult(concerns=[Concern, ...])` | None |
| Non-empty with no parseable bullets | N/A | `ValueError` |

Bullet detection: lines starting with `- ` or `* `.
Excerpt extraction priority: curly quotes `"..."` > straight quotes `"..."` > full bullet text.
`concern_id` is assigned sequentially starting at 1 in bullet order.

### Feedback Compilation Contract

| Input | Field | Derivation |
|-------|-------|------------|
| `verdicts` | `todo_list` | `[v.suggested_fix for v in sorted_verdicts if v.status in {"REWRITE", "REMOVE"} and v.suggested_fix is not None]` |
| `verdicts` | `improvement_suggestions` | `[v.rationale for v in sorted_verdicts if v.status == "KEEP"][:5]` |
| `verdicts` | `rating` | `max(1, min(10, 10 - 2*len(todo_list) - len(improvement_suggestions)))` |
| — | `passed` | Always `False` |
| `todo_list` | `reasoning` | Empty todos: fixed string. Otherwise: numbered list. |
| `verdicts` | `verdicts` | Full sorted list (for citation passthrough) |

## Open Questions

None. This specification documents the implemented behavior.
