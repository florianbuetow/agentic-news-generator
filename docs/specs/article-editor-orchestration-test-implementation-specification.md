# Article Editor Orchestration — Test Implementation Specification

## Test Framework & Conventions

- **Framework:** pytest (project standard)
- **Style:** Class-based grouping with plain `assert` statements
- **Fixtures:** `tmp_path` for filesystem tests; helper functions for model construction
- **Mocking:** Protocol-based dependency injection — no monkey-patching needed. Test doubles implement the same protocols as real agents.
- **Naming:** `test_<behavior>` — describes what is tested, not how

## Test Structure

```
tests/
├── test_chief_editor_orchestrator.py      # TS-1 through TS-34 (this spec)
├── test_article_review_bullet_parser.py   # Already exists (extend)
├── test_mock_agents.py                    # Already exists
└── conftest_article_generation.py         # Shared fixtures (new)
```

All orchestration tests go in `tests/test_chief_editor_orchestrator.py`, grouped into classes by functional area.

## Fixtures & Test Data

### Shared Helper: `conftest_article_generation.py`

Since the project has no `conftest.py`, shared fixtures will be defined as module-level helper functions imported by each test file. This matches the existing pattern in `test_mock_agents.py`.

```python
# Helper functions (not pytest fixtures, to match project pattern)

def make_article(**overrides) -> ArticleResponse:
    """Create a minimal valid ArticleResponse."""
    defaults = {
        "headline": "Test Headline",
        "alternative_headline": "Alt Headline",
        "article_body": "Test body.",
        "description": "Test description.",
    }
    defaults.update(overrides)
    return ArticleResponse(**defaults)

def make_concern(concern_id: int = 1, excerpt: str = "test excerpt",
                 review_note: str = "test note") -> Concern:
    """Create a minimal Concern."""
    return Concern(concern_id=concern_id, excerpt=excerpt, review_note=review_note)

def make_verdict(concern_id: int = 1, misleading: bool = False,
                 status: str = "KEEP", **overrides) -> Verdict:
    """Create a minimal Verdict."""
    defaults = {
        "concern_id": concern_id,
        "misleading": misleading,
        "status": status,
        "rationale": "Test rationale",
        "suggested_fix": None,
        "evidence": None,
        "citations": None,
    }
    defaults.update(overrides)
    return Verdict(**defaults)

def make_mapping(concern_id: int = 1, selected_agent: str = "opinion",
                 **overrides) -> ConcernMapping:
    """Create a minimal ConcernMapping."""
    defaults = {
        "concern_id": concern_id,
        "concern_type": "scope_expansion",
        "selected_agent": selected_agent,
        "confidence": "high",
        "reason": "test reason",
    }
    defaults.update(overrides)
    return ConcernMapping(**defaults)

def make_source_metadata(**overrides) -> dict[str, str | None]:
    """Create valid source_metadata dict."""
    defaults = {
        "channel_name": "TestChannel",
        "slug": "test-article",
        "source_file": "transcript.txt",
        "video_id": "abc123",
        "article_title": "Test Article Title",
        "publish_date": "2025-01-01",
        "references": "[]",
    }
    defaults.update(overrides)
    return defaults
```

### Configurable Test Doubles

Each test double records calls for verification and allows configurable return values:

```python
class RecordingWriterAgent:
    """Writer that records calls and returns configurable articles."""
    def __init__(self, articles: list[ArticleResponse] | None = None):
        self.generate_calls: list[dict] = []
        self.revise_calls: list[dict] = []
        self._articles = articles or [make_article()]
        self._call_index = 0

    def generate(self, **kwargs) -> AgentResult[ArticleResponse]:
        self.generate_calls.append(kwargs)
        article = self._articles[min(self._call_index, len(self._articles) - 1)]
        self._call_index += 1
        return AgentResult(prompt="[test]", output=article)

    def revise(self, **kwargs) -> AgentResult[ArticleResponse]:
        self.revise_calls.append(kwargs)
        article = self._articles[min(self._call_index, len(self._articles) - 1)]
        self._call_index += 1
        return AgentResult(prompt="[test]", output=article)


class RecordingReviewAgent:
    """Review agent returning configurable bullet lists per call."""
    def __init__(self, responses: list[str]):
        self.review_calls: list[dict] = []
        self._responses = responses
        self._call_index = 0

    def review(self, **kwargs) -> AgentResult[ArticleReviewRaw]:
        self.review_calls.append(kwargs)
        bullets = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return AgentResult(prompt="[test]", output=ArticleReviewRaw(markdown_bullets=bullets))


class RecordingMappingAgent:
    """Mapping agent returning configurable mappings."""
    def __init__(self, mappings_per_call: list[list[ConcernMapping]]):
        self.map_calls: list[dict] = []
        self._mappings = mappings_per_call
        self._call_index = 0

    def map_concerns(self, **kwargs) -> AgentResult[ConcernMappingResult]:
        self.map_calls.append(kwargs)
        mappings = self._mappings[min(self._call_index, len(self._mappings) - 1)]
        self._call_index += 1
        return AgentResult(prompt="[test]", output=ConcernMappingResult(mappings=mappings))


class RecordingSpecialistAgent:
    """Specialist that records calls and returns configurable verdicts."""
    def __init__(self, verdicts: list[Verdict] | None = None):
        self.evaluate_calls: list[dict] = []
        self._verdicts = verdicts or [make_verdict()]
        self._call_index = 0

    def evaluate(self, **kwargs) -> AgentResult[Verdict]:
        self.evaluate_calls.append(kwargs)
        verdict = self._verdicts[min(self._call_index, len(self._verdicts) - 1)]
        self._call_index += 1
        return AgentResult(prompt="[test]", output=verdict)
```

### Orchestrator Builder

```python
def build_test_orchestrator(
    tmp_path: Path,
    *,
    writer: WriterAgentProtocol | None = None,
    review: ArticleReviewAgentProtocol | None = None,
    mapping: ConcernMappingAgentProtocol | None = None,
    fact_check: SpecialistAgentProtocol | None = None,
    evidence: SpecialistAgentProtocol | None = None,
    opinion: SpecialistAgentProtocol | None = None,
    attribution: SpecialistAgentProtocol | None = None,
    style_review: SpecialistAgentProtocol | None = None,
    editor_max_rounds: int = 3,
) -> ChiefEditorOrchestrator:
    """Build an orchestrator with test doubles and tmp_path output dirs."""
    config = _make_test_config(tmp_path, editor_max_rounds=editor_max_rounds)
    output_handler = OutputHandler(
        final_articles_dir=tmp_path / "articles",
        run_artifacts_dir=tmp_path / "runs",
    )
    return ChiefEditorOrchestrator(
        config=config,
        writer_agent=writer or MockWriterAgent(),
        article_review_agent=review or MockArticleReviewAgent(),
        concern_mapping_agent=mapping or MockConcernMappingAgent(),
        fact_check_agent=fact_check or MockFactCheckAgent(),
        evidence_finding_agent=evidence or MockEvidenceFindingAgent(),
        opinion_agent=opinion or MockOpinionAgent(),
        attribution_agent=attribution or MockAttributionAgent(),
        style_review_agent=style_review or MockStyleReviewAgent(),
        bullet_parser=ArticleReviewBulletParser(),
        institutional_memory=InstitutionalMemoryStore(
            data_dir=tmp_path / "memory",
            fact_checking_subdir="fact_checking",
            evidence_finding_subdir="evidence_finding",
        ),
        output_handler=output_handler,
    )
```

## Test Scenario Mapping

### Class: `TestOrchestratorHappyPath`

| TS | Scenario | Test Function | Setup (Given) | Action (When) | Assertion (Then) |
|----|----------|--------------|---------------|---------------|-----------------|
| TS-1 | Returns ArticleGenerationResult | `test_returns_article_generation_result` | `build_test_orchestrator` with all mocks, `make_source_metadata()` | `orchestrator.generate_article(...)` | Assert result type, success=True, article/metadata/report not None, error=None |
| TS-2 | SUCCESS zero concerns | `test_success_when_no_concerns` | `MockArticleReviewAgent` (empty bullets) | `generate_article(...)` | success=True, final_status="SUCCESS", total_iterations=1, iterations=[], blocking_concerns=None |
| TS-3 | SUCCESS no misleading | `test_success_when_no_misleading_verdicts` | `RecordingReviewAgent(["- Concern A\n- Concern B"])`, `RecordingMappingAgent` mapping both to opinion, `RecordingSpecialistAgent` with misleading=False | `generate_article(...)` | success=True, total_iterations=1, iteration has 2 concerns/mappings/verdicts, feedback_to_writer=None |

### Class: `TestOrchestratorFeedbackLoop`

| TS | Scenario | Test Function | Setup | Action | Assertion |
|----|----------|--------------|-------|--------|-----------|
| TS-4 | Feedback + revision | `test_feedback_compiled_and_revision_requested` | `RecordingWriterAgent`, `RecordingReviewAgent` [concerns, empty], mapping+specialist with misleading=True REWRITE, max_rounds=3 | `generate_article(...)` | writer called 2x (generate+revise), success=True, total_iterations=2, first iteration has feedback_to_writer with passed=False |
| TS-5 | Multi-round convergence | `test_multi_round_revision_converges` | Review returns concerns on rounds 1&2, empty on 3. Specialist misleading=True on rounds 1&2. max_rounds=3 | `generate_article(...)` | success=True, total_iterations=3, writer called 3x |
| TS-6 | FAILED after max rounds | `test_failed_after_max_rounds` | Review always returns concerns, specialist always misleading=True, max_rounds=2 | `generate_article(...)` | success=False, error contains "Unresolved", final_status="FAILED", blocking_concerns has 1 item, article not None |

### Class: `TestOrchestratorLoopMechanics`

| TS | Scenario | Test Function | Setup | Action | Assertion |
|----|----------|--------------|-------|--------|-----------|
| TS-7 | Step order | `test_loop_step_execution_order` | All recording agents, review returns 1 concern then empty, specialist misleading=True, max_rounds=2 | `generate_article(...)` | Verify call sequence: writer.generate → review → mapping → specialist → writer.revise → review |
| TS-8 | max_rounds=1 no revision | `test_single_round_fails_without_revision` | Specialist misleading=True, max_rounds=1 | `generate_article(...)` | success=False, writer called exactly once (generate, no revise) |
| TS-9 | Sequential dispatch order | `test_specialists_dispatched_in_mapping_order` | 3 concerns mapped to fact_check, opinion, style_review. 3 separate recording specialists | `generate_article(...)` | fact_check called first, opinion second, style_review third |

### Class: `TestOrchestratorSpecialistRouting`

| TS | Scenario | Test Function | Setup | Action | Assertion |
|----|----------|--------------|-------|--------|-----------|
| TS-10 | Routes to correct specialist | `test_routes_each_concern_to_correct_specialist` | 5 concerns, each mapped to a different specialist, 5 recording specialists | `generate_article(...)` | Each specialist received exactly its assigned concern |
| TS-11 | Unknown agent raises | `test_unknown_specialist_raises_value_error` | Mapping returns selected_agent="unknown_agent" | `generate_article(...)` | `pytest.raises(ValueError, match="Unknown specialist agent")` |

### Class: `TestFeedbackCompilation`

Tests for `_compile_feedback()` directly (internal method, but deterministic and critical).

| TS | Scenario | Test Function | Setup | Action | Assertion |
|----|----------|--------------|-------|--------|-----------|
| TS-12 | No LLM calls | `test_feedback_compilation_deterministic` | List of verdicts | `_compile_feedback(iteration=1, verdicts=...)` | Result is WriterFeedback; verify no LLM mock was called |
| TS-13 | Sorted by concern_id | `test_verdicts_sorted_by_concern_id` | Verdicts with ids [3,1,2] | `_compile_feedback(...)` | feedback.verdicts has ids [1,2,3] |
| TS-14 | todo_list from REWRITE/REMOVE | `test_todo_list_from_rewrite_remove` | Verdicts: REWRITE fix="A", KEEP, REMOVE fix="C" | `_compile_feedback(...)` | todo_list=["A","C"], passed=False, all 3 verdicts in feedback.verdicts |
| TS-15 | Null suggested_fix excluded | `test_null_fix_excluded_from_todo` | REWRITE fix=None, REWRITE fix="B" | `_compile_feedback(...)` | todo_list=["B"] |
| TS-16 | Suggestions capped at 5 | `test_suggestions_limited_to_five` | 7 KEEP verdicts | `_compile_feedback(...)` | len(improvement_suggestions)==5 |
| TS-17 | Rating formula | `test_rating_formula` | 2 REWRITE w/fix, 1 KEEP | `_compile_feedback(...)` | rating==5 |
| TS-18 | Rating floor at 1 | `test_rating_clamped_to_minimum` | 5 REWRITE w/fix, 3 KEEP | `_compile_feedback(...)` | rating==1 |
| TS-19 | Reasoning format | `test_reasoning_empty_todos` / `test_reasoning_with_todos` | Empty todos / 2 todos | `_compile_feedback(...)` | Fixed string / numbered list |

### Class: `TestOrchestratorArtifacts`

| TS | Scenario | Test Function | Setup | Action | Assertion |
|----|----------|--------------|-------|--------|-----------|
| TS-20 | Artifacts dir created | `test_run_artifacts_directory_created` | `build_test_orchestrator(tmp_path)` | `generate_article(...)` | Directory exists, path matches `{runs}/TestChannel/test-article/*` |
| TS-21 | Sequential numbering | `test_artifact_files_sequentially_numbered` | Full run with 1 iteration | `generate_article(...)` | Files have 3-digit prefixes, monotonically increasing, finals >= 900 |

### Class: `TestOrchestratorCanonicalOutput`

| TS | Scenario | Test Function | Setup | Action | Assertion |
|----|----------|--------------|-------|--------|-----------|
| TS-22 | Output file written on success | `test_canonical_output_written_on_success` | Successful run | `generate_article(...)` | File at `{articles}/TestChannel/test-article.json` exists, valid JSON |
| TS-23 | Output shape complete | `test_canonical_output_shape` | Successful run | Read canonical JSON | Has keys: success, article, metadata, editor_report, artifacts_dir, error. metadata.generated_at is ISO 8601 |
| TS-24 | Output written on failure | `test_canonical_output_written_on_failure` | Failed run | Read canonical JSON | success=False, error non-null, article not null, final_status="FAILED" |

### Class: `TestOrchestratorMetadataValidation`

| TS | Scenario | Test Function | Setup | Action | Assertion |
|----|----------|--------------|-------|--------|-----------|
| TS-25 | Missing channel_name | `test_missing_channel_name_raises` | metadata with channel_name=None | `generate_article(...)` | `pytest.raises(ValueError, match="channel_name")` |
| TS-26 | Missing slug | `test_missing_slug_raises` | metadata with slug=None | `generate_article(...)` | `pytest.raises(ValueError, match="slug")` |
| TS-27 | Missing source_file | `test_missing_source_file_raises` | metadata with source_file=None | `generate_article(...)` | `pytest.raises(ValueError, match="source_file")` |
| TS-28 | Null publish_date | `test_null_publish_date_accepted` | metadata with publish_date=None | `generate_article(...)` | No error, result.metadata.publish_date is None |

### Class: `TestOrchestratorDependencyInjection`

| TS | Scenario | Test Function | Setup | Action | Assertion |
|----|----------|--------------|-------|--------|-----------|
| TS-29 | Mock agents accepted | `test_accepts_mock_agents_via_protocol` | All mock agent implementations | Construct `ChiefEditorOrchestrator` | No errors, `generate_article()` succeeds |
| TS-30 | Factory creates mocks | `test_factory_creates_mock_agents` | Config with all agent_name="mock" | `build_chief_editor_orchestrator(config)` | Returns orchestrator, `generate_article()` works |
| TS-31 | Factory rejects invalid name | `test_factory_rejects_invalid_agent_name` | Config with writer agent_name="invalid" | `build_chief_editor_orchestrator(config)` | `pytest.raises(ValueError)` |

### Class: `TestOrchestratorErrorHandling`

| TS | Scenario | Test Function | Setup | Action | Assertion |
|----|----------|--------------|-------|--------|-----------|
| TS-32 | Agent error propagates | `test_agent_error_propagates` | Writer that raises RuntimeError | `generate_article(...)` | `pytest.raises(RuntimeError)` |
| TS-33 | Non-bullet review raises | `test_non_bullet_review_raises` | Review returns "plain text, no bullets" | `generate_article(...)` | `pytest.raises(ValueError)` |
| TS-34 | Unknown concern_id raises | `test_unknown_concern_id_raises` | Review returns 1 concern (id=1), mapping references id=99 | `generate_article(...)` | `pytest.raises(ValueError, match="Concern id from mapping not found: 99")` |

## Alignment Check

**Full alignment.** All 34 test scenarios from the orchestration test specification are mapped to test functions. Each test is independent (uses `tmp_path` for filesystem isolation, fresh test doubles per test). No test depends on another test's state.

**Note on initial failure:** Since the orchestrator code already exists, these tests will PASS when implemented. This is expected for the spec-dd workflow applied retroactively to existing code. The tests serve as regression coverage and behavioral documentation.
