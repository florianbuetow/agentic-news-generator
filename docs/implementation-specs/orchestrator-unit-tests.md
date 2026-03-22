# Implementation Spec: Orchestrator Unit Tests (34 Scenarios)

## References

Read ALL of the following files FIRST before writing any code:

1. `CLAUDE.md` -- project conventions (root of repo)
2. `docs/specs/article-editor-orchestration-specification.md` -- behavioral spec
3. `docs/specs/article-editor-orchestration-test-specification.md` -- 34 Given/When/Then scenarios
4. `docs/specs/article-editor-orchestration-test-implementation-specification.md` -- exact test function mapping, fixtures, test doubles
5. `src/agents/article_generation/chief_editor/orchestrator.py` -- the code under test
6. `src/agents/article_generation/base.py` -- agent protocols
7. `src/agents/article_generation/models.py` -- all Pydantic models
8. `src/agents/article_generation/chief_editor/bullet_parser.py` -- bullet parser
9. `src/agents/article_generation/chief_editor/output_handler.py` -- output handler
10. `src/agents/article_generation/chief_editor/institutional_memory.py` -- institutional memory store
11. `src/agents/article_generation/chief_editor/verbose_context_logger.py` -- verbose context logger
12. `src/agents/article_generation/agent.py` -- agent factory (build_chief_editor_orchestrator)
13. `tests/test_mock_agents.py` -- existing test patterns
14. `tests/test_agent_factory.py` -- existing config/factory test patterns (especially `_write_test_config`)
15. `src/config.py` -- Config class (especially `get_article_editor_max_rounds`)

## Task Type

Test-only. The orchestrator is fully implemented. Write tests that verify the existing code. All 34 tests should PASS.

## Constraints

- Python executed only via `uv run` -- never raw `python` or `pip install`
- Never use `git add -A` or `git add .`
- Never include AI attribution in commits
- All test code goes in `tests/test_chief_editor_orchestrator.py`
- Follow existing test patterns from `test_mock_agents.py` and `test_agent_factory.py`: class-based grouping, plain `assert`, helper functions
- Use `tmp_path` for filesystem isolation
- Do NOT create `conftest_article_generation.py` -- put all helpers directly in the test file

## Phase 1: Create the test file

Create `tests/test_chief_editor_orchestrator.py` with:

### Module-level helpers (top of file)

1. `make_article(**overrides) -> ArticleResponse` -- creates minimal valid article
2. `make_concern(concern_id, excerpt, review_note) -> Concern` -- creates minimal concern
3. `make_verdict(concern_id, misleading, status, **overrides) -> Verdict` -- creates minimal verdict
4. `make_mapping(concern_id, selected_agent, **overrides) -> ConcernMapping` -- creates minimal mapping
5. `make_source_metadata(**overrides) -> dict` -- creates valid source_metadata dict with all required keys

### Recording test doubles

These record calls and return configurable responses:

1. `RecordingWriterAgent` -- records `generate_calls` and `revise_calls`, returns configurable `ArticleResponse` list
2. `RecordingReviewAgent` -- records `review_calls`, returns configurable bullet strings per call
3. `RecordingMappingAgent` -- records `map_calls`, returns configurable `ConcernMapping` lists per call
4. `RecordingSpecialistAgent` -- records `evaluate_calls`, returns configurable `Verdict` list per call

Each test double:
- Stores calls in a list of dicts (the kwargs)
- Has a `_call_index` counter
- Returns `self._items[min(self._call_index, len(self._items) - 1)]` and increments counter
- Returns `AgentResult(prompt="[test]", output=...)`

### Test orchestrator builder

`_make_test_config(tmp_path, editor_max_rounds) -> Config` -- follow the pattern from `test_agent_factory.py`'s `_write_test_config` but with all agents set to "mock" and a configurable `editor_max_rounds`.

`build_test_orchestrator(tmp_path, *, writer=None, review=None, mapping=None, fact_check=None, evidence=None, opinion=None, attribution=None, style_review=None, editor_max_rounds=3) -> ChiefEditorOrchestrator`

This creates a `ChiefEditorOrchestrator` with:
- A Config from `_make_test_config`
- An `OutputHandler` with `final_articles_dir=tmp_path / "articles"` and `run_artifacts_dir=tmp_path / "runs"`
- An `ArticleReviewBulletParser()` (real one, not mocked)
- An `InstitutionalMemoryStore` with `data_dir=tmp_path / "memory"`
- All agent slots filled by the provided test doubles, or by the project's existing mock agents (from `src/agents/article_generation/*/mock_agent.py`) as defaults

IMPORTANT: When using the project's existing mock agents as defaults, import them from their actual locations:
- `MockWriterAgent` from `src.agents.article_generation.writer.mock_agent`
- `MockArticleReviewAgent` from `src.agents.article_generation.article_review.mock_agent`
- `MockConcernMappingAgent` from `src.agents.article_generation.concern_mapping.mock_agent`
- `MockFactCheckAgent` from `src.agents.article_generation.specialists.fact_check.mock_agent`
- `MockEvidenceFindingAgent` from `src.agents.article_generation.specialists.evidence_finding.mock_agent`
- `MockOpinionAgent` from `src.agents.article_generation.specialists.opinion.mock_agent`
- `MockAttributionAgent` from `src.agents.article_generation.specialists.attribution.mock_agent`
- `MockStyleReviewAgent` from `src.agents.article_generation.specialists.style_review.mock_agent`

### Test classes and functions

Implement ALL 34 test functions organized into the exact classes specified in the test implementation spec. Here is the complete list:

**TestOrchestratorHappyPath**
- `test_returns_article_generation_result` (TS-1)
- `test_success_when_no_concerns` (TS-2)
- `test_success_when_no_misleading_verdicts` (TS-3)

**TestOrchestratorFeedbackLoop**
- `test_feedback_compiled_and_revision_requested` (TS-4)
- `test_multi_round_revision_converges` (TS-5)
- `test_failed_after_max_rounds` (TS-6)

**TestOrchestratorLoopMechanics**
- `test_loop_step_execution_order` (TS-7)
- `test_single_round_fails_without_revision` (TS-8)
- `test_specialists_dispatched_in_mapping_order` (TS-9)

**TestOrchestratorSpecialistRouting**
- `test_routes_each_concern_to_correct_specialist` (TS-10)
- `test_unknown_specialist_raises_value_error` (TS-11)

**TestFeedbackCompilation**
- `test_feedback_compilation_deterministic` (TS-12)
- `test_verdicts_sorted_by_concern_id` (TS-13)
- `test_todo_list_from_rewrite_remove` (TS-14)
- `test_null_fix_excluded_from_todo` (TS-15)
- `test_suggestions_limited_to_five` (TS-16)
- `test_rating_formula` (TS-17)
- `test_rating_clamped_to_minimum` (TS-18)
- `test_reasoning_empty_todos` (TS-19a)
- `test_reasoning_with_todos` (TS-19b)

**TestOrchestratorArtifacts**
- `test_run_artifacts_directory_created` (TS-20)
- `test_artifact_files_sequentially_numbered` (TS-21)

**TestOrchestratorCanonicalOutput**
- `test_canonical_output_written_on_success` (TS-22)
- `test_canonical_output_shape` (TS-23)
- `test_canonical_output_written_on_failure` (TS-24)

**TestOrchestratorMetadataValidation**
- `test_missing_channel_name_raises` (TS-25)
- `test_missing_slug_raises` (TS-26)
- `test_missing_source_file_raises` (TS-27)
- `test_null_publish_date_accepted` (TS-28)

**TestOrchestratorDependencyInjection**
- `test_accepts_mock_agents_via_protocol` (TS-29)
- `test_factory_creates_mock_agents` (TS-30)
- `test_factory_rejects_invalid_agent_name` (TS-31)

**TestOrchestratorErrorHandling**
- `test_agent_error_propagates` (TS-32)
- `test_non_bullet_review_raises` (TS-33)
- `test_unknown_concern_id_raises` (TS-34)

### Key implementation details for specific tests

**TS-3 (no misleading verdicts):** Use `RecordingReviewAgent(["- Concern A\n- Concern B"])` which returns a 2-bullet review. Use `RecordingMappingAgent` that maps both concerns to "opinion". Use a single `RecordingSpecialistAgent` with `misleading=False`. Wire the specialist as the `opinion` parameter. Verify: iteration report has 2 concerns, 2 mappings, 2 verdicts, `feedback_to_writer is None`.

**TS-4 (feedback + revision):** Use `RecordingReviewAgent(["- Concern A", ""])` -- first call returns 1 concern, second returns empty. Mapping maps concern to "opinion". Specialist returns `misleading=True, status="REWRITE", suggested_fix="Fix this"`. Writer accepts revisions. After the run: writer called 2x (1 generate + 1 revise), success=True, total_iterations=2.

**TS-5 (multi-round):** Review returns concerns on calls 1 and 2, empty on call 3. Specialist returns misleading=True on calls 1 and 2. max_rounds=3. Result: success=True, total_iterations=3, writer called 3x.

**TS-6 (failed max rounds):** Review always returns 1 concern. Specialist always returns misleading=True, status=REWRITE. max_rounds=2. Result: success=False, error contains "Unresolved", blocking_concerns has 1 item.

**TS-7 (step order):** Use a shared `call_order` list. Create wrapper test doubles that append to this list before delegating. Verify exact sequence.

**TS-9 (dispatch order):** 3 concerns mapped to fact_check, opinion, style_review (in that order). Create 3 separate RecordingSpecialistAgents. Verify fact_check called first, opinion second, style_review third.

**TS-10 (routing):** 5 concerns, each mapped to a different specialist. 5 separate RecordingSpecialistAgents. Verify each received exactly its concern.

**TS-11 (unknown agent):** Mapping returns `selected_agent="unknown_agent"`. Use `pytest.raises(ValueError, match="Unknown specialist agent")`.

**TS-12 through TS-19 (feedback compilation):** Call `orchestrator._compile_feedback(iteration=1, verdicts=...)` directly. These test the internal method.

**TS-19:** This scenario specifies two sub-tests: one for empty todos (reasoning = "No required rewrites or removals were identified.") and one with todos (reasoning = "Required changes:\n1. Fix A\n2. Fix B"). Implement as two separate test functions: `test_reasoning_empty_todos` and `test_reasoning_with_todos`.

**TS-20 (artifacts dir):** After `generate_article()`, check that a directory exists under `{runs}/TestChannel/test-article/` and that the run_id matches the pattern `YYYYMMDDTHHMMSSZ_{8_hex_chars}`.

**TS-21 (sequential numbering):** After a full run with 1 iteration, check that artifact files have 3-digit prefixes that increase monotonically, and finals >= 900.

**TS-22-24 (canonical output):** Check files at `{articles}/TestChannel/test-article.json`.

**TS-25-27 (missing metadata):** Use `pytest.raises(ValueError, match="channel_name")` etc.

**TS-29 (DI):** Construct with all mock agents, call `generate_article()`, verify it succeeds.

**TS-30 (factory mocks):** Use `_write_test_config` from `test_agent_factory.py` pattern to create config with all agents set to "mock", call `build_chief_editor_orchestrator`, then call `generate_article()`.

**TS-31 (factory invalid):** Use config with `writer` agent_name="invalid", `pytest.raises(ValueError)`.

**TS-32 (error propagation):** Create a writer that raises `RuntimeError("LLM failed")` on generate. Use `pytest.raises(RuntimeError)`.

**TS-33 (non-bullet review):** RecordingReviewAgent that returns `"This is not a bullet list"`. Use `pytest.raises(ValueError)`.

**TS-34 (unknown concern_id):** Review returns 1 concern (id=1). Mapping references concern_id=99. Use `pytest.raises(ValueError, match="Concern id from mapping not found: 99")`.

## Phase 2: Run the tests

Run: `uv run pytest tests/test_chief_editor_orchestrator.py -v`

All 34 tests must pass. If any fail, diagnose and fix. Do not modify source code -- only fix the test code.

## Phase 3: Run full test suite

Run: `uv run pytest tests/ -v -m "not e2e"`

Ensure no regressions. All existing tests plus the 34 new tests should pass.

## Verification

After all phases:
1. `uv run pytest tests/test_chief_editor_orchestrator.py -v` -- all 34 pass
2. `uv run pytest tests/ -v -m "not e2e"` -- all tests pass, no regressions
3. `uv run ruff check tests/test_chief_editor_orchestrator.py` -- no lint errors
4. `uv run ruff format --check tests/test_chief_editor_orchestrator.py` -- no format issues

## Debrief

When done, report:
1. Anything you did not implement
2. Anything you skipped
3. Any place where you decided something was unnecessary
4. Exactly which tests/checks you ran
Then say "IMPLEMENTATION COMPLETE".
