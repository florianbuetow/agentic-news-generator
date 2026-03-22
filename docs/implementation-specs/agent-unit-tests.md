# Implementation Spec: Agent Unit Tests (116 Scenarios across 9 Test Files)

## References

Read ALL of the following files FIRST before writing any code:

1. `CLAUDE.md` -- project conventions (root of repo)
2. `docs/specs/article-editor-agents-test-implementation-specification.md` -- THE MAIN REFERENCE: exact test function mappings, fixture classes, test doubles for ALL agents
3. `docs/specs/article-editor-writer-agent-test-specification.md` -- Writer agent Given/When/Then scenarios
4. `docs/specs/article-editor-review-agent-test-specification.md` -- Review agent Given/When/Then scenarios
5. `docs/specs/article-editor-concern-mapping-agent-test-specification.md` -- Concern mapping Given/When/Then scenarios
6. `docs/specs/article-editor-fact-check-agent-test-specification.md` -- Fact check Given/When/Then scenarios
7. `docs/specs/article-editor-evidence-finding-agent-test-specification.md` -- Evidence finding Given/When/Then scenarios
8. `docs/specs/article-editor-opinion-agent-test-specification.md` -- Opinion Given/When/Then scenarios
9. `docs/specs/article-editor-attribution-agent-test-specification.md` -- Attribution Given/When/Then scenarios
10. `docs/specs/article-editor-style-review-agent-test-specification.md` -- Style review Given/When/Then scenarios
11. `src/agents/article_generation/base.py` -- BaseAgent, BaseSpecialistAgent, protocols, _parse_json_response, _normalize_query, _build_article_id
12. `src/agents/article_generation/models.py` -- ALL Pydantic models (AgentResult, ArticleResponse, Verdict, Concern, etc.)
13. `src/agents/article_generation/writer/agent.py` -- WriterAgent implementation
14. `src/agents/article_generation/article_review/agent.py` -- ArticleReviewAgent implementation
15. `src/agents/article_generation/concern_mapping/agent.py` -- ConcernMappingAgent implementation
16. `src/agents/article_generation/specialists/fact_check/agent.py` -- FactCheckAgent implementation
17. `src/agents/article_generation/specialists/evidence_finding/agent.py` -- EvidenceFindingAgent implementation
18. `src/agents/article_generation/specialists/opinion/agent.py` -- OpinionAgent implementation
19. `src/agents/article_generation/specialists/attribution/agent.py` -- AttributionAgent implementation
20. `src/agents/article_generation/specialists/style_review/agent.py` -- StyleReviewAgent implementation
21. `src/agents/article_generation/writer/mock_agent.py` -- MockWriterAgent
22. `src/agents/article_generation/article_review/mock_agent.py` -- MockArticleReviewAgent
23. `src/agents/article_generation/concern_mapping/mock_agent.py` -- MockConcernMappingAgent
24. `src/agents/article_generation/specialists/fact_check/mock_agent.py` -- MockFactCheckAgent
25. `src/agents/article_generation/specialists/evidence_finding/mock_agent.py` -- MockEvidenceFindingAgent
26. `src/agents/article_generation/specialists/opinion/mock_agent.py` -- MockOpinionAgent
27. `src/agents/article_generation/specialists/attribution/mock_agent.py` -- MockAttributionAgent
28. `src/agents/article_generation/specialists/style_review/mock_agent.py` -- MockStyleReviewAgent
29. `src/agents/article_generation/prompts/loader.py` -- PromptLoader class
30. `src/agents/article_generation/chief_editor/bullet_parser.py` -- ArticleReviewBulletParser
31. `src/agents/article_generation/chief_editor/institutional_memory.py` -- InstitutionalMemoryStore
32. `src/config.py` -- Config class, LLMConfig class
33. `tests/test_mock_agents.py` -- existing mock agent tests (DO NOT duplicate these)
34. `tests/test_article_review_bullet_parser.py` -- existing bullet parser tests (EXTEND, do not duplicate)
35. `tests/test_chief_editor_orchestrator.py` -- reference for test patterns, test doubles, helpers

## Task Type

Test-only. All agents are fully implemented. Write tests that verify the existing agent code. All tests should PASS.

## Constraints

- Python executed only via `uv run` -- never raw `python` or `pip install`
- Never use `git add -A` or `git add .`
- Never include AI attribution in commits
- Follow existing test patterns: class-based grouping, plain `assert`, module-level helper functions
- Put all helpers (recording test doubles, factory functions) directly in each test file -- do NOT create shared conftest files
- Use `tmp_path` for filesystem isolation where needed
- Do NOT modify any source code in `src/` -- only create/modify test files
- Do NOT duplicate tests that already exist in `tests/test_mock_agents.py`

## Shared Patterns

### Recording Test Doubles

Every test file that tests a real agent needs these. Copy them into each file that uses them:

```python
class RecordingLLMClient:
    """LLM client that returns configurable responses and records calls."""
    def __init__(self, responses: list[str]):
        self.calls: list[dict] = []
        self._responses = responses
        self._call_index = 0

    def complete(self, *, llm_config, messages) -> str:
        self.calls.append({"llm_config": llm_config, "messages": messages})
        response = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return response


class RecordingPromptLoader:
    """PromptLoader that returns configurable templates and records calls."""
    def __init__(self, template: str = "{source_text}"):
        self.load_calls: list[str] = []
        self.load_specialist_calls: list[tuple[str, str]] = []
        self._template = template

    def load_prompt(self, *, prompt_file: str) -> str:
        self.load_calls.append(prompt_file)
        return self._template

    def load_specialist_prompt(self, *, specialists_dir: str, prompt_file: str) -> str:
        self.load_specialist_calls.append((specialists_dir, prompt_file))
        return self._template
```

### Minimal Test Config and LLMConfig

Each test file that constructs real agents needs a Config and LLMConfig. Use this pattern:

```python
def make_test_llm_config(**overrides) -> LLMConfig:
    defaults = {
        "model": "test-model", "api_base": "http://localhost:1234/v1",
        "api_key": "test", "context_window": 32768, "max_tokens": 2048,
        "temperature": 0.3, "context_window_threshold": 90,
        "max_retries": 0, "retry_delay": 1, "timeout_seconds": 60,
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)
```

For Config, copy the `_write_test_config` pattern from `tests/test_agent_factory.py`. Write a minimal `config.yaml` to `tmp_path`, then construct `Config(config_path)`.

### Standard Valid JSON Payloads

```python
VALID_VERDICT_JSON = json.dumps({
    "concern_id": 1,
    "misleading": False,
    "status": "KEEP",
    "rationale": "Test rationale",
    "suggested_fix": None,
    "evidence": None,
    "citations": None,
})

VALID_ARTICLE_JSON = json.dumps({
    "headline": "Test",
    "alternative_headline": "Alt",
    "article_body": "Body text",
    "description": "Desc",
})
```

### Sample Data Helpers

```python
def make_test_concern(concern_id: int = 1) -> Concern:
    return Concern(concern_id=concern_id, excerpt="test excerpt", review_note="test review note")

def make_test_article() -> ArticleResponse:
    return ArticleResponse(
        headline="Test", alternative_headline="Alt",
        article_body="Body", description="Desc",
    )

def make_test_metadata() -> dict[str, str | None]:
    return {
        "channel_name": "TestChannel",
        "slug": "test-slug",
        "source_file": "test.txt",
        "video_id": "vid-1",
        "article_title": "Test",
        "publish_date": "2025-01-01",
        "references": "[]",
        "topic_slug": "test-topic",
    }
```

## Phase 1: Writer Agent Tests (19 tests)

File: `tests/test_writer_agent.py`

### Required imports

```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pytest
import yaml
from pydantic import ValidationError
from src.agents.article_generation.writer.agent import WriterAgent
from src.agents.article_generation.writer.mock_agent import MockWriterAgent
from src.agents.article_generation.models import AgentResult, ArticleResponse, Verdict, WriterFeedback
from src.config import Config, LLMConfig
```

### Required test doubles in this file

- `RecordingLLMClient`
- `RecordingPromptLoader`
- `make_test_llm_config`
- `_write_test_config` (copy from test_agent_factory.py pattern)
- `VALID_ARTICLE_JSON`
- `make_test_metadata`

### Test class: `TestWriterAgent`

Implement these 19 test functions. Refer to the test implementation spec for exact setup/action/assertion per TS:

1. `test_generate_returns_valid_result` (TS-1): Create WriterAgent with RecordingLLMClient([VALID_ARTICLE_JSON]), RecordingPromptLoader. Call agent.generate() with valid args. Assert result is AgentResult, output fields non-empty.

2. `test_missing_fields_raises_validation_error` (TS-2): LLM returns JSON with only headline and article_body (missing alternative_headline, description). Assert pytest.raises(ValidationError).

3. `test_prompt_loaded_via_loader` (TS-3): Check loader.load_calls == ["writer.md"].

4. `test_template_formatted_with_variables` (TS-4): Use template "{style_mode} {reader_preference} {source_text} {source_metadata}". Call generate with style_mode="NATURE_NEWS". Assert prompt in LLM call contains "NATURE_NEWS" and source text.

5. `test_single_user_message_sent` (TS-5): Assert len(llm.calls)==1, messages[0]["role"]=="user".

6. `test_non_json_response_raises` (TS-6): LLM returns "Not JSON". Assert pytest.raises(ValidationError).

7. `test_extra_fields_raise_validation_error` (TS-7): LLM returns valid JSON + "author":"X". Assert pytest.raises(ValidationError).

8. `test_prompt_field_contains_assembled_text` (TS-8): Template "Template: {source_text}". Assert result.prompt == "Template: Hello" when source_text="Hello".

9. `test_revise_returns_valid_result` (TS-9): Call agent.revise() with WriterFeedback. Assert valid AgentResult.

10. `test_revision_template_formatted_with_feedback` (TS-10): Template with feedback vars: "{rating} {pass_status} {reasoning} {todo_list} {improvement_suggestions} {verdicts} {context}". Use RecordingLLMClient, call revise with feedback containing rating=3, todo_list=["Fix A", "Fix B"], etc. Assert prompt contains "3", "False", the todo items, the suggestions, verdicts JSON, and context.

11. `test_mock_satisfies_protocol` (TS-11): MockWriterAgent().generate() and .revise() both callable, no errors.

12. `test_mock_generate_returns_static` (TS-12): MockWriterAgent().generate(), assert prompt=="[mock]", headline contains "Mock".

13. `test_mock_revise_returns_same_as_generate` (TS-13): Both outputs identical.

14. `test_writer_prompt_contains_rules` (TS-14): Read the actual file `prompts/article_editor/writer.md`. Assert it contains "JSON", "source text" (case-insensitive), "NATURE_NEWS", "SCIAM_MAGAZINE". Use `Path(__file__).resolve().parent.parent / "prompts" / "article_editor" / "writer.md"` to locate the file.

15. `test_writer_prompt_contains_target_length` (TS-15): Same file. Assert contains "900" or "1200".

16. `test_empty_reader_preference` (TS-16): reader_preference="", no error.

17. `test_missing_required_json_field` (TS-17): Same as TS-2 (headline+article_body only). pytest.raises(ValidationError).

18. `test_long_source_text_proceeds` (TS-18): 100k chars source_text. No exception.

19. `test_null_metadata_values_serialized` (TS-19): metadata with publish_date=None. No error.

### Key implementation detail for Writer

WriterAgent constructor requires: llm_config, config, llm_client, prompt_loader, writer_prompt_file, revision_prompt_file. Example:

```python
agent = WriterAgent(
    llm_config=make_test_llm_config(),
    config=Config(config_path),
    llm_client=llm,
    prompt_loader=loader,
    writer_prompt_file="writer.md",
    revision_prompt_file="revision.md",
)
```

The `generate()` method signature: `generate(*, source_text, source_metadata, style_mode, reader_preference)`.
The `revise()` method signature: `revise(*, context, feedback)`.

For tests requiring Config, write a minimal config.yaml to tmp_path using the `_write_test_config` helper from test_agent_factory.py. Include all required paths and defaults sections.

## Phase 2: Article Review Agent Tests (10 tests)

File: `tests/test_article_review_agent.py`

### Required imports

```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pytest
import yaml
from src.agents.article_generation.article_review.agent import ArticleReviewAgent
from src.agents.article_generation.article_review.mock_agent import MockArticleReviewAgent
from src.agents.article_generation.models import AgentResult, ArticleResponse, ArticleReviewRaw
from src.config import Config, LLMConfig
```

### Required test doubles

- `RecordingLLMClient`
- `RecordingPromptLoader`
- `make_test_llm_config`
- `_write_test_config`
- `make_test_article`
- `make_test_metadata`

### Test class: `TestArticleReviewAgent`

10 test functions:

1. `test_review_returns_raw_bullets` (TS-1): LLM returns "- Concern A". Assert result.output.markdown_bullets == "- Concern A".

2. `test_response_preserved_as_is` (TS-2): LLM returns "- A\n- B". Assert markdown_bullets == "- A\n- B".

3. `test_empty_response_signals_no_concerns` (TS-3): LLM returns "". Assert markdown_bullets == "".

4. `test_prompt_loaded_via_loader` (TS-4): Check loader called with correct file.

5. `test_template_formatted_with_variables` (TS-5): Template "{source_text} {source_metadata} {generated_article}". Assert prompt contains source_text, serialized metadata, serialized article.

6. `test_response_stripped` (TS-6): LLM returns "  \n- A\n  ". Assert markdown_bullets == "- A".

7. `test_prompt_references_article_and_source` (TS-7): Read actual `prompts/article_editor/article_review.md`. Assert it references article and source text.

8. `test_no_json_parsing_attempted` (TS-8): LLM returns '{"key":"val"}'. Assert markdown_bullets is raw JSON string.

9. `test_mock_satisfies_protocol` (TS-19): MockArticleReviewAgent().review() callable.

10. `test_mock_returns_empty` (TS-20): Mock review, assert markdown_bullets=="" and prompt=="[mock]".

### Key implementation detail

ArticleReviewAgent constructor: llm_config, config, llm_client, prompt_loader, prompt_file.
The `review()` method: `review(*, article, source_text, source_metadata)`. article is ArticleResponse.

## Phase 3: Extend Bullet Parser Tests

File: `tests/test_article_review_bullet_parser.py` (EXTEND existing file)

### Tests to ADD (do NOT duplicate existing ones)

The file already has:
- `test_parse_empty_output` (TS-9) -- exists
- `test_parse_with_multiline_bullets_and_excerpt_extraction` (TS-12 partial) -- exists
- `test_non_empty_without_bullets_raises` (TS-18) -- exists

Add these NEW tests inside the existing `TestArticleReviewBulletParser` class:

1. `test_whitespace_only_returns_empty` (TS-10): markdown_bullets="  \n\t\n  ". Assert concerns == [].

2. `test_dash_and_asterisk_bullets` (TS-11): "- First\n* Second". Assert 2 concerns, review_note values are "First" and "Second".

3. `test_concern_ids_sequential` (TS-13): 5 bullets. Assert concern_ids == [1, 2, 3, 4, 5].

4. `test_review_note_is_full_bullet` (TS-17): '- **"Excerpt"** -- explanation'. Assert review_note == '**"Excerpt"** -- explanation' and excerpt == "Excerpt".

5. `test_numbered_list_raises` (TS-21): "1. First\n2. Second". pytest.raises(ValueError).

6. `test_preamble_before_bullets_ignored` (TS-22): "Here are concerns:\n\n- First\n- Second". Assert 2 concerns, preamble not in any concern.

7. `test_nested_sub_bullets_as_continuation` (TS-23): "- Parent\n  - Sub one\n  - Sub two\n- Next". Assert 2 concerns. Concern 1 review_note contains "Parent", "Sub one", "Sub two". Concern 2 review_note == "Next".

8. `test_formatting_preserved` (TS-24): '- **Bold** and *italic* text'. Assert review_note == "**Bold** and *italic* text".

## Phase 4: Concern Mapping Agent Tests (13 tests)

File: `tests/test_concern_mapping_agent.py`

### Required test doubles

- `RecordingLLMClient`
- `RecordingPromptLoader`
- `make_test_llm_config`
- `_write_test_config`
- `make_test_concern`

### Test class: `TestConcernMappingAgent`

13 test functions:

1. `test_returns_valid_mappings` (TS-1): LLM returns valid JSON array with 2 mappings. Assert result.output.mappings has 2 items with correct fields.

2. `test_invalid_concern_type_raises` (TS-2): LLM returns concern_type="invalid_type". pytest.raises(ValidationError).

3. `test_invalid_selected_agent_raises` (TS-3): selected_agent="nonexistent". pytest.raises(ValidationError).

4. `test_json_array_format` (TS-4): LLM returns '[{...}]'. Parsed correctly.

5. `test_json_object_format` (TS-5): LLM returns '{"mappings":[{...}]}'. Parsed correctly.

6. `test_array_non_object_items_raises` (TS-6): '[1,2,3]'. pytest.raises(ValueError).

7. `test_mock_satisfies_protocol` (TS-7): MockConcernMappingAgent callable.

8. `test_mock_returns_empty` (TS-8): Mock returns mappings==[].

9. `test_single_concern_mapped` (TS-9): 1 concern, 1 mapping returned.

10. `test_invalid_agent_literal_rejected` (TS-10): selected_agent="sentiment". pytest.raises(ValidationError).

11. `test_markdown_fenced_object_stripped` (TS-11): '```json\n{"mappings":[...]}\n```'. Parsed correctly.

12. `test_empty_array_accepted` (TS-12): '[]'. mappings==[].

13. `test_prompt_formatted_with_context` (TS-13): Template with vars. Assert prompt contains style, source, article, concerns.

### Key implementation detail

ConcernMappingAgent constructor: llm_config, config, llm_client, prompt_loader, prompt_file.
The `map_concerns()` method: `map_concerns(*, style_requirements, source_text, generated_article_json, concerns)`.

A valid mapping JSON for tests:
```python
VALID_MAPPING_JSON = json.dumps({
    "concern_id": 1,
    "concern_type": "scope_expansion",
    "selected_agent": "opinion",
    "confidence": "high",
    "reason": "test reason",
})
```

## Phase 5: Fact Check Agent Tests (16 tests)

File: `tests/test_fact_check_agent.py`

### Required test doubles

- `RecordingLLMClient`
- `RecordingPromptLoader`
- `RecordingKBRetriever`
- `RecordingInstitutionalMemory`
- `make_test_llm_config`
- `_write_test_config`
- `VALID_VERDICT_JSON`
- `make_test_concern`
- `make_test_article`
- `make_test_metadata`

### Test class: `TestFactCheckAgent`

16 test functions:

1. `test_returns_verdict_on_cache_miss` (TS-1): Empty memory, KB results, LLM returns verdict. Assert valid verdict.

2. `test_cache_hit_skips_kb_and_llm` (TS-2): Pre-populated memory (fact_check_hit set). Assert prompt=="[cache-hit]", KB not called, LLM not called.

3. `test_kb_queried_with_top_k_5` (TS-3): Assert search called with top_k=5.

4. `test_kb_results_in_prompt` (TS-4): KB returns [{"snippet":"evidence"}]. Assert "evidence" in prompt.

5. `test_prompt_loaded_via_specialist_loader` (TS-5): Assert load_specialist_calls recorded.

6. `test_prompt_has_all_variables` (TS-6): Template "{style_requirements} {concern} {article_excerpt} {source_text} {source_metadata} {kb_evidence}". Assert all values present.

7. `test_invalid_json_raises` (TS-7): LLM returns "not json". pytest.raises(ValidationError).

8. `test_record_persisted` (TS-8): Assert fact_check_persists has 1 record.

9. `test_invalid_status_raises` (TS-9): status="INVALID". pytest.raises(ValidationError).

10. `test_different_kb_version_cache_miss` (TS-10): Memory has v1 record, agent configured v2. Assert KB queried (cache miss).

11. `test_query_normalization` (TS-11): "  HELLO   World  " -> "hello world". Call _normalize_query directly on agent.

12. `test_mock_implements_protocol` (TS-12): MockFactCheckAgent callable.

13. `test_mock_returns_keep_with_matching_id` (TS-13): concern_id=42 -> verdict.concern_id==42, status=="KEEP".

14. `test_kb_returns_empty` (TS-14): KB returns []. LLM still called, prompt has "[]".

15. `test_kb_timeout_propagates` (TS-15): KB raises TimeoutError. pytest.raises(TimeoutError).

16. `test_missing_source_file_raises` (TS-16): metadata without source_file. pytest.raises(ValueError).

### Key implementation detail

FactCheckAgent constructor: llm_config, config, llm_client, prompt_loader, specialists_dir, prompt_file, knowledge_base_retriever, institutional_memory, kb_index_version, kb_timeout_seconds.

The `evaluate()` method: `evaluate(*, concern, article, source_text, source_metadata, style_requirements)`.

For the RecordingInstitutionalMemory: when fact_check_hit is set, lookup_fact_check returns it. The hit should be a FactCheckRecord with a valid Verdict inside:

```python
from src.agents.article_generation.models import FactCheckRecord, Verdict

cached_verdict = Verdict(concern_id=1, misleading=False, status="KEEP", rationale="cached", suggested_fix=None, evidence=None, citations=None)
cached_record = FactCheckRecord(
    timestamp="2025-01-01T00:00:00Z", article_id="test:test", concern_id=1,
    prompt="cached prompt", query="test query", normalized_query="test query",
    model_name="test-model", kb_index_version="v1", cache_key_hash="hash1",
    kb_response="[]", verdict=cached_verdict,
)
memory = RecordingInstitutionalMemory(fact_check_hit=cached_record)
```

For TS-15 (KB timeout), create a custom KB retriever class that raises TimeoutError on search:
```python
class TimeoutKBRetriever:
    def search(self, *, query, top_k, timeout_seconds):
        raise TimeoutError("KB timed out")
```

For TS-16 (missing source_file), use metadata without "source_file" key. The error is raised by `_build_article_id()` in BaseSpecialistAgent.

IMPORTANT: The metadata dict must also contain "topic_slug" when source_file is present. The `_build_article_id` method checks both. When testing missing source_file, include topic_slug but omit source_file.

## Phase 6: Evidence Finding Agent Tests (13 tests)

File: `tests/test_evidence_finding_agent.py`

### Required test doubles

- `RecordingLLMClient`
- `RecordingPromptLoader`
- `RecordingPerplexityClient`
- `RecordingInstitutionalMemory`
- `make_test_llm_config`
- `_write_test_config`
- `VALID_VERDICT_JSON`
- `make_test_concern`
- `make_test_article`
- `make_test_metadata`

### Test class: `TestEvidenceFindingAgent`

13 test functions:

1. `test_returns_verdict_on_cache_miss` (TS-1): Empty memory, Perplexity returns data, LLM returns verdict. Assert valid result.

2. `test_cache_hit_skips_perplexity_and_llm` (TS-2): Pre-populated memory (evidence_hit set). Assert prompt=="[cache-hit]", Perplexity not called, LLM not called.

3. `test_perplexity_called_on_miss` (TS-3): Assert search_calls has entry.

4. `test_citations_extracted_from_list` (TS-4): response with citations=["url1","url2"]. Call agent._extract_citations() directly. Assert ["url1","url2"].

5. `test_citations_empty_when_no_key` (TS-5): response without citations key. Call _extract_citations(). Assert [].

6. `test_perplexity_response_in_prompt` (TS-6): Assert prompt contains web evidence content.

7. `test_record_persisted_with_citations` (TS-7): Assert evidence_persists has record with citations.

8. `test_verdict_includes_evidence_fields` (TS-8): LLM returns verdict with evidence and citations. Assert populated.

9. `test_cache_key_excludes_kb_version` (TS-9): Check build_evidence_cache_key_hash doesn't include kb_index_version. Call memory.build_evidence_cache_key_hash with same query/model. It should produce same hash regardless of other params.

10. `test_mock_implements_protocol` (TS-10): MockEvidenceFindingAgent callable.

11. `test_mock_returns_keep` (TS-11): concern_id=7, assert verdict matches.

12. `test_perplexity_timeout_propagates` (TS-12): Perplexity raises TimeoutError. pytest.raises(TimeoutError).

13. `test_non_string_citations_filtered` (TS-13): citations=["url", 42, None]. Call _extract_citations(). Assert ["url"].

### Key implementation detail

EvidenceFindingAgent constructor: llm_config, config, llm_client, prompt_loader, specialists_dir, prompt_file, perplexity_client, perplexity_model, institutional_memory.

For the RecordingInstitutionalMemory with evidence_hit:
```python
from src.agents.article_generation.models import EvidenceRecord, Verdict

cached_verdict = Verdict(concern_id=1, misleading=False, status="KEEP", rationale="cached", suggested_fix=None, evidence=None, citations=None)
cached_record = EvidenceRecord(
    timestamp="2025-01-01T00:00:00Z", article_id="test:test", concern_id=1,
    prompt="cached prompt", query="test query", normalized_query="test query",
    model_name="test-model", cache_key_hash="hash1",
    perplexity_response="{}", citations=[], verdict=cached_verdict,
)
memory = RecordingInstitutionalMemory(evidence_hit=cached_record)
```

For TS-12 (timeout), create TimeoutPerplexityClient:
```python
class TimeoutPerplexityClient:
    def search(self, *, query, model, timeout_seconds):
        raise TimeoutError("Perplexity timed out")
```

The _extract_citations method signature: `_extract_citations(self, *, search_response: dict[str, object]) -> list[str]`. Call it as `agent._extract_citations(search_response=...)`.

## Phase 7: Opinion Agent Tests (11 tests)

File: `tests/test_opinion_agent.py`

### Test class: `TestOpinionAgent`

11 test functions following the specialist template from the implementation spec:

1. `test_returns_valid_verdict` (TS-1)
2. `test_prompt_loaded_via_specialist_loader` (TS-2)
3. `test_prompt_formatted_with_variables` (TS-3): Template "{style_requirements} {concern} {article_excerpt} {source_text} {source_metadata}". Assert all values in prompt.
4. `test_rewrite_verdict_parsed` (TS-4): status="REWRITE", suggested_fix present.
5. `test_keep_verdict_parsed` (TS-5): status="KEEP", fix=None.
6. `test_remove_verdict_parsed` (TS-6): status="REMOVE".
7. `test_mock_implements_protocol` (TS-7)
8. `test_mock_returns_keep_with_matching_id` (TS-8)
9. `test_invalid_json_raises` (TS-9): "not json" -> ValidationError.
10. `test_markdown_fenced_json_handled` (TS-10): '```json\n{...}\n```' parsed correctly.
11. `test_no_external_service_calls` (TS-11): Only LLM called, no KB/Perplexity/memory.

### Key implementation detail

OpinionAgent constructor: llm_config, config, llm_client, prompt_loader, specialists_dir, prompt_file.
The `evaluate()` method: same as all specialists.

For TS-11 (no external service calls): construct OpinionAgent (it doesn't take KB/Perplexity/memory args). Call evaluate(). Assert llm.calls has 1 entry. The absence of KB/Perplexity is structural -- the agent doesn't accept those parameters.

## Phase 8: Attribution Agent Tests (10 tests)

File: `tests/test_attribution_agent.py`

### Test class: `TestAttributionAgent`

10 test functions following the specialist template. The implementation spec maps these as:

1. `test_returns_valid_verdict` (TS-1)
2. `test_prompt_formatted_with_variables` (TS-2): Same 5-var template.
3. `test_no_external_service_calls` (TS-3): Only LLM called.
4. `test_rewrite_verdict_parsed` (TS-4): REWRITE with misleading=True, fix contains "attribution".
5. `test_keep_verdict_parsed` (TS-5): REWRITE with fix containing "suggested" (attribution inflation).
6. `test_keep_null_fix` (TS-6): KEEP with suggested_fix=None.
7. `test_mock_implements_protocol` (TS-7)
8. `test_mock_returns_keep_with_matching_id` (TS-8)
9. `test_invalid_json_raises` (TS-9)
10. `test_extra_fields_rejected` (TS-10): Verdict JSON + extra field "source_support" -> ValidationError (Verdict uses extra="forbid").

IMPORTANT: TS-4 and TS-5 in the attribution spec test REWRITE verdicts (not keep/rewrite sequence). TS-4 tests missing attribution (misleading=True), TS-5 tests attribution inflation. TS-6 tests KEEP. Read the attribution test specification carefully for the exact assertions.

### Key implementation detail

AttributionAgent constructor: llm_config, config, llm_client, prompt_loader, specialists_dir, prompt_file.

## Phase 9: Style Review Agent Tests (10 tests)

File: `tests/test_style_review_agent.py`

### Test class: `TestStyleReviewAgent`

10 test functions following the specialist template:

1. `test_returns_valid_verdict` (TS-1)
2. `test_prompt_formatted_with_variables` (TS-2): 5-var template.
3. `test_no_external_service_calls` (TS-3): Only LLM.
4. `test_rewrite_verdict_parsed` (TS-4): REWRITE, fix contains "Replace".
5. `test_keep_verdict_parsed` (TS-5): KEEP, non-empty rationale, fix=None.
6. `test_remove_verdict_parsed` (TS-6): REMOVE, misleading=True.
7. `test_mock_implements_protocol` (TS-7)
8. `test_mock_returns_keep_with_matching_id` (TS-8)
9. `test_invalid_json_raises` (TS-9)
10. `test_markdown_fenced_json_handled` (TS-10): Fenced JSON parsed correctly.

### Key implementation detail

StyleReviewAgent constructor: llm_config, config, llm_client, prompt_loader, specialists_dir, prompt_file.

## Verification

After ALL phases:

1. `uv run pytest tests/test_writer_agent.py tests/test_article_review_agent.py tests/test_concern_mapping_agent.py tests/test_fact_check_agent.py tests/test_evidence_finding_agent.py tests/test_opinion_agent.py tests/test_attribution_agent.py tests/test_style_review_agent.py tests/test_article_review_bullet_parser.py -v` -- all tests pass
2. `uv run pytest tests/ -v -m "not e2e"` -- all tests pass, no regressions
3. `uv run ruff check tests/test_writer_agent.py tests/test_article_review_agent.py tests/test_concern_mapping_agent.py tests/test_fact_check_agent.py tests/test_evidence_finding_agent.py tests/test_opinion_agent.py tests/test_attribution_agent.py tests/test_style_review_agent.py` -- no lint errors
4. `uv run ruff format --check tests/test_writer_agent.py tests/test_article_review_agent.py tests/test_concern_mapping_agent.py tests/test_fact_check_agent.py tests/test_evidence_finding_agent.py tests/test_opinion_agent.py tests/test_attribution_agent.py tests/test_style_review_agent.py` -- no format issues

## Debrief

When done, report:
1. Anything you did not implement
2. Anything you skipped
3. Any place where you decided something was unnecessary
4. Exactly which tests/checks you ran
Then say "IMPLEMENTATION COMPLETE".
