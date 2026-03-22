# Article Editor Agents — Test Implementation Specification

Covers test implementation for all 8 agent test specifications:
- Writer Agent (19 scenarios)
- Article Review Agent (24 scenarios, including bullet parser)
- Concern Mapping Agent (13 scenarios)
- Fact Check Agent (16 scenarios)
- Evidence Finding Agent (13 scenarios)
- Opinion Agent (11 scenarios)
- Attribution Agent (10 scenarios)
- Style Review Agent (10 scenarios)

## Test Framework & Conventions

- **Framework:** pytest with class-based grouping
- **Assertions:** Plain `assert` statements
- **Mocking:** Protocol-based test doubles (same pattern as orchestration spec). For agents that call `_call_llm()`, inject a `RecordingLLMClient` that returns configurable responses.
- **No monkey-patching:** All dependencies are constructor-injected
- **Naming:** `test_<behavior>`

## Test Structure

```
tests/
├── test_writer_agent.py                   # Writer agent unit tests (new)
├── test_article_review_agent.py           # Review agent unit tests (new)
├── test_article_review_bullet_parser.py   # Already exists — extend
├── test_concern_mapping_agent.py          # Concern mapping unit tests (new)
├── test_fact_check_agent.py               # Fact check unit tests (new)
├── test_evidence_finding_agent.py         # Evidence finding unit tests (new)
├── test_opinion_agent.py                  # Opinion agent unit tests (new)
├── test_attribution_agent.py              # Attribution agent unit tests (new)
├── test_style_review_agent.py             # Style review agent unit tests (new)
├── test_mock_agents.py                    # Already exists — covers mock protocol tests
```

## Fixtures & Test Data

### RecordingLLMClient

All real agents use `_call_llm()` which delegates to an injected `LLMClient`. The test double:

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
```

### RecordingPromptLoader

```python
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

### RecordingKBRetriever

```python
class RecordingKBRetriever:
    """KB retriever returning configurable results."""
    def __init__(self, results: list[dict[str, str]] | None = None):
        self.search_calls: list[dict] = []
        self._results = results or []

    def search(self, *, query: str, top_k: int, timeout_seconds: int) -> list[dict[str, str]]:
        self.search_calls.append({"query": query, "top_k": top_k, "timeout_seconds": timeout_seconds})
        return self._results
```

### RecordingPerplexityClient

```python
class RecordingPerplexityClient:
    """Perplexity client returning configurable search results."""
    def __init__(self, response: dict[str, object] | None = None):
        self.search_calls: list[dict] = []
        self._response = response or {"content": "test", "citations": []}

    def search(self, *, query: str, model: str, timeout_seconds: int) -> dict[str, object]:
        self.search_calls.append({"query": query, "model": model, "timeout_seconds": timeout_seconds})
        return self._response
```

### RecordingInstitutionalMemory

```python
class RecordingInstitutionalMemory:
    """Institutional memory that records lookups/persists and returns configurable cache hits."""
    def __init__(self, *, fact_check_hit=None, evidence_hit=None):
        self.fact_check_lookups: list[dict] = []
        self.evidence_lookups: list[dict] = []
        self.fact_check_persists: list = []
        self.evidence_persists: list = []
        self._fact_check_hit = fact_check_hit
        self._evidence_hit = evidence_hit

    def lookup_fact_check(self, **kwargs): self.fact_check_lookups.append(kwargs); return self._fact_check_hit
    def lookup_evidence(self, **kwargs): self.evidence_lookups.append(kwargs); return self._evidence_hit
    def persist_fact_check(self, *, record): self.fact_check_persists.append(record)
    def persist_evidence(self, *, record): self.evidence_persists.append(record)
    def build_fact_check_cache_key_hash(self, **kwargs): return "testcachekey1234"
    def build_evidence_cache_key_hash(self, **kwargs): return "testcachekey5678"
```

### Standard Valid Verdict JSON

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

### Minimal Test Config

```python
def make_test_llm_config(**overrides) -> LLMConfig:
    defaults = {
        "model": "test-model", "api_base": "http://localhost:1234/v1",
        "api_key": "test", "context_window": 32768, "max_tokens": 2048,
        "temperature": 0.3, "context_window_threshold": 90,
        "max_retries": 0, "retry_delay": 0, "timeout_seconds": 60,
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)
```

## Test Scenario Mapping — Writer Agent

File: `tests/test_writer_agent.py`

| TS | Test Function | Setup | Action | Assertion |
|----|--------------|-------|--------|-----------|
| TS-1 | `test_generate_returns_valid_result` | `RecordingLLMClient([VALID_ARTICLE_JSON])`, `RecordingPromptLoader` | `agent.generate(...)` | result is `AgentResult[ArticleResponse]`, all fields non-empty |
| TS-2 | `test_missing_fields_raises_validation_error` | LLM returns `{"headline":"X","article_body":"Y"}` | `agent.generate(...)` | `pytest.raises(ValidationError)` |
| TS-3 | `test_prompt_loaded_via_loader` | `RecordingPromptLoader` | `agent.generate(...)` | `loader.load_calls == ["writer.md"]` |
| TS-4 | `test_template_formatted_with_variables` | Template `"{style_mode} {reader_preference} {source_text} {source_metadata}"`, LLM captures prompt | `agent.generate(style_mode="NATURE_NEWS", ...)` | Prompt contains "NATURE_NEWS" and source text |
| TS-5 | `test_single_user_message_sent` | `RecordingLLMClient` | `agent.generate(...)` | `len(llm.calls)==1`, `messages[0]["role"]=="user"` |
| TS-6 | `test_non_json_response_raises` | LLM returns `"Not JSON"` | `agent.generate(...)` | `pytest.raises(ValidationError)` |
| TS-7 | `test_extra_fields_raise_validation_error` | LLM returns valid JSON + `"author":"X"` | `agent.generate(...)` | `pytest.raises(ValidationError)` |
| TS-8 | `test_prompt_field_contains_assembled_text` | Template `"Template: {source_text}"` | `agent.generate(source_text="Hello")` | `result.prompt == "Template: Hello"` |
| TS-9 | `test_revise_returns_valid_result` | `RecordingLLMClient([VALID_ARTICLE_JSON])` | `agent.revise(context="...", feedback=...)` | result is valid `AgentResult[ArticleResponse]` |
| TS-10 | `test_revision_template_formatted_with_feedback` | Template with feedback vars, `RecordingLLMClient` captures prompt | `agent.revise(feedback=WriterFeedback(...))` | Prompt contains rating, todos, suggestions, verdicts JSON, context |
| TS-11 | `test_mock_satisfies_protocol` | `MockWriterAgent()` | call generate() and revise() | No errors |
| TS-12 | `test_mock_generate_returns_static` | `MockWriterAgent()` | `generate(...)` | `prompt=="[mock]"`, headline contains "Mock" |
| TS-13 | `test_mock_revise_returns_same_as_generate` | `MockWriterAgent()` | `revise(...)` | Output identical to `generate()` output |
| TS-14 | `test_writer_prompt_contains_rules` | Read `prompts/article_editor/writer.md` | Check content | Contains "JSON", "source text", "NATURE_NEWS", "SCIAM_MAGAZINE" |
| TS-15 | `test_writer_prompt_contains_target_length` | Read prompt file | Check content | Contains "900" or "1200" |
| TS-16 | `test_empty_reader_preference` | reader_preference="" | `agent.generate(...)` | No error, prompt assembled |
| TS-17 | `test_missing_required_json_field` | LLM returns `{"headline":"X","article_body":"Y"}` | `agent.generate(...)` | `pytest.raises(ValidationError)` |
| TS-18 | `test_long_source_text_proceeds` | 100k chars source_text | `agent.generate(...)` | No exception from token validation |
| TS-19 | `test_null_metadata_values_serialized` | metadata with `publish_date=None` | `agent.generate(...)` | No error |

## Test Scenario Mapping — Review Agent + Bullet Parser

File: `tests/test_article_review_agent.py` (new) + extend `tests/test_article_review_bullet_parser.py`

**Review Agent tests** (`test_article_review_agent.py`):

| TS | Test Function | Setup | Action | Assertion |
|----|--------------|-------|--------|-----------|
| TS-1 | `test_review_returns_raw_bullets` | LLM returns `"- Concern A"` | `agent.review(...)` | result.output.markdown_bullets == `"- Concern A"` |
| TS-2 | `test_response_preserved_as_is` | LLM returns `"- A\n- B"` | `agent.review(...)` | markdown_bullets is exact response |
| TS-3 | `test_empty_response_signals_no_concerns` | LLM returns `""` | `agent.review(...)` | markdown_bullets == `""` |
| TS-4 | `test_prompt_loaded_via_loader` | `RecordingPromptLoader` | `agent.review(...)` | Loader called with correct file |
| TS-5 | `test_template_formatted_with_variables` | Template with vars | `agent.review(...)` | Prompt contains source, metadata, article |
| TS-6 | `test_response_stripped` | LLM returns `"  \n- A\n  "` | `agent.review(...)` | markdown_bullets == `"- A"` |
| TS-7 | `test_prompt_references_article_and_source` | Read prompt file | Check content | Contains "source" and "article" |
| TS-8 | `test_no_json_parsing_attempted` | LLM returns `'{"key":"val"}'` | `agent.review(...)` | markdown_bullets is raw JSON string |
| TS-19 | `test_mock_satisfies_protocol` | `MockArticleReviewAgent()` | `review(...)` | No errors |
| TS-20 | `test_mock_returns_empty` | `MockArticleReviewAgent()` | `review(...)` | markdown_bullets=="" |

**Bullet Parser tests** (extend `test_article_review_bullet_parser.py` — some already exist):

| TS | Test Function | Exists? |
|----|--------------|---------|
| TS-9 | `test_parse_empty_output` | YES — already exists |
| TS-10 | `test_whitespace_only_returns_empty` | Add variant |
| TS-11 | `test_dash_and_asterisk_bullets` | Add new |
| TS-12 | `test_multiline_continuation` | YES — exists partially |
| TS-13 | `test_concern_ids_sequential` | Add new |
| TS-14 | `test_curly_quotes_highest_priority` | YES — exists |
| TS-15 | `test_straight_quotes_fallback` | YES — exists |
| TS-16 | `test_no_quotes_uses_full_text` | YES — exists |
| TS-17 | `test_review_note_is_full_bullet` | Add new |
| TS-18 | `test_non_bullet_text_raises` | YES — exists |
| TS-21 | `test_numbered_list_raises` | Add new |
| TS-22 | `test_preamble_before_bullets_ignored` | Add new |
| TS-23 | `test_nested_sub_bullets_as_continuation` | Add new |
| TS-24 | `test_formatting_preserved` | Add new |

## Test Scenario Mapping — Concern Mapping Agent

File: `tests/test_concern_mapping_agent.py`

| TS | Test Function | Setup | Action | Assertion |
|----|--------------|-------|--------|-----------|
| TS-1 | `test_returns_valid_mappings` | LLM returns valid mapping JSON | `agent.map_concerns(...)` | result has correct mappings |
| TS-2 | `test_invalid_concern_type_raises` | LLM returns `concern_type="invalid"` | `map_concerns(...)` | `pytest.raises(ValidationError)` |
| TS-3 | `test_invalid_selected_agent_raises` | LLM returns `selected_agent="nonexistent"` | `map_concerns(...)` | `pytest.raises(ValidationError)` |
| TS-4 | `test_json_array_format` | LLM returns `[{...}]` | `map_concerns(...)` | Parsed correctly |
| TS-5 | `test_json_object_format` | LLM returns `{"mappings":[{...}]}` | `map_concerns(...)` | Parsed correctly |
| TS-6 | `test_array_non_object_items_raises` | LLM returns `[1,2,3]` | `map_concerns(...)` | `pytest.raises(ValueError)` |
| TS-7 | `test_mock_satisfies_protocol` | `MockConcernMappingAgent()` | `map_concerns(...)` | No errors |
| TS-8 | `test_mock_returns_empty` | Mock | `map_concerns(...)` | mappings==[] |
| TS-9 | `test_single_concern_mapped` | 1 concern, LLM returns 1 mapping | `map_concerns(...)` | 1 mapping in result |
| TS-10 | `test_invalid_agent_literal_rejected` | LLM returns `selected_agent="sentiment"` | `map_concerns(...)` | `pytest.raises(ValidationError)` |
| TS-11 | `test_markdown_fenced_object_stripped` | LLM returns `` ```json\n{...}\n``` `` | `map_concerns(...)` | Parsed correctly |
| TS-12 | `test_empty_array_accepted` | LLM returns `[]` | `map_concerns(...)` | mappings==[] |
| TS-13 | `test_prompt_formatted_with_context` | Template with vars, RecordingLLMClient | `map_concerns(...)` | Prompt contains style, source, article, concerns |

## Test Scenario Mapping — Specialist Agents

### Fact Check Agent (`tests/test_fact_check_agent.py`)

| TS | Test Function | Setup | Action | Assertion |
|----|--------------|-------|--------|-----------|
| TS-1 | `test_returns_verdict_on_cache_miss` | Empty memory, KB results, LLM returns verdict | `evaluate(...)` | Valid verdict returned |
| TS-2 | `test_cache_hit_skips_kb_and_llm` | Pre-populated memory | `evaluate(...)` | prompt=="[cache-hit]", KB not called, LLM not called |
| TS-3 | `test_kb_queried_with_top_k_5` | `RecordingKBRetriever` | `evaluate(...)` | search called with top_k=5 |
| TS-4 | `test_kb_results_in_prompt` | KB returns `[{"snippet":"evidence"}]`, RecordingLLMClient | `evaluate(...)` | Prompt contains "evidence" |
| TS-5 | `test_prompt_loaded_via_specialist_loader` | `RecordingPromptLoader` | `evaluate(...)` | `load_specialist_calls` has entry |
| TS-6 | `test_prompt_has_all_variables` | Template with 6 vars | `evaluate(...)` | Prompt contains all values |
| TS-7 | `test_invalid_json_raises` | LLM returns "not json" | `evaluate(...)` | `pytest.raises(ValidationError)` |
| TS-8 | `test_record_persisted` | `RecordingInstitutionalMemory` | `evaluate(...)` | fact_check_persists has 1 record |
| TS-9 | `test_invalid_status_raises` | LLM returns status="INVALID" | `evaluate(...)` | `pytest.raises(ValidationError)` |
| TS-10 | `test_different_kb_version_cache_miss` | Memory has record for v1, agent configured v2 | `evaluate(...)` | Cache miss, KB queried |
| TS-11 | `test_query_normalization` | concern with `"  HELLO   World  "` | `_normalize_query(...)` | returns `"hello world"` |
| TS-12 | `test_mock_implements_protocol` | `MockFactCheckAgent()` | `evaluate(...)` | No errors |
| TS-13 | `test_mock_returns_keep_with_matching_id` | concern_id=42 | Mock `evaluate(...)` | verdict.concern_id==42, status=="KEEP" |
| TS-14 | `test_kb_returns_empty` | KB returns [] | `evaluate(...)` | LLM still called, prompt has "[]" |
| TS-15 | `test_kb_timeout_propagates` | KB raises TimeoutError | `evaluate(...)` | `pytest.raises(TimeoutError)` |
| TS-16 | `test_missing_source_file_raises` | metadata without source_file | `evaluate(...)` | `pytest.raises(ValueError)` |

### Evidence Finding Agent (`tests/test_evidence_finding_agent.py`)

| TS | Test Function | Setup | Action | Assertion |
|----|--------------|-------|--------|-----------|
| TS-1 | `test_returns_verdict_on_cache_miss` | Empty memory, Perplexity response, LLM verdict | `evaluate(...)` | Valid verdict |
| TS-2 | `test_cache_hit_skips_perplexity_and_llm` | Pre-populated memory | `evaluate(...)` | prompt=="[cache-hit]" |
| TS-3 | `test_perplexity_called_on_miss` | `RecordingPerplexityClient` | `evaluate(...)` | search called |
| TS-4 | `test_citations_extracted_from_list` | Response with citations=["url1","url2"] | `_extract_citations(...)` | ["url1","url2"] |
| TS-5 | `test_citations_empty_when_no_key` | Response without citations key | `_extract_citations(...)` | [] |
| TS-6 | `test_perplexity_response_in_prompt` | RecordingLLMClient | `evaluate(...)` | Prompt contains web evidence |
| TS-7 | `test_record_persisted_with_citations` | `RecordingInstitutionalMemory` | `evaluate(...)` | evidence_persists has record with citations |
| TS-8 | `test_verdict_includes_evidence_fields` | LLM returns verdict with evidence+citations | `evaluate(...)` | evidence and citations populated |
| TS-9 | `test_cache_key_excludes_kb_version` | Same query, different kb contexts | Check cache hash | Same hash produced |
| TS-10 | `test_mock_implements_protocol` | Mock | `evaluate(...)` | No errors |
| TS-11 | `test_mock_returns_keep` | Mock, concern_id=7 | `evaluate(...)` | concern_id==7, KEEP |
| TS-12 | `test_perplexity_timeout_propagates` | Perplexity raises TimeoutError | `evaluate(...)` | `pytest.raises(TimeoutError)` |
| TS-13 | `test_non_string_citations_filtered` | citations=["url", 42, None] | `_extract_citations(...)` | ["url"] |

### Opinion / Attribution / Style Review Agents

These three share identical code structure. Each gets its own test file but the test patterns are the same.

Files: `tests/test_opinion_agent.py`, `tests/test_attribution_agent.py`, `tests/test_style_review_agent.py`

For each agent, the mapping follows this template (replacing Agent class and prompt file):

| TS | Test Function | Setup | Action | Assertion |
|----|--------------|-------|--------|-----------|
| TS-1 | `test_returns_valid_verdict` | LLM returns valid verdict JSON | `evaluate(...)` | Valid AgentResult[Verdict] |
| TS-2 | `test_prompt_loaded_via_specialist_loader` | RecordingPromptLoader | `evaluate(...)` | load_specialist_calls recorded |
| TS-3 | `test_prompt_formatted_with_variables` | Template with 5 vars, RecordingLLMClient | `evaluate(...)` | Prompt contains all values |
| TS-4 | `test_rewrite_verdict_parsed` | LLM returns REWRITE verdict | `evaluate(...)` | status=="REWRITE", suggested_fix present |
| TS-5 | `test_keep_verdict_parsed` | LLM returns KEEP, fix=null | `evaluate(...)` | status=="KEEP", fix==None |
| TS-6 | `test_remove_verdict_parsed` | LLM returns REMOVE | `evaluate(...)` | status=="REMOVE" |
| TS-7 | `test_mock_implements_protocol` | Mock agent | `evaluate(...)` | No errors |
| TS-8 | `test_mock_returns_keep_with_matching_id` | Mock, concern_id=N | `evaluate(...)` | concern_id matches |
| TS-9 | `test_invalid_json_raises` | LLM returns "not json" | `evaluate(...)` | `pytest.raises(ValidationError)` |
| TS-10 | `test_markdown_fenced_json_handled` | LLM returns fenced verdict | `evaluate(...)` | Parsed correctly |
| TS-11 | `test_no_external_service_calls` | Only LLM mock configured | `evaluate(...)` | Only LLM called |

## Alignment Check

**Full alignment.** All 116 agent test scenarios (19 + 24 + 13 + 16 + 13 + 11 + 10 + 10) are mapped to test functions with setup, action, and assertion strategies defined. Combined with the 34 orchestration scenarios, all 150 test scenarios are accounted for.

**Existing coverage:** 3 bullet parser tests already exist (TS-9, TS-12 partial, TS-18). 8 mock agent tests already exist (covering TS-12/13 mock patterns). These will be extended rather than duplicated.

**Test isolation:** Each test constructs its own agent instance with fresh test doubles. No shared mutable state between tests. `tmp_path` provides filesystem isolation where needed.
