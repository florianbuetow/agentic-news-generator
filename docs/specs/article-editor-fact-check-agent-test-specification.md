# Fact Check Agent — Test Specification

## Coverage Matrix

| Spec Requirement | Test Scenario(s) |
|-----------------|------------------|
| AC-1.1: evaluate() returns AgentResult[Verdict] | TS-1 |
| AC-1.2: Cache hit returns cached verdict | TS-2 |
| AC-1.3: Cache miss triggers KB query with top_k=5 | TS-3 |
| AC-1.4: KB results injected as {kb_evidence} | TS-4 |
| AC-1.5: Prompt loaded via load_specialist_prompt() | TS-5 |
| AC-1.6: Prompt formatted with all variables | TS-6 |
| AC-1.7: Invalid JSON response raises ValidationError | TS-7 |
| AC-1.8: Record persisted to institutional memory | TS-8 |
| AC-2.1: Verdict fields validated | TS-9 |
| AC-4.1: Cache key includes kb_index_version | TS-10 |
| AC-4.2: Query normalization (whitespace/casing) | TS-11 |
| AC-5.1: MockFactCheckAgent implements protocol | TS-12 |
| AC-5.2: Mock returns KEEP verdict with matching concern_id | TS-13 |
| EC-1: KB returns zero results | TS-14 |
| EC-3: Cache hit from different article | TS-2 |
| EC-4: KB timeout propagates | TS-15 |
| EC-5: Missing source_file in metadata | TS-16 |

## Test Scenarios

**TS-1: evaluate() returns valid verdict on cache miss**
```
Given a FactCheckAgent with mocked LLM, mocked KB retriever, and empty institutional memory
When evaluate() is called with a concern
Then result is an AgentResult[Verdict]
And result.output has concern_id matching the input concern
```

**TS-2: Cache hit skips KB and LLM calls**
```
Given institutional memory containing a cached FactCheckRecord for the same query/model/kb_version
When evaluate() is called with a concern matching the cached query
Then result.prompt is "[cache-hit]"
And result.output is the cached verdict
And the KB retriever was NOT called
And the LLM client was NOT called
```

**TS-3: Cache miss triggers KB query with top_k=5**
```
Given empty institutional memory
And a mocked KB retriever that records calls
When evaluate() is called
Then KB retriever.search() was called with top_k=5
```

**TS-4: KB results serialized into prompt as {kb_evidence}**
```
Given a mocked KB retriever returning [{"snippet": "Evidence A", "score": 0.9}]
And a mocked LLM that captures the prompt
When evaluate() is called
Then the prompt contains "Evidence A"
And the prompt contains "0.9"
```

**TS-5: Prompt loaded via load_specialist_prompt**
```
Given a mock PromptLoader that records calls
When evaluate() is called
Then load_specialist_prompt() was called with the configured specialists_dir and prompt_file
```

**TS-6: Prompt formatted with all 6 variables**
```
Given a mock template "{style_requirements} {concern} {article_excerpt} {source_text} {source_metadata} {kb_evidence}"
When evaluate() is called with style_requirements="SCIAM_MAGAZINE" and concern with review_note="Test concern"
Then the prompt contains "SCIAM_MAGAZINE" and "Test concern"
```

**TS-7: Invalid JSON from LLM raises ValidationError**
```
Given a mocked LLM returning "Not valid JSON"
When evaluate() is called
Then ValidationError is raised
```

**TS-8: FactCheckRecord persisted after successful evaluation**
```
Given a mock institutional memory that records persist calls
When evaluate() completes successfully
Then institutional_memory.persist_fact_check() was called once
And the record contains the verdict, query, normalized_query, and kb_index_version
```

**TS-9: Verdict status validates as KEEP/REWRITE/REMOVE literal**
```
Given a mocked LLM returning status="INVALID"
When evaluate() is called
Then ValidationError is raised
```

**TS-10: Different kb_index_version causes cache miss**
```
Given institutional memory with a record for kb_index_version="v1"
And the agent is configured with kb_index_version="v2"
When evaluate() is called with the same query
Then cache miss occurs (KB and LLM are called)
```

**TS-11: Query normalization produces stable cache keys**
```
Given concern.review_note = "  HELLO   World  "
When _normalize_query() is called
Then result is "hello world"
```

**TS-12: MockFactCheckAgent implements protocol**
```
Given a MockFactCheckAgent instance
When used where SpecialistAgentProtocol is expected
Then evaluate() is callable without errors
```

**TS-13: Mock returns KEEP with matching concern_id**
```
Given a concern with concern_id=42
When MockFactCheckAgent.evaluate() is called
Then result.output.concern_id is 42
And result.output.misleading is False
And result.output.status is "KEEP"
```

**TS-14: KB returns zero results**
```
Given a mocked KB retriever returning []
And a mocked LLM that captures the prompt
When evaluate() is called
Then the prompt contains "[]" as kb_evidence
And the LLM is still called (evaluation proceeds with no evidence)
```

**TS-15: KB timeout raises exception**
```
Given a mocked KB retriever that raises TimeoutError
When evaluate() is called
Then TimeoutError propagates to the caller
```

**TS-16: Missing source_file in metadata raises ValueError**
```
Given source_metadata without "source_file" key
And a mocked LLM returning a valid verdict
When evaluate() is called
Then ValueError is raised with message about "source_file"
```

## Traceability

All acceptance criteria, cache behavior, and edge cases from the fact check behavioral specification are covered.
