# Evidence Finding Agent — Test Specification

## Coverage Matrix

| Spec Requirement | Test Scenario(s) |
|-----------------|------------------|
| AC-1.1: evaluate() returns AgentResult[Verdict] | TS-1 |
| AC-1.2: Cache hit skips Perplexity and LLM | TS-2 |
| AC-1.3: Cache miss calls perplexity_client.search() | TS-3 |
| AC-1.4: Citations extracted from response | TS-4, TS-5 |
| AC-1.5: Perplexity response injected as {web_evidence} | TS-6 |
| AC-1.7: Record persisted with citations | TS-7 |
| AC-2.1: Verdict includes evidence and citations fields | TS-8 |
| AC-4.1: Cache key does NOT include kb_index_version | TS-9 |
| AC-4.4: EvidenceRecord includes perplexity_response and citations | TS-7 |
| AC-5.1: MockEvidenceFindingAgent implements protocol | TS-10 |
| AC-5.2: Mock returns KEEP with matching concern_id | TS-11 |
| EC-1: Perplexity returns no citations | TS-5 |
| EC-2: Perplexity timeout propagates | TS-12 |
| EC-6: Non-URL citation strings passed through | TS-13 |

## Test Scenarios

**TS-1: evaluate() returns valid verdict on cache miss**
```
Given an EvidenceFindingAgent with mocked LLM, mocked Perplexity, empty institutional memory
When evaluate() is called with a concern
Then result is an AgentResult[Verdict]
And result.output has concern_id matching the input concern
```

**TS-2: Cache hit skips Perplexity and LLM**
```
Given institutional memory containing a cached EvidenceRecord for the same query/model
When evaluate() is called with a concern matching the cached query
Then result.prompt is "[cache-hit]"
And the Perplexity client was NOT called
And the LLM client was NOT called
```

**TS-3: Cache miss triggers Perplexity search**
```
Given empty institutional memory
And a mocked Perplexity client that records calls
When evaluate() is called
Then perplexity_client.search() was called with the concern's review_note
```

**TS-4: Citations extracted from response with citations array**
```
Given Perplexity response {"citations": ["https://example.com/a", "https://example.com/b"], "content": "..."}
When _extract_citations() is called
Then result is ["https://example.com/a", "https://example.com/b"]
```

**TS-5: Citations empty when response has no citations key**
```
Given Perplexity response {"content": "Some answer"} (no citations key)
When _extract_citations() is called
Then result is []
```

**TS-6: Perplexity response serialized into prompt**
```
Given a mocked Perplexity returning {"content": "Web evidence", "citations": ["url1"]}
And a mocked LLM that captures the prompt
When evaluate() is called
Then the prompt contains "Web evidence"
And the prompt contains "url1"
```

**TS-7: EvidenceRecord persisted with citations**
```
Given a mock institutional memory that records persist calls
And Perplexity returns citations ["url1", "url2"]
When evaluate() completes successfully
Then institutional_memory.persist_evidence() was called
And the record.citations is ["url1", "url2"]
And the record.perplexity_response contains the full serialized response
```

**TS-8: Verdict can include evidence and citations**
```
Given a mocked LLM returning verdict with evidence="Found support" and citations=["url1"]
When evaluate() is called
Then result.output.evidence is "Found support"
And result.output.citations is ["url1"]
```

**TS-9: Cache key excludes kb_index_version**
```
Given same query and model but different kb_index_version values
When cache lookup is performed
Then the same cache key hash is produced (kb_index_version not in key)
```

**TS-10: MockEvidenceFindingAgent implements protocol**
```
Given a MockEvidenceFindingAgent instance
When evaluate() is called
Then it returns without error
```

**TS-11: Mock returns KEEP with matching concern_id**
```
Given concern with concern_id=7
When MockEvidenceFindingAgent.evaluate() is called
Then result.output.concern_id is 7
And result.output.misleading is False
And result.output.status is "KEEP"
```

**TS-12: Perplexity timeout propagates**
```
Given a mocked Perplexity client that raises TimeoutError
When evaluate() is called
Then TimeoutError propagates
```

**TS-13: Non-URL citation strings preserved**
```
Given Perplexity response with citations: ["https://valid.url", "Not a URL", 42]
When _extract_citations() is called
Then result is ["https://valid.url", "Not a URL"] (strings kept, non-strings filtered)
```

## Traceability

All acceptance criteria and edge cases from the evidence finding behavioral specification are covered.
