# Style Review Agent — Test Specification

## Coverage Matrix

| Spec Requirement | Test Scenario(s) |
|-----------------|------------------|
| AC-1.1: evaluate() returns AgentResult[Verdict] | TS-1 |
| AC-1.2: Prompt loaded and formatted | TS-2 |
| AC-1.3: No external dependencies | TS-3 |
| AC-3.1: REWRITE verdict has specific style fix | TS-4 |
| AC-3.2: KEEP verdict explains why acceptable | TS-5 |
| AC-3.3: REMOVE is valid but rare | TS-6 |
| AC-4.1: MockStyleReviewAgent implements protocol | TS-7 |
| AC-4.2: Mock returns KEEP with matching concern_id | TS-8 |

## Test Scenarios

**TS-1: evaluate() returns valid verdict**
```
Given a StyleReviewAgent with mocked LLM returning valid Verdict JSON
When evaluate() is called with a concern
Then result is an AgentResult[Verdict]
And result.output.concern_id matches the input concern
```

**TS-2: Prompt formatted with all 5 variables**
```
Given a mock template "{style_requirements} {concern} {article_excerpt} {source_text} {source_metadata}"
And a mocked LLM that captures the prompt
When evaluate() is called with style_requirements="NATURE_NEWS"
Then the prompt contains "NATURE_NEWS"
And contains the concern, excerpt, source text, and metadata
```

**TS-3: No external service calls**
```
Given a StyleReviewAgent (no KB, no Perplexity, no institutional memory)
When evaluate() is called
Then only the LLM client was called
```

**TS-4: REWRITE verdict for hype language**
```
Given a mocked LLM returning status="REWRITE", misleading=false,
    suggested_fix="Replace 'breakthrough' with 'advance'"
When evaluate() is called
Then result.output.status is "REWRITE"
And result.output.suggested_fix contains "Replace"
```

**TS-5: KEEP verdict with explanatory rationale**
```
Given a mocked LLM returning status="KEEP",
    rationale="The hook is source-supported and appropriate for SCIAM_MAGAZINE style"
When evaluate() is called
Then result.output.status is "KEEP"
And result.output.rationale is non-empty
And result.output.suggested_fix is None
```

**TS-6: REMOVE verdict accepted**
```
Given a mocked LLM returning status="REMOVE", misleading=true
When evaluate() is called
Then result.output.status is "REMOVE"
And result.output.misleading is True
```

**TS-7: MockStyleReviewAgent implements protocol**
```
Given a MockStyleReviewAgent instance
When evaluate() is called
Then it returns without error
```

**TS-8: Mock returns KEEP with matching concern_id**
```
Given concern with concern_id=12
When MockStyleReviewAgent.evaluate() is called
Then result.output.concern_id is 12
And result.output.misleading is False
And result.output.status is "KEEP"
```

**TS-9: Invalid JSON from LLM raises ValidationError**
```
Given a mocked LLM returning "Not JSON"
When evaluate() is called
Then ValidationError is raised
```

**TS-10: Markdown-fenced JSON response handled**
```
Given a mocked LLM returning '```json\n{"concern_id":1,"misleading":false,"status":"KEEP","rationale":"Fine","suggested_fix":null,"evidence":null,"citations":null}\n```'
When evaluate() is called
Then the fences are stripped and verdict is parsed correctly
```

## Traceability

All acceptance criteria from the style review agent behavioral specification are covered. Prompt content quality (misleading vs acceptable style definitions, mode-specific criteria) is validated via integration testing.
