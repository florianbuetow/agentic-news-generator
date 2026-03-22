# Attribution Agent — Test Specification

## Coverage Matrix

| Spec Requirement | Test Scenario(s) |
|-----------------|------------------|
| AC-1.1: evaluate() returns AgentResult[Verdict] | TS-1 |
| AC-1.2: Prompt loaded and formatted | TS-2 |
| AC-1.3: No external dependencies | TS-3 |
| AC-3.1: REWRITE for missing attribution has fix | TS-4 |
| AC-3.2: REWRITE for attribution inflation has fix | TS-5 |
| AC-3.4: KEEP has null suggested_fix | TS-6 |
| AC-4.1: MockAttributionAgent implements protocol | TS-7 |
| AC-4.2: Mock returns KEEP with matching concern_id | TS-8 |

## Test Scenarios

**TS-1: evaluate() returns valid verdict**
```
Given an AttributionAgent with mocked LLM returning valid Verdict JSON
When evaluate() is called with a concern
Then result is an AgentResult[Verdict]
And result.output.concern_id matches the input concern
```

**TS-2: Prompt formatted with all 5 variables**
```
Given a mock template "{style_requirements} {concern} {article_excerpt} {source_text} {source_metadata}"
And a mocked LLM that captures the prompt
When evaluate() is called
Then the prompt contains all 5 formatted values
```

**TS-3: No external service calls**
```
Given an AttributionAgent (no KB, no Perplexity, no institutional memory)
When evaluate() is called
Then only the LLM client was called
```

**TS-4: REWRITE verdict for missing attribution**
```
Given a mocked LLM returning status="REWRITE", misleading=true,
    suggested_fix="Add attribution: 'According to Altman, ...'"
When evaluate() is called
Then result.output.status is "REWRITE"
And result.output.misleading is True
And result.output.suggested_fix contains "attribution"
```

**TS-5: REWRITE verdict for attribution inflation**
```
Given a mocked LLM returning status="REWRITE",
    suggested_fix="Replace 'confirmed' with 'suggested'"
When evaluate() is called
Then result.output.suggested_fix contains "suggested"
```

**TS-6: KEEP verdict has null suggested_fix**
```
Given a mocked LLM returning status="KEEP", suggested_fix=null
When evaluate() is called
Then result.output.suggested_fix is None
```

**TS-7: MockAttributionAgent implements protocol**
```
Given a MockAttributionAgent instance
When evaluate() is called
Then it returns without error
```

**TS-8: Mock returns KEEP with matching concern_id**
```
Given concern with concern_id=3
When MockAttributionAgent.evaluate() is called
Then result.output.concern_id is 3
And result.output.misleading is False
And result.output.status is "KEEP"
```

**TS-9: Invalid JSON from LLM raises ValidationError**
```
Given a mocked LLM returning "Not JSON"
When evaluate() is called
Then ValidationError is raised
```

**TS-10: Verdict with extra fields rejected**
```
Given a mocked LLM returning valid Verdict JSON plus extra field "source_support"
When evaluate() is called
Then ValidationError is raised (Verdict uses extra="forbid")
```

## Traceability

All acceptance criteria from the attribution agent behavioral specification are covered. Prompt content quality (attribution pattern detection, tracing instructions) is validated via integration testing.
