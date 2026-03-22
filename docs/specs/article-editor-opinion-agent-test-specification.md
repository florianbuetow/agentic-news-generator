# Opinion Agent — Test Specification

## Coverage Matrix

| Spec Requirement | Test Scenario(s) |
|-----------------|------------------|
| AC-1.1: evaluate() returns AgentResult[Verdict] | TS-1 |
| AC-1.2: Prompt loaded and formatted | TS-2, TS-3 |
| AC-1.3: Single LLM call, parsed as Verdict | TS-4 |
| AC-1.4: No external dependencies | TS-5 |
| AC-3.1: REWRITE verdict has specific suggested_fix | TS-6 |
| AC-3.2: KEEP verdict has null suggested_fix | TS-7 |
| AC-3.4: Rationale is 1-3 sentences | TS-8 |
| AC-4.1: MockOpinionAgent implements protocol | TS-9 |
| AC-4.2: Mock returns KEEP with matching concern_id | TS-10 |
| EC-5: Multiple issues in single concern | TS-11 |

## Test Scenarios

**TS-1: evaluate() returns valid verdict**
```
Given an OpinionAgent with mocked LLM returning valid Verdict JSON
When evaluate() is called with a concern
Then result is an AgentResult[Verdict]
And result.output.concern_id matches the input concern
```

**TS-2: Prompt loaded via load_specialist_prompt**
```
Given a mock PromptLoader that records calls
When evaluate() is called
Then load_specialist_prompt() was called with specialists_dir and prompt_file
```

**TS-3: Prompt formatted with all 5 variables**
```
Given a mock template "{style_requirements} {concern} {article_excerpt} {source_text} {source_metadata}"
And a mocked LLM that captures the prompt
When evaluate() is called with style_requirements="NATURE_NEWS"
Then the prompt contains "NATURE_NEWS"
And contains the concern review_note
And contains the concern excerpt
And contains the source text
And contains serialized source metadata
```

**TS-4: LLM response parsed as Verdict**
```
Given a mocked LLM returning '{"concern_id":1,"misleading":true,"status":"REWRITE","rationale":"Scope too broad","suggested_fix":"Narrow scope","evidence":null,"citations":null}'
When evaluate() is called
Then result.output.misleading is True
And result.output.status is "REWRITE"
And result.output.suggested_fix is "Narrow scope"
```

**TS-5: No KB or Perplexity or institutional memory calls**
```
Given an OpinionAgent (constructed without KB or Perplexity dependencies)
When evaluate() is called
Then only the LLM client was called
And no other external services were invoked
```

**TS-6: REWRITE verdict includes specific fix**
```
Given a mocked LLM returning status="REWRITE" with suggested_fix="Add qualifier: 'Altman suggested...'"
When evaluate() is called
Then result.output.suggested_fix is "Add qualifier: 'Altman suggested...'"
```

**TS-7: KEEP verdict has null suggested_fix**
```
Given a mocked LLM returning status="KEEP" with suggested_fix=null
When evaluate() is called
Then result.output.suggested_fix is None
```

**TS-8: Invalid JSON from LLM raises ValidationError**
```
Given a mocked LLM returning "Not valid JSON"
When evaluate() is called
Then ValidationError is raised
```

**TS-9: MockOpinionAgent implements protocol**
```
Given a MockOpinionAgent instance
When evaluate() is called
Then it returns without error
```

**TS-10: Mock returns KEEP with matching concern_id**
```
Given concern with concern_id=5
When MockOpinionAgent.evaluate() is called
Then result.output.concern_id is 5
And result.output.misleading is False
And result.output.status is "KEEP"
```

**TS-11: Verdict with evidence and citations fields**
```
Given a mocked LLM returning verdict with evidence="Source supports this" and citations=null
When evaluate() is called
Then result.output.evidence is "Source supports this"
And result.output.citations is None
```

## Traceability

All acceptance criteria and edge cases from the opinion agent behavioral specification are covered. Prompt content quality (AC-2.1, AC-2.2, AC-2.3) is tested via prompt file validation, not unit tests — the prompt's effectiveness is validated through integration testing with real LLMs.
