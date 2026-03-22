# Concern Mapping Agent — Test Specification

## Coverage Matrix

| Spec Requirement | Test Scenario(s) |
|-----------------|------------------|
| AC-1.1: map_concerns() returns AgentResult[ConcernMappingResult] | TS-1 |
| AC-1.2: ConcernMapping fields validated | TS-2, TS-3 |
| AC-1.3: Exactly one specialist per concern | TS-1 |
| AC-4.1: JSON array format accepted | TS-4 |
| AC-4.2: JSON object format accepted | TS-5 |
| AC-4.3: Invalid array items raise ValidationError | TS-6 |
| AC-5.1: MockConcernMappingAgent implements protocol | TS-7 |
| AC-5.2: Mock returns empty mappings | TS-8 |
| EC-1: Single concern | TS-9 |
| EC-5: Invalid selected_agent raises ValidationError | TS-10 |
| EC-6: Array with markdown fences | TS-11 |
| EC-7: Empty array | TS-12 |

## Test Scenarios

**TS-1: map_concerns() returns valid mapping for each concern**
```
Given a ConcernMappingAgent with mocked LLM returning valid mapping JSON
And 2 input concerns
When map_concerns() is called
Then result.output.mappings has 2 items
And each mapping has concern_id, concern_type, selected_agent, confidence, reason
```

**TS-2: concern_type validates against ConcernType literals**
```
Given a mocked LLM returning concern_type="invalid_type"
When map_concerns() is called
Then ValidationError is raised
```

**TS-3: selected_agent validates against 5 literal values**
```
Given a mocked LLM returning selected_agent="nonexistent"
When map_concerns() is called
Then ValidationError is raised
```

**TS-4: JSON array format parsed correctly**
```
Given a mocked LLM returning '[{"concern_id":1,"concern_type":"scope_expansion","selected_agent":"opinion","confidence":"high","reason":"test"}]'
When map_concerns() is called
Then result.output.mappings has 1 item
And the mapping fields match the input
```

**TS-5: JSON object format parsed correctly**
```
Given a mocked LLM returning '{"mappings":[{"concern_id":1,"concern_type":"scope_expansion","selected_agent":"opinion","confidence":"high","reason":"test"}]}'
When map_concerns() is called
Then result.output.mappings has 1 item
```

**TS-6: Array with non-object items raises ValueError**
```
Given a mocked LLM returning '[1, 2, 3]'
When map_concerns() is called
Then ValueError is raised with message about JSON objects
```

**TS-7: MockConcernMappingAgent structurally satisfies protocol**
```
Given a MockConcernMappingAgent instance
When used where ConcernMappingAgentProtocol is expected
Then no type errors and map_concerns() is callable
```

**TS-8: Mock returns empty mappings**
```
When MockConcernMappingAgent.map_concerns() is called
Then result.output.mappings is []
```

**TS-9: Single concern mapped correctly**
```
Given 1 input concern
And mocked LLM returning 1 mapping
When map_concerns() is called
Then result.output.mappings has exactly 1 item
```

**TS-10: Invalid selected_agent value rejected**
```
Given a mocked LLM returning selected_agent="sentiment_analysis"
When map_concerns() is called
Then ValidationError is raised
```

**TS-11: Markdown-fenced array format (object path strips fences)**
```
Given a mocked LLM returning '```json\n{"mappings":[...]}\n```'
When map_concerns() is called
Then the fences are stripped and mappings are parsed correctly
```

**TS-12: Empty array returns empty mappings**
```
Given a mocked LLM returning '[]'
When map_concerns() is called
Then result.output.mappings is []
```

**TS-13: Prompt formatted with all context variables**
```
Given a mock template "{style_requirements} {source_text} {generated_article} {concerns}"
And a mocked LLM that captures the prompt
When map_concerns() is called with style_requirements="NATURE_NEWS"
Then the prompt contains "NATURE_NEWS"
And contains the source text
And contains the article JSON
And contains the serialized concerns
```

## Traceability

All acceptance criteria and edge cases from the concern mapping behavioral specification are covered.
