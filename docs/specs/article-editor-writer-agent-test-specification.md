# Writer Agent — Test Specification

## Coverage Matrix

| Spec Requirement | Test Scenario(s) |
|-----------------|------------------|
| AC-1.1: generate() returns AgentResult[ArticleResponse] | TS-1 |
| AC-1.2: ArticleResponse has 4 required fields | TS-2 |
| AC-1.3: Prompt loaded via PromptLoader | TS-3 |
| AC-1.4: Prompt template variables | TS-4 |
| AC-1.5: Single user message sent | TS-5 |
| AC-1.6: Invalid JSON response raises ValidationError | TS-6, TS-7 |
| AC-1.7: AgentResult.prompt contains assembled prompt | TS-8 |
| AC-2.1: revise() returns AgentResult[ArticleResponse] | TS-9 |
| AC-2.3: Revision template variables | TS-10 |
| AC-2.5: Revised article same schema as initial | TS-9 |
| AC-3.1: MockWriterAgent implements protocol | TS-11 |
| AC-3.2: Mock generate returns static article | TS-12 |
| AC-3.3: Mock revise returns same static article | TS-13 |
| AC-4.1: Writer prompt rules present | TS-14 |
| AC-4.4: Target length in prompt | TS-15 |
| EC-2: Empty reader_preference | TS-16 |
| EC-3: LLM returns non-JSON | TS-6 |
| EC-4: LLM returns JSON with extra fields | TS-7 |
| EC-5: LLM returns JSON with missing fields | TS-17 |
| EC-6: Long source_text | TS-18 |
| EC-8: Source metadata with null values | TS-19 |

## Test Scenarios

### Initial Draft Generation

**TS-1: generate() returns valid AgentResult**
```
Given a WriterAgent with a mocked LLM client that returns valid ArticleResponse JSON
When generate() is called with valid source_text, metadata, style_mode, reader_preference
Then the result is an AgentResult[ArticleResponse]
And result.output.headline is non-empty
And result.output.article_body is non-empty
And result.output.description is non-empty
```

**TS-2: ArticleResponse rejects missing fields**
```
Given a mocked LLM client that returns JSON with only "headline" and "article_body"
When generate() is called
Then ValidationError is raised (missing alternative_headline and description)
```

**TS-3: Prompt loaded from PromptLoader, not hardcoded**
```
Given a WriterAgent with writer_prompt_file="writer.md"
And a mock PromptLoader that records load_prompt() calls
When generate() is called
Then PromptLoader.load_prompt(prompt_file="writer.md") was called exactly once
```

**TS-4: Template formatted with all variables**
```
Given a mock PromptLoader returning template "{style_mode} {reader_preference} {source_text} {source_metadata}"
And a mocked LLM client that captures the prompt
When generate() is called with style_mode="NATURE_NEWS", reader_preference="focus on methods",
    source_text="Sample transcript", source_metadata={"channel_name": "Test"}
Then the assembled prompt contains "NATURE_NEWS"
And contains "focus on methods"
And contains "Sample transcript"
And contains "Test" (from serialized metadata)
```

**TS-5: Single user message sent to LLM**
```
Given a mocked LLM client that records messages
When generate() is called
Then the LLM received exactly 1 message
And the message has role="user"
```

### Response Parsing

**TS-6: Non-JSON LLM response raises error**
```
Given a mocked LLM client that returns "This is not JSON"
When generate() is called
Then ValidationError is raised
```

**TS-7: Extra fields in JSON raise ValidationError**
```
Given a mocked LLM client that returns valid ArticleResponse JSON with extra field "author"
When generate() is called
Then ValidationError is raised (extra="forbid" on ArticleResponse)
```

**TS-8: AgentResult.prompt contains full assembled prompt**
```
Given a mock PromptLoader returning "Template: {source_text}"
And source_text="Hello"
When generate() is called
Then result.prompt is "Template: Hello"
And result.prompt is not the raw template string
```

### Revision

**TS-9: revise() returns valid revised article**
```
Given a WriterAgent with a mocked LLM client that returns valid ArticleResponse JSON
And a WriterFeedback with iteration=1, rating=5, passed=False, todo_list=["Fix X"]
When revise() is called with context and feedback
Then the result is an AgentResult[ArticleResponse]
And result.output follows the same schema as generate()
```

**TS-10: Revision template formatted with feedback fields**
```
Given a mock PromptLoader returning template with all feedback variables
And feedback with rating=3, passed=False, reasoning="Fix it",
    todo_list=["Fix A", "Fix B"], improvement_suggestions=["Improve C"],
    verdicts=[verdict_with_citations]
And a mocked LLM that captures the prompt
When revise() is called
Then the prompt contains "3" (rating)
And contains "False" (pass_status)
And contains "Fix it" (reasoning)
And contains "- Fix A\n- Fix B" (todo_list formatted as bullets)
And contains "- Improve C" (improvement_suggestions formatted)
And contains JSON-serialized verdicts
And contains the context JSON string
```

### Mock Agent

**TS-11: MockWriterAgent structurally satisfies protocol**
```
Given a MockWriterAgent instance
When it is used where WriterAgentProtocol is expected
Then no type errors occur at runtime
And generate() and revise() are both callable
```

**TS-12: Mock generate returns static article**
```
When MockWriterAgent.generate() is called with any arguments
Then result.prompt is "[mock]"
And result.output.headline contains "Mock Article"
And result.output.article_body is non-empty
```

**TS-13: Mock revise returns same static article**
```
When MockWriterAgent.revise() is called with any feedback
Then result.output is identical to generate() output
```

### Prompt Content Validation

**TS-14: Writer prompt template contains required rules**
```
Given the writer prompt file at prompts/article_editor/writer.md
When the file content is read
Then it contains "JSON" (output format requirement)
And contains "source text" (source fidelity rule)
And contains "NATURE_NEWS" (style mode definition)
And contains "SCIAM_MAGAZINE" (style mode definition)
And contains "headline" (output field)
And contains "article_body" (output field)
```

**TS-15: Writer prompt includes target length**
```
Given the writer prompt file
When the file content is read
Then it contains "900" or "1200" (target word count range)
```

### Edge Cases

**TS-16: Empty reader_preference handled gracefully**
```
Given reader_preference=""
When generate() is called
Then no error is raised
And the prompt is assembled with empty reader preference
```

**TS-17: LLM returns JSON missing required field**
```
Given a mocked LLM returning {"headline": "Test", "article_body": "Body"}
When generate() is called
Then ValidationError is raised (missing alternative_headline and description)
```

**TS-18: Very long source_text triggers token validation warning**
```
Given source_text of 100,000 characters
And a mocked LLM client
When generate() is called
Then _validate_tokens() is called (verify via log or mock)
And the LLM call proceeds (no exception from token validation)
```

**TS-19: Source metadata with null values serialized correctly**
```
Given source_metadata={"channel_name": "Test", "publish_date": None}
When generate() is called
Then the prompt contains serialized metadata including the null value
And no error is raised
```

## Traceability

Every acceptance criterion and edge case from the writer agent behavioral specification is covered by at least one test scenario.
