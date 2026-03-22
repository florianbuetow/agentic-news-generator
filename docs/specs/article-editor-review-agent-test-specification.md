# Article Review Agent — Test Specification

## Coverage Matrix

| Spec Requirement | Test Scenario(s) |
|-----------------|------------------|
| AC-1.1: review() returns AgentResult[ArticleReviewRaw] | TS-1 |
| AC-1.2: Output is markdown bullets | TS-2 |
| AC-1.3: Empty output when no concerns | TS-3 |
| AC-1.4: Prompt loaded and formatted | TS-4, TS-5 |
| AC-1.5: Raw response stored as markdown_bullets | TS-6 |
| AC-2.1: Bullets quote article text | TS-7 |
| AC-4.4: Output is bullets only, no JSON | TS-8 |
| AC-5.1: Empty input → empty concerns | TS-9 |
| AC-5.2: Whitespace-only → empty concerns | TS-10 |
| AC-5.3: Bullet detection (- and *) | TS-11 |
| AC-5.4: Multi-line bullet continuation | TS-12 |
| AC-5.5: concern_id sequential from 1 | TS-13 |
| AC-5.6: Excerpt extraction priority | TS-14, TS-15, TS-16 |
| AC-5.7: review_note is full bullet text | TS-17 |
| AC-5.8: Non-bullet text raises ValueError | TS-18 |
| AC-6.1: MockArticleReviewAgent implements protocol | TS-19 |
| AC-6.2: Mock returns empty bullets | TS-20 |
| EC-3: Numbered list raises ValueError | TS-21 |
| EC-4: Bullets without quoted excerpts | TS-16 |
| EC-5: Preamble before bullets ignored | TS-22 |
| EC-6: Nested sub-bullets | TS-23 |
| EC-7: Bold/italic in bullets | TS-24 |

## Test Scenarios

### Review Agent

**TS-1: review() returns AgentResult with markdown bullets**
```
Given an ArticleReviewAgent with mocked LLM returning bullet list text
When review() is called with a valid article, source_text, and metadata
Then the result is an AgentResult[ArticleReviewRaw]
And result.output.markdown_bullets is a non-empty string
```

**TS-2: LLM response preserved as-is in markdown_bullets**
```
Given a mocked LLM returning "- Concern A\n- Concern B"
When review() is called
Then result.output.markdown_bullets is "- Concern A\n- Concern B"
```

**TS-3: Empty LLM response signals no concerns**
```
Given a mocked LLM returning ""
When review() is called
Then result.output.markdown_bullets is ""
```

**TS-4: Prompt loaded via PromptLoader**
```
Given an ArticleReviewAgent with prompt_file="article_review.md"
And a mock PromptLoader that records calls
When review() is called
Then PromptLoader.load_prompt(prompt_file="article_review.md") was called
```

**TS-5: Prompt template formatted with all variables**
```
Given a mock template "{source_text} {source_metadata} {generated_article}"
And a mocked LLM that captures the prompt
When review() is called with source_text="Transcript here"
Then the prompt contains "Transcript here"
And contains JSON-serialized source_metadata
And contains JSON-serialized article
```

**TS-6: Response stripped of whitespace**
```
Given a mocked LLM returning "  \n- Concern A\n  "
When review() is called
Then result.output.markdown_bullets is "- Concern A"
```

**TS-7: Prompt instructs quoting of article text**
```
Given the article review prompt file at prompts/article_editor/article_review.md
When the file content is read
Then it references the generated article and source text
```

**TS-8: Agent does not parse JSON from response**
```
Given a mocked LLM returning valid JSON (e.g., '{"concerns": []}')
When review() is called
Then result.output.markdown_bullets contains the raw JSON string
And no JSON parsing was attempted by the agent
```

### Bullet Parser

**TS-9: Empty string returns empty concerns**
```
Given markdown_bullets=""
When parse() is called
Then result.concerns is []
```

**TS-10: Whitespace-only returns empty concerns**
```
Given markdown_bullets="  \n\t\n  "
When parse() is called
Then result.concerns is []
```

**TS-11: Dash and asterisk bullets both detected**
```
Given markdown_bullets="- First\n* Second"
When parse() is called
Then result.concerns has 2 items
And concern 1 review_note is "First"
And concern 2 review_note is "Second"
```

**TS-12: Multi-line bullet continuation**
```
Given markdown_bullets="- Line one\n  continuation\n  more\n- Next bullet"
When parse() is called
Then result.concerns has 2 items
And concern 1 review_note contains "Line one" and "continuation" and "more"
And concern 2 review_note is "Next bullet"
```

**TS-13: concern_id assigned sequentially from 1**
```
Given markdown_bullets with 5 bullets
When parse() is called
Then concern_ids are [1, 2, 3, 4, 5]
```

**TS-14: Excerpt extraction — curly quotes have highest priority**
```
Given bullet text: '**\u201cCurly quoted\u201d** and "straight quoted" text'
When _extract_excerpt() is called
Then excerpt is "Curly quoted" (curly quotes win)
```

**TS-15: Excerpt extraction — straight quotes used when no curly quotes**
```
Given bullet text: 'The article says "straight quoted" which is wrong'
When _extract_excerpt() is called
Then excerpt is "straight quoted"
```

**TS-16: Excerpt extraction — full bullet text when no quotes**
```
Given bullet text: "No quotes at all in this bullet"
When _extract_excerpt() is called
Then excerpt is "No quotes at all in this bullet"
```

**TS-17: review_note is full bullet text verbatim**
```
Given markdown_bullets='- **"Excerpt"** — explanation of the concern'
When parse() is called
Then concern.review_note is '**"Excerpt"** — explanation of the concern'
And concern.excerpt is "Excerpt"
```

**TS-18: Non-empty text without bullets raises ValueError**
```
Given markdown_bullets="This is just a paragraph without bullets"
When parse() is called
Then ValueError is raised with message containing "no markdown bullets"
```

### Mock Agent

**TS-19: MockArticleReviewAgent structurally satisfies protocol**
```
Given a MockArticleReviewAgent instance
When used where ArticleReviewAgentProtocol is expected
Then no type errors occur and review() is callable
```

**TS-20: Mock returns empty review**
```
When MockArticleReviewAgent.review() is called with any arguments
Then result.prompt is "[mock]"
And result.output.markdown_bullets is ""
```

### Edge Cases

**TS-21: Numbered list raises ValueError**
```
Given markdown_bullets="1. First item\n2. Second item"
When parse() is called
Then ValueError is raised (no markdown bullet pattern matched)
```

**TS-22: Preamble text before bullets is ignored**
```
Given markdown_bullets="Here are my concerns:\n\n- First concern\n- Second concern"
When parse() is called
Then result.concerns has 2 items
And the preamble text is not in any concern
```

**TS-23: Nested sub-bullets treated as continuation**
```
Given markdown_bullets="- Parent concern\n  - Sub-point one\n  - Sub-point two\n- Next concern"
When parse() is called
Then result.concerns has 2 items
And concern 1 review_note contains "Parent concern" and "Sub-point one" and "Sub-point two"
And concern 2 review_note is "Next concern"
```

**TS-24: Bold and italic formatting preserved in review_note**
```
Given markdown_bullets='- **Bold** and *italic* text'
When parse() is called
Then concern.review_note is "**Bold** and *italic* text"
```

## Traceability

Every acceptance criterion and edge case from the article review agent behavioral specification is covered. Bullet parser behavior (AC-5.x) is tested independently of the review agent since the parser is a separate class.
