# Writer Agent — Behavioral Specification

## Objective

The Writer Agent is the article-producing engine of the editorial pipeline. It generates an initial science journalism article from source text (a transcript or similar document) and revises that article in response to structured editorial feedback. The agent's output is a strict JSON object containing the headline, alternative headline, article body (Markdown), and description. The agent is invoked by the Chief Editor Orchestrator and never called directly by the pipeline script.

## Existing Implementation

- **Real agent:** `src/agents/article_generation/writer/agent.py` — fully implemented
- **Mock agent:** `src/agents/article_generation/writer/mock_agent.py` — returns static draft, no LLM
- **System prompt (legacy, unused by current flow):** `src/agents/article_generation/agent_prompts.py`
- **Prompt template (active):** `prompts/article_editor/writer.md` — comprehensive, production-ready
- **Revision template (active):** `prompts/article_editor/revision.md` — functional but minimal

## User Stories & Acceptance Criteria

### US-1: As the orchestrator, I want the writer agent to generate an initial article draft from source text, so that the editorial review loop has material to evaluate.

**AC-1.1:** Given source_text, source_metadata, style_mode, and reader_preference, when `generate()` is called, then the agent returns an `AgentResult[ArticleResponse]` containing both the assembled prompt and a valid `ArticleResponse`.

**AC-1.2:** The `ArticleResponse` contains exactly four fields: `headline` (non-empty string), `alternative_headline` (non-empty string), `article_body` (Markdown string using `\n` for line breaks), and `description` (1-2 sentence string).

**AC-1.3:** The agent loads the prompt template from `prompts/article_editor/writer.md` via the injected `PromptLoader`. The template is never hardcoded in the agent source.

**AC-1.4:** The prompt template is formatted with these variables:
- `{style_mode}` — `"NATURE_NEWS"` or `"SCIAM_MAGAZINE"`
- `{reader_preference}` — free-text angle or empty string
- `{source_text}` — the full transcript
- `{source_metadata}` — JSON-serialized metadata dict

**AC-1.5:** The agent sends the formatted prompt as a single user message (`[{"role": "user", "content": prompt}]`) to the LLM via `_call_llm()`.

**AC-1.6:** The LLM response is parsed via `_parse_json_response()` into an `ArticleResponse`. If the response is not valid JSON or does not match the `ArticleResponse` schema, a `ValidationError` is raised (fail-fast).

**AC-1.7:** The `AgentResult.prompt` field contains the full assembled prompt text (not the template). This is used by the orchestrator for artifact logging.

---

### US-2: As the orchestrator, I want the writer agent to revise a draft based on editorial feedback, so that identified issues are addressed without losing the original context.

**AC-2.1:** Given a `context` JSON string and a `WriterFeedback` object, when `revise()` is called, then the agent returns an `AgentResult[ArticleResponse]` with the revised article.

**AC-2.2:** The revision prompt template is loaded from `prompts/article_editor/revision.md` via `PromptLoader`.

**AC-2.3:** The revision template is formatted with these variables:
- `{rating}` — integer 1-10 from `feedback.rating`
- `{pass_status}` — string representation of `feedback.passed` (always `"False"`)
- `{reasoning}` — deterministic summary from `feedback.reasoning`
- `{todo_list}` — bullet list of required changes, formatted as `"- {todo}\n- {todo}\n..."`
- `{improvement_suggestions}` — bullet list of optional improvements, formatted as `"- {suggestion}\n..."`
- `{verdicts}` — JSON-serialized list of all verdict objects (for citation access)
- `{context}` — JSON string containing source_text, source_metadata, style_mode, reader_preference, and current_article

**AC-2.4:** The `context` parameter is a JSON string built by the orchestrator containing:
```json
{
  "style_mode": "...",
  "reader_preference": "...",
  "source_text": "...",
  "source_metadata": {...},
  "current_article": {...}
}
```
The writer agent passes this through to the prompt template without parsing it.

**AC-2.5:** The revised article output follows the same `ArticleResponse` schema as the initial draft. The agent does not track revision history — each revision is a fresh generation from the prompt.

**AC-2.6:** The revision prompt includes the full verdict list so the writer can access `citations` from fact-check and evidence-finding verdicts to properly format footnotes.

---

### US-3: As a developer, I want a mock writer agent that returns static articles without LLM calls, so that I can test the orchestration loop independently.

**AC-3.1:** `MockWriterAgent` implements `WriterAgentProtocol` (structurally typed via Python protocols — no explicit inheritance required).

**AC-3.2:** `MockWriterAgent.generate()` returns a static `AgentResult[ArticleResponse]` with hardcoded headline, body, and description. The prompt field is `"[mock]"`.

**AC-3.3:** `MockWriterAgent.revise()` returns the same static article as `generate()`. The mock never modifies output based on feedback.

**AC-3.4:** The agent factory instantiates `MockWriterAgent` when `agent_name == "mock"` in the writer agent's config section.

---

### US-4: As a pipeline operator, I want the writer prompt to enforce strict journalistic rules, so that articles are high-quality and source-faithful.

**AC-4.1:** The writer prompt (`prompts/article_editor/writer.md`) MUST include the following rules:
1. **JSON-only output** — no preamble, no commentary
2. **Strict JSON validity** — double quotes, no trailing commas, `\n` for line breaks (not raw newlines)
3. **Source fidelity** — claims MUST be supported by source text; no fabricated facts, numbers, quotes, citations, or attributions
4. **Anti-hype** — no promotional language unless directly quoted from source and contextualized skeptically
5. **Evidence discipline** — calibrated language ("suggests", "is consistent with"), uncertainty surfaced prominently
6. **Attribution** — clear attribution when entities are in source; no fabricated competing views
7. **No fabricated links** — only URLs from input metadata; never invent DOIs, paper links, or journal references
8. **No fabricated images** — only include images if URLs provided in input metadata
9. **No first-person** — no "I/we", no moralizing, no clickbait, no rhetorical filler
10. **Description** — 1-2 sentences, non-sensational

**AC-4.2:** The writer prompt MUST define style behavior for each mode:
- **NATURE_NEWS:** Neutral, compact, high signal-to-noise. Structure: fast lede → why it matters → what was done → results → context → limitations → next steps.
- **SCIAM_MAGAZINE:** Stronger hooks (if source-supported), slightly conversational. Structure: hook → plain-language core idea → how it works → context → limitations → what comes next.

**AC-4.3:** The writer prompt MUST include structured steps guiding the LLM:
1. Extract main claim, methods, quantitative results, scope, limitations from source
2. Identify the news peg
3. Structure by style mode
4. Calibrate certainty
5. Add uncertainty section
6. Define technical terms if needed (≤ 6 terms in glossary)
7. Produce all four output fields
8. Validate JSON

**AC-4.4:** The writer prompt MUST specify target article length as 900-1200 words.

**AC-4.5:** The writer prompt MUST support the `{reader_preference}` variable, allowing the operator to steer the article angle (e.g., "focus on methods", "focus on implications for healthcare").

---

### US-5: As the orchestrator, I want the revision prompt to provide actionable editorial instructions, so that the writer addresses specific concerns rather than rewriting blindly.

**AC-5.1:** The revision prompt (`prompts/article_editor/revision.md`) MUST present:
1. The numerical rating (1-10)
2. The pass/fail status
3. The reasoning summary (numbered list of required changes)
4. The required changes (todo_list) as a bullet list
5. The optional improvement suggestions as a separate bullet list
6. The full specialist verdicts as JSON (for citation access)
7. The original context (source text, metadata, style, current article)

**AC-5.2:** The revision prompt MUST instruct the writer to:
1. Address all items in the required changes list
2. Consider (but not require) the improvement suggestions
3. Maintain the same JSON output format as the initial draft
4. Not introduce new unsupported content while fixing flagged issues
5. Use citations from verdicts for footnotes when external evidence is available

**AC-5.3:** The revision prompt MUST remind the writer of the original rules (source fidelity, anti-hype, evidence discipline) so that revisions do not regress on quality.

**AC-5.4:** The revision prompt output MUST be JSON-only — same `ArticleResponse` schema as the initial draft.

## Constraints

### Technical
- The writer agent extends `BaseAgent` and uses `_call_llm()` and `_parse_json_response()` from the base class
- The agent accepts `LLMConfig`, `Config`, `LLMClient`, `PromptLoader`, `writer_prompt_file`, and `revision_prompt_file` via keyword-only constructor arguments
- Prompt templates are loaded from disk via `PromptLoader` — never embedded as Python constants in the agent
- The agent sends a single user message (not a system+user pair) to the LLM
- Token validation runs before every LLM call via `_validate_tokens()`

### Configuration
- `writer_prompt_file` and `revision_prompt_file` are read from `config.yaml` under `article_generation.editor.prompts`
- LLM parameters (model, api_base, api_key, context_window, max_tokens, temperature, timeout_seconds, max_retries, retry_delay) are configured per-agent under `article_generation.agents.writer`
- The writer uses higher temperature (0.7) than review/specialist agents (0.2-0.3) to encourage creative writing

### Output Format
- `headline`: Primary title. Informative, not clickbait. Under 100 characters preferred.
- `alternative_headline`: Secondary title or subtitle. Complements the headline.
- `article_body`: Full article in Markdown. Uses `\n` for line breaks inside the JSON string. May include headings (`##`, `###`), emphasis, lists, and tables. Must NOT include raw newlines (which would break JSON).
- `description`: 1-2 sentence teaser for search snippets. Accurate, non-sensational.

## Edge Cases

### EC-1: Empty source_text
The agent passes empty text to the LLM. Behavior depends on the LLM — it may produce a minimal article or fail. The agent does not validate source_text content; validation is the caller's responsibility.

### EC-2: Empty reader_preference
The prompt template includes `{reader_preference}` which formats to an empty string. The writer produces a general-purpose article without angle steering.

### EC-3: LLM returns non-JSON response
`_parse_json_response()` attempts markdown fence stripping (```json ... ```), then calls `model_validate_json()`. If the response is not valid JSON, `ValidationError` is raised.

### EC-4: LLM returns JSON with extra fields
`ArticleResponse` uses `extra="forbid"`, so extra fields cause `ValidationError`.

### EC-5: LLM returns JSON with missing fields
`ArticleResponse` requires all four fields. Missing fields cause `ValidationError`.

### EC-6: Very long source_text exceeding context window
`_validate_tokens()` warns when token count exceeds the configured threshold percentage of the context window. The LLM call proceeds regardless (the LLM may truncate or error). The agent does not truncate source text.

### EC-7: Revision with zero todo items
The writer receives a revision prompt where `{todo_list}` is empty. The `{improvement_suggestions}` may still have items. The writer should focus on optional improvements. This case arises when all verdicts are KEEP but the overall loop hasn't passed (which shouldn't happen per orchestrator logic, but the agent handles it gracefully).

### EC-8: Source metadata contains null values
The `source_metadata` dict is JSON-serialized including null values. The prompt template receives the full JSON string.

### EC-9: Verdicts in revision contain citations
The writer should use `citations` from verdict objects to format footnotes in `article_body`. Citations are URLs or reference strings provided by the fact-check or evidence-finding agents. The writer must NOT invent additional citations.

## Non-Goals

- **The writer does not perform self-review.** All quality assessment is done by downstream agents.
- **The writer does not access external data sources.** It works solely from the source_text and metadata provided.
- **The writer does not track revision history.** Each `revise()` call is stateless — the agent has no memory of previous iterations.
- **The writer does not determine target length dynamically.** Target length (900-1200 words) is fixed in the prompt template.
- **The writer does not select style mode.** Style mode is passed by the orchestrator from config.

## Open Questions

None. The writer agent implementation is complete and the prompt template is comprehensive.
