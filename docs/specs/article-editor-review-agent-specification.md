# Article Review Agent — Behavioral Specification

## Objective

The Article Review Agent is the editorial quality gate of the pipeline. It compares a generated article against its source text and identifies every addition, interpretation, inference, or framing choice that is not directly backed by the source. The agent does not judge whether additions are acceptable — it only surfaces them. Downstream agents (concern mapping + specialists) make the keep/rewrite/remove decisions.

The agent's output is a **markdown bullet list** (one bullet per concern). This raw output is then parsed by the deterministic `ArticleReviewBulletParser` into structured `Concern` objects. The parser is tightly coupled to this agent's output format and is specified here alongside the agent.

## Existing Implementation

- **Real agent:** `src/agents/article_generation/article_review/agent.py` — fully implemented (code structure complete)
- **Mock agent:** `src/agents/article_generation/article_review/mock_agent.py` — returns empty review (no concerns)
- **Bullet parser:** `src/agents/article_generation/chief_editor/bullet_parser.py` — fully implemented
- **Prompt template:** `prompts/article_editor/article_review.md` — **STUB (11 lines)**. This is the primary gap. The prompt lacks the detailed instructions needed for the LLM to produce consistently structured, high-quality reviews.

## User Stories & Acceptance Criteria

### US-1: As the orchestrator, I want the review agent to identify all unsupported additions in a generated article, so that the editorial loop can evaluate each one.

**AC-1.1:** Given an `ArticleResponse`, source_text, and source_metadata, when `review()` is called, then the agent returns an `AgentResult[ArticleReviewRaw]` containing the assembled prompt and a `markdown_bullets` string.

**AC-1.2:** The `markdown_bullets` string contains zero or more markdown bullets. Each bullet starts with `- ` or `* ` at the beginning of a line. Multi-line bullets are supported (continuation lines do not start with `- ` or `* `).

**AC-1.3:** If the article faithfully represents the source text with no unsupported additions, the agent returns an empty string for `markdown_bullets`.

**AC-1.4:** The agent loads the prompt template from `prompts/article_editor/article_review.md` via the injected `PromptLoader`. The template is formatted with:
- `{source_text}` — the full transcript
- `{source_metadata}` — JSON-serialized metadata dict
- `{generated_article}` — JSON-serialized `ArticleResponse`

**AC-1.5:** The agent sends the formatted prompt as a single user message to the LLM and returns the raw response (stripped of leading/trailing whitespace) as `markdown_bullets`. No JSON parsing is performed — the output is free-form markdown.

---

### US-2: As the orchestrator, I want each review bullet to quote the problematic text from the article, so that downstream agents can evaluate the specific concern.

**AC-2.1:** Each bullet SHOULD contain a quoted excerpt from the article enclosed in curly quotes `"..."` or straight quotes `"..."`. The excerpt identifies the specific text that is not backed by the source.

**AC-2.2:** Each bullet SHOULD contain an explanation of why the text is considered an unsupported addition. The explanation follows the quoted excerpt.

**AC-2.3:** Example of a well-formed bullet:
```
- **"Younger users treat the model like an operating system"** — the source says "young people" in a specific context, but the article generalizes to "younger users" broadly, which is an extrapolation beyond the speaker's framing.
```

**AC-2.4:** Example of a minimal but acceptable bullet:
```
- "2026 label" — the source says "next year" without naming 2026 explicitly.
```

---

### US-3: As the orchestrator, I want the review agent prompt to define the concern taxonomy, so that the LLM produces categorizable concerns.

**AC-3.1:** The review prompt MUST define the following categories of unsupported additions and instruct the LLM to consider all of them:

1. **Unsupported factual addition** — A new factual detail not present anywhere in the source text. Example: adding a specific date, number, or event that the source never mentions.

2. **Inferred fact** — A factual detail derived or calculated from the source but never explicitly stated. Example: mapping "next year" to "2026" based on contextual clues, or combining two separate source statements to produce a third conclusion.

3. **Scope expansion / generalization** — Broadening a qualified statement beyond its original scope. Example: source says "some enterprises" but article says "organizations broadly"; source says "young people in this context" but article says "younger users".

4. **Editorializing** — Loaded phrasing, characterizations, or subjective judgments not present in the source. Example: describing a speaker as "unusually direct" or "cautious" when the source contains no such characterization.

5. **Structured framing addition** — Tables, glossaries, "what remains uncertain" sections, structured timelines, or other organizational frameworks that impose interpretive structure not present in the source. Example: creating a three-column table mapping age groups to behaviors when the source discusses these informally.

6. **Attribution gap** — Claims presented in narrator voice (as if factual) that should be attributed to a specific source speaker. Example: "AI will transform coding" stated as fact when the source has a specific person saying this.

7. **Certainty inflation** — Shifting from tentative language ("might", "suggests", "could") in the source to definitive language ("will", "shows", "proves") in the article. Example: source says "this might indicate" but article says "this demonstrates".

8. **Truncation completion** — The source text is cut off or incomplete, but the article "finishes" the thought or fills in what was likely meant. Example: source transcript ends mid-sentence but the article completes the idea as if the speaker said it.

**AC-3.2:** The review prompt MUST instruct the LLM that these categories are for the reviewer's awareness only — the reviewer does not need to label each concern with a category. Labeling is the Concern Mapping Agent's job.

---

### US-4: As the orchestrator, I want the review prompt to produce comprehensive, detailed reviews, so that no significant unsupported addition goes undetected.

**AC-4.1:** The review prompt MUST instruct the LLM to:
1. Read the SOURCE_TEXT completely before reviewing the article
2. Compare each claim, characterization, and structural element in the article against the source
3. Quote the specific article text that is unsupported (using `"..."` quotes)
4. Explain what the source actually says (or doesn't say) that makes this an unsupported addition
5. Err on the side of flagging — it is better to flag a legitimate inference than to miss a fabrication

**AC-4.2:** The review prompt MUST instruct the LLM to check the following article elements:
- Factual claims and statistics
- Characterizations of people, organizations, or events
- Causal relationships or implications
- Temporal claims (dates, sequences, timelines)
- Scope of claims (who/what/where qualifiers)
- Tables, lists, and structured summaries
- Glossary definitions
- Headlines and description (these can also contain unsupported claims)
- Attribution accuracy (who said what)
- Certainty level of language vs source

**AC-4.3:** The review prompt MUST instruct the LLM NOT to:
- Judge whether an addition is acceptable or misleading (that's the specialist's job)
- Suggest fixes or rewrites
- Comment on writing quality or style (that's the style review agent's job)
- Flag things that ARE supported by the source (only flag unsupported additions)
- Add preamble, summary, or commentary outside the bullet list format

**AC-4.4:** The review prompt MUST instruct the LLM to return ONLY a markdown bullet list (or an empty string if no concerns are found). No JSON, no headers, no preamble.

---

### US-5: As a developer, I want the bullet parser to deterministically convert markdown bullets into structured concerns, so that downstream processing is reliable and testable.

**AC-5.1:** The bullet parser takes `markdown_bullets` (string) and returns `ArticleReviewResult` containing a list of `Concern` objects.

**AC-5.2:** **Empty input handling:** If `markdown_bullets` is empty or whitespace-only, the parser returns `ArticleReviewResult(concerns=[])`.

**AC-5.3:** **Bullet detection:** A new bullet starts on any line beginning with `- ` (dash-space) or `* ` (asterisk-space). Lines not starting with either pattern are continuation lines of the current bullet.

**AC-5.4:** **Multi-line bullets:** Continuation lines are joined with `\n` into the current bullet's text. The bullet text is stripped of leading/trailing whitespace.

**AC-5.5:** **concern_id assignment:** IDs are assigned sequentially starting at 1, in the order bullets appear in the input.

**AC-5.6:** **Excerpt extraction priority:**
1. First, search for curly-quoted text: `\u201c...\u201d`. If found, use the first match as `excerpt`.
2. Else, search for straight-quoted text: `"..."`. If found, use the first match as `excerpt`.
3. Else, use the full bullet text as `excerpt`.

**AC-5.7:** **review_note:** The full bullet text (verbatim, after joining multi-line content).

**AC-5.8:** **Invalid output handling:** If `markdown_bullets` is non-empty (after stripping) but no bullets are detected (no lines starting with `- ` or `* `), the parser raises `ValueError`. This is a fail-fast signal that the review agent returned an unexpected format.

---

### US-6: As a developer, I want a mock review agent that returns empty reviews, so that I can test the orchestration loop without LLM calls.

**AC-6.1:** `MockArticleReviewAgent` implements `ArticleReviewAgentProtocol` structurally.

**AC-6.2:** `MockArticleReviewAgent.review()` returns `AgentResult(prompt="[mock]", output=ArticleReviewRaw(markdown_bullets=""))`.

**AC-6.3:** An empty `markdown_bullets` string causes the orchestrator to skip the editorial loop and return SUCCESS immediately (per the orchestration spec AC-2.1 step 3).

**AC-6.4:** The agent factory instantiates `MockArticleReviewAgent` when `agent_name == "mock"` in the article_review agent's config section.

---

### US-7: As a pipeline operator, I want the review agent prompt to handle edge cases in source text, so that reviews are robust across different input types.

**AC-7.1:** The review prompt SHOULD handle these source text characteristics:
- **Transcripts with disfluencies:** "you know", "I mean", "um" — the article may clean these up, which is NOT an unsupported addition (normal editorial cleanup).
- **Cut-off transcripts:** Source text may end mid-sentence. The review must flag any article content that "completes" the cut-off thought.
- **Multiple speakers:** Source may contain dialogue between multiple people. The review must flag misattribution (attributing Speaker A's statement to Speaker B).
- **Informal language:** Source speakers may use casual language that the article formalizes. Formalization is acceptable editorial practice; factual transformation is not.
- **Ambiguous statements:** When the source is genuinely ambiguous, the review should flag article interpretations that pick one meaning over another without qualification.

**AC-7.2:** The review prompt MUST distinguish between:
- **Editorial cleanup** (acceptable, do NOT flag): grammar fixes, removing filler words, reordering for clarity, combining related statements
- **Unsupported additions** (MUST flag): new facts, generalizations beyond source scope, certainty inflation, fabricated characterizations

## Constraints

### Technical
- The article review agent extends `BaseAgent` and uses `_call_llm()` from the base class
- The agent accepts `LLMConfig`, `Config`, `LLMClient`, `PromptLoader`, and `prompt_file` via keyword-only constructor arguments
- The agent does NOT use `_parse_json_response()` — its output is free-form markdown, not JSON
- The raw LLM response is stripped of whitespace and stored as `ArticleReviewRaw.markdown_bullets`
- The bullet parser is a separate class (`ArticleReviewBulletParser`) invoked by the orchestrator, NOT by the review agent itself

### Configuration
- `prompt_file` is read from `config.yaml` under `article_generation.editor.prompts.article_review_prompt_file`
- LLM parameters are configured under `article_generation.agents.article_review`
- The review agent uses lower temperature (0.3) than the writer (0.7) for more deterministic output
- `max_tokens` is set to 2048 (sufficient for a bullet list; the review never produces article-length output)

### Output Format
- **Success:** A markdown bullet list where each bullet quotes problematic text and explains the mismatch
- **No concerns:** An empty string (or whitespace-only, which the parser treats as empty)
- **Never:** JSON, numbered lists, headers, or prose paragraphs

## Edge Cases

### EC-1: Article perfectly represents source
The review agent returns an empty string. The orchestrator sees zero concerns and returns SUCCESS.

### EC-2: Source text is very long (exceeds context window)
Token validation warns but the LLM call proceeds. The LLM may truncate the source internally, potentially missing unsupported additions in the truncated portion. This is a known limitation — the system relies on context window being large enough for the source text + article + prompt.

### EC-3: LLM returns numbered list instead of bullet list
The bullet parser will raise `ValueError` because no lines start with `- ` or `* `. This is a fail-fast signal that the prompt needs adjustment or the LLM deviated from instructions.

### EC-4: LLM returns bullets without quoted excerpts
The parser falls back to using the full bullet text as `excerpt` (AC-5.6 step 3). The concern is still processed but the excerpt may be less precise for downstream agents.

### EC-5: LLM returns preamble before bullets
Lines before the first bullet that don't start with `- ` or `* ` are silently ignored by the parser (they aren't continuation lines because no bullet has started yet). The concerns from actual bullets are still parsed correctly.

### EC-6: LLM returns bullets with nested sub-bullets
Sub-bullets (indented `- ` or `* `) are treated as continuation lines of the parent bullet because they are indented (don't match the start-of-line pattern). The full nested text becomes part of the parent concern's `review_note`.

### EC-7: LLM returns markdown with bold/italic formatting in bullets
Formatting markers (`**`, `*`, `_`) are preserved in `review_note` and may appear in `excerpt` if inside quotes. This does not affect parsing.

### EC-8: Article contains footnotes from a previous revision
The review should flag footnote content that is not supported by the source text, just like any other content. However, footnotes backed by citations from specialist verdicts (from a previous iteration) are legitimate — the review agent has no way to know this, so it may flag them. The concern mapping agent and specialists will then determine they are acceptable.

### EC-9: Source metadata is empty or minimal
The prompt template includes `{source_metadata}` which may serialize to `{}` or `{"channel_name": "...", ...}`. The review agent works regardless of metadata richness.

## Non-Goals

- **The review agent does not judge quality.** It only identifies unsupported additions. Whether an addition is acceptable is determined by specialist agents.
- **The review agent does not suggest fixes.** It describes the concern; the specialist provides the fix.
- **The review agent does not categorize concerns.** Categorization (unsupported_fact, scope_expansion, etc.) is done by the Concern Mapping Agent.
- **The review agent does not check style.** Style compliance is the Style Review Agent's job.
- **The review agent does not access external data.** It compares only the article against the provided source text.
- **The review agent does not remember previous iterations.** Each review is a fresh comparison of the current article against the source.

## Prompt Gap Analysis

The current prompt (`prompts/article_editor/article_review.md`) is 11 lines:

```
Please review the generated article. For any additions that were made that were
not present in the source text, create a bullet point list of facts or statements
that are not backed by the source text.

SOURCE_TEXT:
{source_text}

SOURCE_METADATA:
{source_metadata}

GENERATED_ARTICLE:
{generated_article}
```

**What's missing vs this spec:**
1. The concern taxonomy (AC-3.1) — the 8 categories of unsupported additions
2. The instruction to quote specific article text (AC-2.1)
3. The instruction to explain what the source actually says (AC-4.1)
4. The distinction between editorial cleanup and unsupported additions (AC-7.2)
5. The instruction to check all article elements including headlines and description (AC-4.2)
6. The instruction to err on the side of flagging (AC-4.1)
7. The explicit "do NOT" instructions (AC-4.3)
8. The output format constraint (markdown bullets only, no JSON) (AC-4.4)
9. Handling of transcript-specific edge cases (disfluencies, cut-offs, multiple speakers) (AC-7.1)

**The prompt needs to be rewritten to incorporate all of the above.** The code (`article_review/agent.py`) requires no changes — only the prompt template needs updating.

## Open Questions

None. The behavioral contract is fully specified. Implementation requires updating the prompt template only.
