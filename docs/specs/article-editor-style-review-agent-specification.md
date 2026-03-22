# Style Review Agent — Behavioral Specification

## Objective

The Style Review Agent evaluates whether a concern involves a style choice that could mislead readers. Unlike other specialists that focus on factual accuracy, attribution, or interpretive fairness, this agent focuses on **how something is written** — tone, word choice, framing, and rhetorical devices — and whether the writing style creates a misleading impression even when the underlying content is technically accurate.

The agent does NOT perform a general style review of the entire article. It only evaluates specific concerns routed to it by the Concern Mapping Agent. It flags style as misleading only when the style choice changes the perceived meaning or risk of reader misunderstanding. Engaging, well-crafted writing that is still accurate is explicitly acceptable.

The Style Review Agent has no external dependencies (no KB, no web search, no institutional memory).

## Existing Implementation

- **Real agent:** `src/agents/article_generation/specialists/style_review/agent.py` — fully implemented (code complete, identical structure to opinion and attribution agents)
- **Mock agent:** `src/agents/article_generation/specialists/style_review/mock_agent.py` — returns static KEEP verdict
- **Prompt template:** `prompts/article_editor/specialists/style_review.md` — **STUB (27 lines)**

## User Stories & Acceptance Criteria

### US-1: As the orchestrator, I want the style review agent to evaluate whether a style concern could mislead readers.

**AC-1.1:** Given a `Concern`, `ArticleResponse`, source_text, source_metadata, and style_requirements, when `evaluate()` is called, then the agent returns an `AgentResult[Verdict]`.

**AC-1.2:** The agent loads the prompt from `prompts/article_editor/specialists/style_review.md` and formats it with: `{style_requirements}`, `{concern}`, `{article_excerpt}`, `{source_text}`, `{source_metadata}`.

**AC-1.3:** No external dependencies — each evaluation is a standalone LLM call.

---

### US-2: As the orchestrator, I want the style review prompt to define what constitutes misleading style, so that evaluations distinguish legitimate craft from problematic writing.

**AC-2.1:** The style review prompt MUST define **misleading style** (flag as `misleading=true`):

1. **Hype language:** "breakthrough", "game-changing", "revolutionary", "unprecedented" — unless directly quoted from the source AND contextualized skeptically in the article. Even when quoting hype from a source, the article should frame it critically.

2. **Unjustified certainty in tone:** Using definitive, authoritative tone for speculative or uncertain content. Example: writing "This will transform the industry" in a declarative style when the source speaker said "this might change things." The tone implies certainty the source doesn't support.

3. **Loaded framing:** Word choices that imply judgment not present in the source. Example: "stunning revelation" when the source calmly mentioned a fact; "admitted" instead of "said" when there's no implication of reluctance in the source.

4. **Clickbait patterns:** Headlines or descriptions designed to provoke clicks rather than inform. Example: "You Won't Believe What Happened Next" style. Also: questions in headlines that the article doesn't answer.

5. **Emotional manipulation:** Writing designed to evoke fear, excitement, or outrage beyond what the source warrants. Example: catastrophizing language about AI risks when the source discussed risks in measured terms.

6. **False balance:** Presenting a fringe view as equally weighted to a consensus view, creating a misleading impression of controversy. (Only applies when the article adds this framing; if the source itself presents false balance, that's a source issue, not a style issue.)

7. **Vague authority appeals:** "Experts say", "Scientists believe", "Research shows" without any named source in the input. This creates an illusion of broader support.

**AC-2.2:** The style review prompt MUST define **acceptable style** (flag as `misleading=false`):

1. **Engaging hooks:** Opening with a compelling question or scenario that is supported by the source content — standard journalistic craft.

2. **Active voice and strong verbs:** Using "Altman argued" instead of "It was stated by Altman that" — clarity improvement, not misleading.

3. **Metaphor and analogy:** When used to explain complex concepts and clearly labeled as explanatory devices, not presented as literal claims.

4. **Narrative flow:** Reordering source material for better storytelling when the reordering doesn't change meaning or create false causal implications.

5. **Conversational tone:** Slightly informal language in SCIAM_MAGAZINE mode that makes the article more accessible without distorting meaning.

6. **Emphasis through formatting:** Using bold, headings, or bullet points to highlight key information — organizational choice, not misleading.

7. **Characterizations that accurately reflect the source:** "Altman emphasized..." when the source clearly shows emphasis through repetition or explicit statements. This requires the characterization to be traceable to source evidence.

**AC-2.3:** The style review prompt MUST instruct the LLM to evaluate differently based on **style mode**:

- **NATURE_NEWS:**
  - Stricter standard for tone neutrality
  - Minimal narrative flourish
  - Any characterization beyond neutral verbs ("said", "described", "noted") requires clear source support
  - Information density prioritized over engagement
  - Hype language is almost never acceptable even when quoting

- **SCIAM_MAGAZINE:**
  - Slightly more latitude for conversational flow
  - Hooks and narrative devices are acceptable when source-supported
  - Stronger verbs ("argued", "emphasized", "cautioned") are acceptable when supported by context
  - Still zero tolerance for hype, certainty inflation, and emotional manipulation
  - Accessibility takes priority over density, but accuracy is non-negotiable

---

### US-3: As the orchestrator, I want style review verdicts to provide specific, minimal fixes.

**AC-3.1:** When `status=REWRITE`, `suggested_fix` MUST provide the specific style change needed. Examples:
- "Replace 'breakthrough' with 'advance' or 'development'"
- "Remove 'unusually direct' — it's an editorial characterization without source support; use 'Altman said' instead"
- "Change 'stunning results' to 'results that [speaker] described as significant'"
- "Rewrite headline to be informative rather than provocative: suggest '[specific alternative]'"

**AC-3.2:** When `status=KEEP`, `rationale` MUST explain why the style choice is acceptable despite being flagged. The `suggested_fix` SHOULD be `null`, but MAY suggest minor optional improvements.

**AC-3.3:** `status=REMOVE` is rare for style concerns. It applies when the style issue is so embedded in the content that it cannot be fixed by rewriting — for example, an entire section that exists solely for dramatic effect with no informational content.

**AC-3.4:** The `rationale` MUST be specific about which style guideline is or isn't violated. Generic responses like "this seems fine" are insufficient.

---

### US-4: As a developer, I want a mock style review agent for testing.

**AC-4.1:** `MockStyleReviewAgent` implements `SpecialistAgentProtocol` structurally.

**AC-4.2:** Returns a static KEEP verdict with `misleading=False` and matching `concern_id`.

**AC-4.3:** Instantiated by the agent factory when `agent_name == "mock"`.

## Constraints

### Technical
- Extends `BaseSpecialistAgent`; uses `_call_llm()` and `_parse_json_response()`
- No external dependencies
- Temperature: 0.3
- Prompt template variables: `{style_requirements}`, `{concern}`, `{article_excerpt}`, `{source_text}`, `{source_metadata}`

### Style vs Content Boundary
- The style review agent evaluates HOW something is written, not WHETHER it's true
- A factually correct statement written in misleading style is `REWRITE` (fix the style), not `REMOVE` (delete the content)
- A concern that is purely about content accuracy (not style) should have been routed to a different specialist; the style agent should note this and evaluate only the style dimension

### The Misleading Threshold
- The question is not "Is this perfectly neutral?" but "Would the style cause a reasonable reader to form a materially incorrect impression?"
- Minor stylistic preferences (e.g., "said" vs "stated") are not misleading
- The agent should not impose its own stylistic preferences — only evaluate against the specified STYLE_MODE requirements

## Edge Cases

### EC-1: Concern about a single adjective
Example: "unusually direct" — evaluate whether the adjective creates a misleading impression. A characterization like "unusually" implies comparison knowledge the source doesn't provide. Typically `REWRITE`.

### EC-2: Concern about section headings
Headings are evaluated for accuracy and tone. A section headed "A Founder's Crisis" when the source speaker briefly mentioned adversity may be `REWRITE` if the heading overstates the content.

### EC-3: Concern about the description/teaser
The description appears in search results and social shares. It has outsized influence on reader expectations. Sensational descriptions for measured content are `REWRITE`.

### EC-4: Concern about writing quality (not misleadingness)
If the concern is about poor writing (e.g., unclear sentences) rather than misleading style, the agent should `KEEP` with a note that the concern is about quality, not misleadingness. The style agent only flags style that *misleads*.

### EC-5: Concern about a glossary entry
Glossary definitions are style additions. If they're accurate and helpful, `KEEP`. If they contain loaded framing or editorial spin, `REWRITE`.

### EC-6: Style concern where the source itself uses hype language
When the source speaker uses hype language and the article quotes it, the agent evaluates whether the article properly contextualizes the quote. Uncritical presentation of source hype may be `REWRITE` (add skeptical framing).

## Non-Goals

- **The style review agent does not perform a full article style audit.** It evaluates only the specific concern routed to it.
- **The style review agent does not verify facts.** Only style impact.
- **The style review agent does not check attribution.** Only tone and framing.
- **The style review agent does not evaluate interpretive fairness.** That's the Opinion Agent.
- **The style review agent is not called on every iteration.** It's only called when the concern mapping agent selects it for a specific concern.

## Prompt Gap Analysis

The current prompt (27 lines) says "Check whether the concern adheres to style requirements. Flag only style choices likely to mislead." but provides:
- No definition of misleading vs acceptable style (AC-2.1, AC-2.2)
- No style-mode-specific evaluation criteria (AC-2.3)
- No examples of specific style issues
- No guidance on fix granularity
- No discussion of the misleading threshold

**The prompt needs to be substantially rewritten.** The agent code requires no changes.

## Open Questions

None.
