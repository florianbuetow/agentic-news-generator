# Attribution Agent — Behavioral Specification

## Objective

The Attribution Agent verifies that claims in a generated article are properly attributed to their source. In journalism, attribution is the practice of clearly indicating who said, claimed, or suggested something. Incorrect attribution can mislead readers by presenting one person's opinion as fact, attributing a claim to the wrong person, or presenting the article's own interpretation as if it came from a source speaker.

When the Concern Mapping Agent routes a concern to the attribution specialist, the agent examines the article excerpt, traces the claim back to the source text, and determines whether the attribution is accurate, missing, or misleading.

The Attribution Agent has no external dependencies (no KB, no web search, no institutional memory).

## Existing Implementation

- **Real agent:** `src/agents/article_generation/specialists/attribution/agent.py` — fully implemented (code complete, identical structure to opinion agent)
- **Mock agent:** `src/agents/article_generation/specialists/attribution/mock_agent.py` — returns static KEEP verdict
- **Prompt template:** `prompts/article_editor/specialists/attribution.md` — **STUB (27 lines)**

## User Stories & Acceptance Criteria

### US-1: As the orchestrator, I want the attribution agent to verify source attribution accuracy, so that readers know who said what.

**AC-1.1:** Given a `Concern`, `ArticleResponse`, source_text, source_metadata, and style_requirements, when `evaluate()` is called, then the agent returns an `AgentResult[Verdict]`.

**AC-1.2:** The agent loads the prompt from `prompts/article_editor/specialists/attribution.md` and formats it with: `{style_requirements}`, `{concern}`, `{article_excerpt}`, `{source_text}`, `{source_metadata}`.

**AC-1.3:** No external dependencies — each evaluation is a standalone LLM call.

---

### US-2: As the orchestrator, I want the attribution agent prompt to define specific attribution rules, so that evaluations are rigorous and consistent.

**AC-2.1:** The attribution prompt MUST instruct the LLM to check for these **attribution patterns**:

**Proper attribution (misleading=false):**
- Direct attribution: "Altman said...", "According to [speaker]...", "[Speaker] argued that..."
- Qualified attribution: "Altman suggested...", "[Speaker] indicated that..."
- Clearly labeled interpretation: "This implies...", "The data suggests...", "One reading is..."
- Standard journalistic shorthand when context makes the source clear

**Improper attribution (misleading=true):**
- **Missing attribution (narrator voice):** The article presents a speaker's claim as objective fact without any attribution. Example: Source has "Altman said AI will change coding" → article says "AI will change coding" as if it's an established fact.
- **Misattribution:** The article attributes a claim to the wrong speaker. Example: Speaker A says X, but the article attributes it to Speaker B.
- **Attribution inflation:** The article upgrades a suggestion to a definitive claim. Example: source has "he guessed that..." → article says "he confirmed that..."
- **Attribution to unnamed sources:** The article references "experts say" or "researchers believe" when no such sources exist in the input. This is fabrication.
- **Self-attribution disguised as source:** The article's own analysis is written in a way that suggests it came from the source speaker.

**AC-2.2:** The attribution prompt MUST instruct the LLM to trace claims:
1. Find the specific location in the source text where the claim originates (if it exists)
2. Identify who made the claim in the source (speaker name, role, or context)
3. Compare the article's attribution against the source
4. Determine if the attribution accurately reflects who said it, how confidently they said it, and in what context

**AC-2.3:** The attribution prompt MUST instruct the LLM to populate the `evidence` field (or a `source_support` description in the `rationale`) with:
- A brief quote or pointer to the relevant source text passage
- Who the source attributes the claim to
- How the source frames the claim (tentative vs definitive)

---

### US-3: As the orchestrator, I want the attribution verdict to provide specific fix instructions, so that the writer can correct attribution issues precisely.

**AC-3.1:** When `status=REWRITE` for missing attribution, `suggested_fix` MUST provide the correct attribution phrasing. Examples:
- "Add attribution: 'According to Altman, ...' instead of presenting as fact"
- "Change from narrator voice to attributed: 'The speaker suggested that...'"
- "Qualify with: 'In Altman's view, ...' to make clear this is one person's perspective"

**AC-3.2:** When `status=REWRITE` for attribution inflation, `suggested_fix` MUST downgrade the attribution verb. Examples:
- "Replace 'confirmed' with 'suggested' to match the source's tentative framing"
- "Replace 'showed' with 'described' — the speaker was recounting observations, not presenting evidence"

**AC-3.3:** When `status=REMOVE` (rare for attribution issues), `suggested_fix` MUST explain why the claim cannot be salvaged through better attribution. This typically applies when the article contains fabricated attribution to non-existent sources.

**AC-3.4:** When `status=KEEP`, `rationale` MUST explain why the current attribution is acceptable. The `suggested_fix` SHOULD be `null`.

---

### US-4: As a developer, I want a mock attribution agent for testing.

**AC-4.1:** `MockAttributionAgent` implements `SpecialistAgentProtocol` structurally.

**AC-4.2:** Returns a static KEEP verdict with `misleading=False` and matching `concern_id`.

**AC-4.3:** Instantiated by the agent factory when `agent_name == "mock"`.

## Constraints

### Technical
- Extends `BaseSpecialistAgent`; uses `_call_llm()` and `_parse_json_response()`
- No external dependencies
- Temperature: 0.2 (precise evaluation preferred)
- Prompt template variables: `{style_requirements}`, `{concern}`, `{article_excerpt}`, `{source_text}`, `{source_metadata}`

### Attribution Principles
- Attribution serves transparency: readers should know the provenance of every claim
- The attribution agent does not evaluate whether a claim is true — only whether it's properly attributed
- Missing attribution is not automatically misleading: some claims are common knowledge or established fact that don't need attribution. The threshold is: would a reasonable reader incorrectly assume this claim is established fact vs one person's opinion?
- Style mode affects attribution expectations: NATURE_NEWS demands more rigorous attribution than SCIAM_MAGAZINE, which allows slightly more conversational narrator voice

## Edge Cases

### EC-1: Multiple speakers in source
The source may contain dialogue between multiple people. The agent must verify that each attributed claim in the article maps to the correct speaker.

### EC-2: Paraphrased quotes
The article may paraphrase a speaker rather than quoting directly. Paraphrasing is acceptable if it preserves meaning and attributes correctly. Paraphrasing that changes meaning is `REWRITE`.

### EC-3: Implicit attribution through context
Sometimes a section of the article clearly discusses one speaker, making attribution implicit. The agent should consider paragraph context, not just the individual sentence.

### EC-4: Article adds editorial characterizations
When the article characterizes a speaker (e.g., "Altman was unusually direct"), this is editorializing, not attribution. The attribution agent may flag it if the characterization implies the speaker said something they didn't, but pure tone characterization is better handled by the opinion or style review agents.

### EC-5: Source has no named speakers
If the source is a document without named authors/speakers, attribution to "the source" or "the document" is acceptable. The attribution agent should not demand specific names when the source doesn't provide them.

### EC-6: Article uses "according to" for its own inferences
This is a form of misleading attribution — the article implies the source said something when it's actually the article's interpretation. This is `misleading=true`, `status=REWRITE`.

## Non-Goals

- **The attribution agent does not verify factual accuracy.** Only attribution quality.
- **The attribution agent does not evaluate interpretive fairness.** That's the Opinion Agent.
- **The attribution agent does not check style.** That's the Style Review Agent.
- **The attribution agent does not search for external sources.** It only traces claims to the provided source text.

## Prompt Gap Analysis

The current prompt (27 lines) says "Check whether the claim is properly attributed and traceable to the source text" but provides:
- No definition of proper vs improper attribution patterns (AC-2.1)
- No instructions for tracing claims to specific source passages (AC-2.2)
- No guidance on fix phrasing (AC-3.1, AC-3.2)
- No handling of edge cases (multiple speakers, paraphrasing, implicit attribution)

**The prompt needs to be substantially rewritten.** The agent code requires no changes.

## Open Questions

None.
