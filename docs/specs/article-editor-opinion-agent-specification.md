# Opinion Agent — Behavioral Specification

## Objective

The Opinion Agent is the most frequently used specialist in the editorial pipeline. It evaluates whether a flagged concern represents **fair journalistic interpretation** or a **misleading extrapolation**. Most concerns from the Article Review Agent involve interpretive choices — generalizations, inferred conclusions, structural framing, or editorial characterizations — rather than verifiable factual claims. The Opinion Agent applies journalistic judgment principles to determine whether each such choice is acceptable or crosses the line into misleading territory.

The Opinion Agent has no access to external data sources (no KB, no web search). It relies entirely on the source text, the article excerpt, the style requirements, and principled reasoning.

## Existing Implementation

- **Real agent:** `src/agents/article_generation/specialists/opinion/agent.py` — fully implemented (code complete)
- **Mock agent:** `src/agents/article_generation/specialists/opinion/mock_agent.py` — returns static KEEP verdict
- **Prompt template:** `prompts/article_editor/specialists/opinion.md` — **STUB (27 lines)**. This is the highest-priority prompt to improve, as the opinion agent handles the majority of concerns.

## User Stories & Acceptance Criteria

### US-1: As the orchestrator, I want the opinion agent to evaluate interpretive concerns, so that reasonable journalism is preserved while misleading content is flagged.

**AC-1.1:** Given a `Concern`, `ArticleResponse`, source_text, source_metadata, and style_requirements, when `evaluate()` is called, then the agent returns an `AgentResult[Verdict]`.

**AC-1.2:** The agent loads the prompt from `prompts/article_editor/specialists/opinion.md` and formats it with: `{style_requirements}`, `{concern}`, `{article_excerpt}`, `{source_text}`, `{source_metadata}`.

**AC-1.3:** The agent sends a single user message to the LLM and parses the response as a `Verdict`.

**AC-1.4:** The agent has no external dependencies (no KB, no Perplexity, no institutional memory). Each evaluation is a standalone LLM call.

---

### US-2: As the orchestrator, I want the opinion agent prompt to define clear judgment criteria, so that evaluations are consistent and principled.

**AC-2.1:** The opinion prompt MUST define the core distinction the agent is making:

**Fair journalistic interpretation (misleading=false):**
- Drawing reasonable conclusions that are clearly implied by the source, even if not explicitly stated
- Using standard journalistic shorthand (e.g., "appearance at an event" when someone spoke at an event)
- Organizing source material into structured formats (tables, timelines) when the structure faithfully represents the source content
- Using more precise language than the source (e.g., replacing "stuff" with specific terms) when the meaning is preserved
- Adding contextual framing that helps the reader understand the significance, when the framing is clearly labeled as interpretive
- Narrowing verbose or repetitive source material to its essential meaning

**Misleading extrapolation (misleading=true):**
- Generalizing beyond the source's explicit qualifiers (e.g., "some" → "most", "in this context" → "broadly")
- Inflating certainty (e.g., "might" → "will", "suggests" → "demonstrates")
- Completing truncated statements (source cuts off, article "finishes" the thought)
- Adding causal relationships not stated in the source (e.g., "because" or "this led to" when the source only describes temporal sequence)
- Characterizing a speaker's tone, intent, or emotional state beyond what the source shows
- Creating analytical frameworks (e.g., "What remains debated" sections listing specific debates not present in source)
- Presenting the article's own analysis as if it came from the source speaker

**AC-2.2:** The opinion prompt MUST provide guidance on **borderline cases** — the gray zone where reasonable people could disagree:

- **Scope narrowing is usually acceptable** — replacing "people" with "engineers" when the context makes it clear the speaker is discussing engineers
- **Scope broadening is usually misleading** — replacing "engineers" with "everyone" when the source only discussed engineers
- **Structural additions (tables, timelines) are acceptable** if they faithfully reorganize stated information without adding interpretive columns
- **Structural additions are misleading** if they add "implication" or "significance" columns that contain inferences not in the source
- **Formalizing casual speech is acceptable** — cleaning up disfluencies, improving grammar
- **Changing meaning through formalization is misleading** — replacing a hedged statement with a definitive one through "cleanup"
- **When in doubt, flag as misleading** with `status=REWRITE` and suggest adding attribution or qualification rather than `REMOVE`

**AC-2.3:** The opinion prompt MUST instruct the agent to consider the **style mode** when evaluating:
- **NATURE_NEWS:** Stricter standard. Fewer interpretive liberties. Neutral, evidence-first framing. Even minor editorializing should be flagged.
- **SCIAM_MAGAZINE:** Slightly more latitude for conversational hooks and contextual framing, but still no tolerance for certainty inflation, scope expansion, or misleading characterizations.

---

### US-3: As the orchestrator, I want the opinion verdict to provide actionable fixes, so that the writer knows exactly how to address concerns.

**AC-3.1:** When `status=REWRITE`, `suggested_fix` MUST provide a specific instruction. Examples:
- "Add attribution: 'Altman suggested...' instead of stating as fact"
- "Narrow scope: replace 'younger users' with 'young people he described' to match source"
- "Add qualifier: change 'will transform' to 'could transform' to match the speaker's tentative framing"
- "Remove interpretive column from table; keep only directly stated information"

**AC-3.2:** When `status=KEEP`, `suggested_fix` SHOULD be `null`. The `rationale` explains why the interpretation is acceptable.

**AC-3.3:** When `status=REMOVE`, `suggested_fix` SHOULD explain what to replace the removed content with, or state that the section can be dropped entirely.

**AC-3.4:** The `rationale` MUST be 1-3 sentences explaining the reasoning. It should reference the specific source text that does or does not support the article's interpretation.

---

### US-4: As a developer, I want a mock opinion agent for testing.

**AC-4.1:** `MockOpinionAgent` implements `SpecialistAgentProtocol` structurally.

**AC-4.2:** Returns a static KEEP verdict with `misleading=False` and matching `concern_id`.

**AC-4.3:** Instantiated by the agent factory when `agent_name == "mock"`.

## Constraints

### Technical
- Extends `BaseSpecialistAgent`; uses `_call_llm()` and `_parse_json_response()`
- No external dependencies (no KB, no Perplexity, no institutional memory)
- Each evaluation is stateless — no memory across calls
- Temperature: 0.3

### Judgment Principles
- The opinion agent's job is to protect readers from being misled, not to enforce rigid fidelity to source wording
- A good article adds value through organization, clarity, and context — the agent should not penalize these contributions
- The threshold for `misleading=true` is: would a reasonable reader form a materially different understanding of the topic because of this addition?
- When the concern is borderline, prefer `REWRITE` (add qualification) over `REMOVE` (delete content)

## Edge Cases

### EC-1: Concern about a table or structured summary
The agent evaluates whether the table faithfully represents source content or adds interpretive elements. Tables with "implications" columns that aren't in the source are typically `REWRITE` (remove the interpretive column).

### EC-2: Concern about a "What remains uncertain" section
These sections are analytical additions. If the individual uncertainty items are genuinely open questions raised by the source, `KEEP`. If they assert specific debates not present in the source, `REWRITE`.

### EC-3: Concern about headline or description
Headlines and descriptions are evaluated with the same criteria as body text. A clickbait headline not supported by the source is `REWRITE`.

### EC-4: Concern about glossary definitions
Glossary definitions are non-source additions by design. If definitions are generic and accurate, `KEEP` with note that they should be labeled as editorial additions. If definitions contain claims not in the source, `REWRITE`.

### EC-5: Multiple interpretive issues in a single concern
The agent evaluates the concern as a whole. If some aspects are acceptable and others are not, `REWRITE` with specific guidance on which parts to fix.

### EC-6: Concern references a cut-off transcript
If the source ends mid-sentence and the article completes the thought, this is almost always `misleading=true`, `status=REWRITE` or `REMOVE`. The article should not present inferred completions as stated content.

## Non-Goals

- **The opinion agent does not verify facts.** It evaluates interpretive choices, not factual accuracy.
- **The opinion agent does not search for evidence.** No KB, no web search.
- **The opinion agent does not check attribution specifically.** Attribution quality is the Attribution Agent's domain.
- **The opinion agent does not evaluate overall article quality.** Only the specific concern.

## Prompt Gap Analysis

The current prompt (27 lines) says "Assess whether the concern represents fair journalistic interpretation or a misleading extrapolation" but provides no detailed judgment criteria, no examples, no borderline case guidance, no style-mode-specific instructions, and no guidance on fix types.

**The prompt needs to be substantially rewritten** with the detailed criteria from AC-2.1, AC-2.2, and AC-2.3. The agent code requires no changes.

## Open Questions

None.
