# Concern Mapping Agent — Behavioral Specification

## Objective

The Concern Mapping Agent is the routing brain of the editorial pipeline. It receives a list of concerns (unsupported additions identified by the Article Review Agent) and decides which specialist agent is best suited to evaluate each one. The mapping is based on the nature of the concern — factual claims go to fact-checking, attribution issues go to the attribution agent, and so on.

This agent exists so that concern-to-specialist routing is **adaptive rather than hardcoded**. An LLM evaluates each concern in context (considering the source text, article, and style requirements) and selects the most appropriate specialist. This allows the system to handle novel concern patterns without code changes.

The mapping agent assigns **exactly one** specialist per concern. There is no secondary agent or fallback routing.

## Existing Implementation

- **Real agent:** `src/agents/article_generation/concern_mapping/agent.py` — fully implemented
- **Mock agent:** `src/agents/article_generation/concern_mapping/mock_agent.py` — returns empty mappings
- **Prompt template:** `prompts/article_editor/concern_mapping.md` — **complete and production-ready** (35 lines)

## User Stories & Acceptance Criteria

### US-1: As the orchestrator, I want each concern mapped to exactly one specialist agent, so that every concern receives a single authoritative evaluation.

**AC-1.1:** Given a list of `Concern` objects, style_requirements, source_text, and generated_article_json, when `map_concerns()` is called, then the agent returns an `AgentResult[ConcernMappingResult]` containing one `ConcernMapping` per input concern.

**AC-1.2:** Each `ConcernMapping` contains exactly these fields:
- `concern_id` (int) — matches a `concern_id` from the input list
- `concern_type` (ConcernType literal) — one of: `"unsupported_fact"`, `"inferred_fact"`, `"scope_expansion"`, `"editorializing"`, `"structured_addition"`, `"attribution_gap"`, `"certainty_inflation"`, `"truncation_completion"`
- `selected_agent` (literal) — one of: `"fact_check"`, `"evidence_finding"`, `"opinion"`, `"attribution"`, `"style_review"`
- `confidence` (literal) — one of: `"high"`, `"medium"`, `"low"`
- `reason` (str) — 1-2 sentence explanation of why this specialist was chosen

**AC-1.3:** The mapping agent maps each concern to exactly one specialist. There is no `secondary_agent` field, no fallback routing, and no multi-agent evaluation for a single concern.

**AC-1.4:** Every `concern_id` in the output MUST correspond to a `concern_id` in the input concern list. The orchestrator validates this and raises `ValueError` if a mapping references a non-existent concern.

---

### US-2: As the orchestrator, I want the mapping agent to understand each specialist's capabilities, so that routing decisions are well-informed.

**AC-2.1:** The mapping prompt MUST describe each specialist's capabilities:

1. **fact_check** — Validates factual statements using a knowledge base (RAG). Best for: verifiable factual claims, specific dates/numbers, named entity claims that can be cross-referenced. Has access to a pre-indexed knowledge base of source documents.

2. **evidence_finding** — Uses web search (Perplexity API) to find supporting or refuting evidence. Best for: claims that need external verification beyond the knowledge base, well-known facts, research findings, industry claims. Returns URLs/citations that can be used in article footnotes.

3. **opinion** — Uses critical thinking to determine if exaggerations, generalizations, extrapolations, or interpretive framing are acceptable journalistic interpretation or misleading. Best for: scope expansions, editorializing, certainty inflation, inferential leaps, structured framing additions. This is the most commonly used specialist.

4. **attribution** — Finds references to claims in the source text and checks attribution quality. Best for: attribution gaps (narrator voice vs speaker attribution), misattribution, claims where the source speaker said something but the article presents it differently.

5. **style_review** — Checks adherence to style requirements and flags only style choices that could mislead. Best for: tone issues, hype language, loaded framing, characterizations that imply judgment. Only flags style as misleading when it changes the perceived meaning — engaging writing that is still accurate is acceptable.

**AC-2.2:** The mapping prompt MUST include the complete concern type taxonomy (the 8 types from AC-1.2) so the LLM can accurately classify each concern.

---

### US-3: As the orchestrator, I want the mapping agent to consider context when routing, so that ambiguous concerns go to the best specialist.

**AC-3.1:** The mapping prompt receives the full context:
- `{style_requirements}` — the style mode (NATURE_NEWS or SCIAM_MAGAZINE)
- `{source_text}` — the full transcript for cross-reference
- `{generated_article}` — the full article JSON for context
- `{concerns}` — JSON-serialized list of all concern objects

**AC-3.2:** The LLM should consider the concern text, the relevant article excerpt, and the source text when deciding which specialist is most appropriate. A concern about "2026" might go to `fact_check` (to verify from metadata) or `opinion` (to judge if the inference is fair) depending on context.

**AC-3.3:** The `confidence` field reflects how certain the mapping agent is about its routing decision:
- `"high"` — clear match between concern type and specialist capability
- `"medium"` — reasonable match but another specialist could also handle it
- `"low"` — uncertain; the concern doesn't clearly fit one specialist

---

### US-4: As a developer, I want the mapping agent to handle both JSON array and JSON object response formats, so that LLM output format variation does not cause failures.

**AC-4.1:** The agent accepts two response formats from the LLM:

**Format A — JSON array:**
```json
[
  {"concern_id": 1, "concern_type": "scope_expansion", "selected_agent": "opinion", "confidence": "high", "reason": "..."},
  {"concern_id": 2, "concern_type": "inferred_fact", "selected_agent": "fact_check", "confidence": "medium", "reason": "..."}
]
```

**Format B — JSON object with mappings key:**
```json
{
  "mappings": [
    {"concern_id": 1, "concern_type": "scope_expansion", "selected_agent": "opinion", "confidence": "high", "reason": "..."},
    {"concern_id": 2, "concern_type": "inferred_fact", "selected_agent": "fact_check", "confidence": "medium", "reason": "..."}
  ]
}
```

**AC-4.2:** Format detection: if the stripped response starts with `[`, parse as Format A. Otherwise, parse as Format B via `_parse_json_response()` into `ConcernMappingResult`.

**AC-4.3:** For Format A, each array item is validated individually via `ConcernMapping.model_validate()`. Invalid items raise `ValidationError`.

**AC-4.4:** For both formats, the prompt instructs the LLM to return either format, so both are equally valid.

---

### US-5: As a developer, I want a mock concern mapping agent for testing the orchestration loop.

**AC-5.1:** `MockConcernMappingAgent` implements `ConcernMappingAgentProtocol` structurally.

**AC-5.2:** `MockConcernMappingAgent.map_concerns()` returns `AgentResult(prompt="[mock]", output=ConcernMappingResult(mappings=[]))`.

**AC-5.3:** An empty mappings list means the orchestrator has zero concerns to dispatch to specialists. Combined with a `MockArticleReviewAgent` that returns empty bullets, this allows testing the SUCCESS path without any LLM calls.

**AC-5.4:** The agent factory instantiates `MockConcernMappingAgent` when `agent_name == "mock"`.

## Constraints

### Technical
- The concern mapping agent extends `BaseAgent`
- It sends a single user message to the LLM
- Response parsing handles both JSON array and JSON object formats
- Pydantic validation enforces strict field types (literals for concern_type, selected_agent, confidence)
- The agent does not access external services (no KB, no Perplexity)

### Configuration
- `prompt_file` is read from `config.yaml` under `article_generation.editor.prompts.concern_mapping_prompt_file`
- LLM parameters under `article_generation.agents.concern_mapping`
- Temperature: 0.3 (deterministic routing preferred)
- `max_tokens`: 2048

### Output Validation
- `concern_type` MUST be one of the 8 literal values. Invalid values cause `ValidationError`.
- `selected_agent` MUST be one of 5 literal values. Invalid values cause `ValidationError`.
- `confidence` MUST be `"high"`, `"medium"`, or `"low"`. Invalid values cause `ValidationError`.
- `concern_id` MUST be an integer. Type mismatch causes `ValidationError`.

## Edge Cases

### EC-1: Single concern in input
The agent returns a single-item mapping list. Works identically to multi-concern input.

### EC-2: LLM returns fewer mappings than concerns
The orchestrator iterates only over returned mappings. Unmapped concerns are silently skipped (they are not evaluated by any specialist). This is a potential quality gap but not an error.

### EC-3: LLM returns mappings for concern_ids not in the input
The orchestrator raises `ValueError` when it cannot find the concern by ID.

### EC-4: LLM returns duplicate concern_ids
Both mappings are processed. The orchestrator dispatches the concern to a specialist twice with potentially different agents. This is suboptimal but not an error in the mapping agent itself.

### EC-5: LLM returns invalid selected_agent value
`ConcernMapping.model_validate()` raises `ValidationError` because `selected_agent` is a strict Literal type.

### EC-6: LLM returns response with markdown fences
Format B handling via `_parse_json_response()` strips markdown fences (```json...```). Format A handling does not — if the array is wrapped in fences, `json.loads()` will fail.

### EC-7: LLM returns empty array
`ConcernMappingResult(mappings=[])` is valid. The orchestrator has no concerns to dispatch and proceeds to check pass/fail (which will pass since there are no misleading verdicts).

## Non-Goals

- **The mapping agent does not evaluate concerns.** It only routes them.
- **The mapping agent does not access external data.** Routing is based on the concern text and context.
- **The mapping agent does not assign multiple specialists per concern.** One concern → one specialist.
- **The mapping agent does not filter or prioritize concerns.** All input concerns are mapped.
- **The mapping agent does not determine if a concern is valid.** It assumes all concerns from the review agent are worth routing.

## Open Questions

None. The prompt template and agent code are both complete.
