# Evidence Finding Agent — Behavioral Specification

## Objective

The Evidence Finding Agent uses external web search (Perplexity API) to find supporting or refuting evidence for claims flagged in a generated article. When the Concern Mapping Agent routes a concern to this specialist, the agent searches the web, evaluates the findings, and produces a verdict with citations that the writer can use for footnotes. Like the Fact Check Agent, it integrates with institutional memory to avoid redundant Perplexity API calls.

The key distinction from the Fact Check Agent: **fact check uses a local knowledge base; evidence finding uses live web search.** Evidence finding is appropriate for claims that need broader external validation — well-known facts, industry claims, research findings, or statements verifiable through public sources.

## Existing Implementation

- **Real agent:** `src/agents/article_generation/specialists/evidence_finding/agent.py` — fully implemented (code complete)
- **Mock agent:** `src/agents/article_generation/specialists/evidence_finding/mock_agent.py` — returns static KEEP verdict
- **Prompt template:** `prompts/article_editor/specialists/evidence_finding.md` — **STUB (29 lines)**
- **Perplexity client:** `src/agents/article_generation/perplexity_client.py` — fully implemented (HTTPS client)
- **Institutional memory:** shared with fact check agent

## User Stories & Acceptance Criteria

### US-1: As the orchestrator, I want the evidence finding agent to search the web for evidence related to a concern, so that article claims are validated against external sources.

**AC-1.1:** Given a `Concern`, `ArticleResponse`, source_text, source_metadata, and style_requirements, when `evaluate()` is called, then the agent returns an `AgentResult[Verdict]`.

**AC-1.2:** Before any Perplexity query, the agent checks institutional memory using cache key: `(agent_name="evidence_finding", normalized_query, model_name)`. Note: no `kb_index_version` in the key (unlike fact check). If a cache hit occurs, the cached `Verdict` is returned immediately with `prompt="[cache-hit]"`.

**AC-1.3:** On cache miss, the agent calls `perplexity_client.search()` with the concern's `review_note` as query, the configured `perplexity_model`, and `timeout_seconds` from the agent's LLM config.

**AC-1.4:** Citations are extracted from the Perplexity response via `_extract_citations()`: if `response["citations"]` is a list, filter to string items. Otherwise, return empty list.

**AC-1.5:** The full Perplexity response (JSON-serialized) is injected into the prompt as `{web_evidence}`.

**AC-1.6:** The prompt is formatted with: `{style_requirements}`, `{concern}`, `{article_excerpt}`, `{source_text}`, `{source_metadata}`, `{web_evidence}`.

**AC-1.7:** After a successful LLM evaluation, the agent persists an `EvidenceRecord` to institutional memory containing: timestamp, article_id, concern_id, prompt, query, normalized_query, model_name, cache_key_hash, perplexity_response, citations, and verdict.

---

### US-2: As the orchestrator, I want the evidence finding verdict to include citations, so that the writer can add properly sourced footnotes.

**AC-2.1:** The `Verdict` object returned contains all standard fields (concern_id, misleading, status, rationale, suggested_fix) plus:
- `evidence` (str | None) — brief summary of what the web search found
- `citations` (list[str] | None) — URLs returned by Perplexity; these are real URLs from the search results, never invented by the LLM

**AC-2.2:** Citations in the verdict flow through the feedback loop to the writer agent, which uses them to create footnotes using `[^n]` markdown references. The pipeline rule: **citations must come from the search results, never from the LLM's own knowledge.**

---

### US-3: As the orchestrator, I want the evidence finding prompt to guide nuanced evaluation of external evidence, so that the agent correctly handles supporting, contradicting, and absent evidence.

**AC-3.1:** The evidence finding prompt MUST instruct the LLM to handle these scenarios:

**Supporting evidence found:**
- `misleading=false`, `status=KEEP`
- Main article text must NOT be modified to add new factual claims from the evidence
- Supporting evidence MAY be included as a footnote that provides additional context and cites the source
- `suggested_fix`: recommend adding a footnote with the supporting citation if it strengthens the article
- `citations`: include relevant URLs from the web search

**Contradicting evidence found:**
- `misleading=true`, `status=REWRITE` or `REMOVE`
- The article claim must be corrected or removed to avoid misleading readers
- A footnote MAY summarize the contradiction and cite the sources
- `suggested_fix`: specific correction instruction, e.g., "Rewrite to note that [source] contradicts this claim" or "Remove this claim; [source] provides contradicting evidence"
- `citations`: include the contradicting source URLs

**No evidence found (web search returns nothing relevant):**
- Evaluate based on whether the claim is reasonable from source context alone
- If the claim is a specific factual assertion with no external support: `misleading=true`, `status=REWRITE`, suggest adding qualification
- If the claim is a reasonable inference or a common-knowledge statement: `misleading=false`, `status=KEEP`

**Ambiguous or mixed evidence:**
- `misleading=false` if the balance of evidence doesn't clearly contradict the claim
- `status=REWRITE` if qualification would improve accuracy
- `suggested_fix`: suggest hedging language or attribution

**AC-3.2:** The evidence finding prompt MUST enforce the external evidence usage rules from the requirements doc:
- External evidence may ONLY be introduced as clearly labeled footnotes
- External evidence must NEVER be silently injected as new main-text facts
- The writer decides how to use the evidence; the specialist only provides the verdict and citations

**AC-3.3:** The evidence finding prompt MUST instruct the LLM to:
1. Read the web evidence and assess its relevance and credibility
2. Compare the article's claim against both the web evidence AND the source text
3. Distinguish between evidence that supports/refutes the specific claim vs tangentially related information
4. Provide a brief, human-readable `evidence` summary (not raw JSON)
5. Include only relevant URLs in `citations`
6. Output strict JSON matching the Verdict schema

---

### US-4: As a developer, I want institutional memory to prevent redundant Perplexity API calls.

**AC-4.1:** Cache key hash: `sha256("|".join(["evidence_finding", normalized_query, model_name])).hexdigest()[:16]`.

**AC-4.2:** No `kb_index_version` in the evidence cache key (unlike fact check) because Perplexity results are not tied to a local index version.

**AC-4.3:** Cache records are persisted at `{institutional_memory_dir}/evidence_finding/{YYYY-MM-DD}/{cache_key_hash}.json`.

**AC-4.4:** The `EvidenceRecord` includes the full `perplexity_response` and extracted `citations` list, so cached verdicts retain their citation information.

---

### US-5: As a developer, I want a mock evidence finding agent for testing without Perplexity API calls.

**AC-5.1:** `MockEvidenceFindingAgent` implements `SpecialistAgentProtocol` structurally.

**AC-5.2:** Returns a static KEEP verdict with `misleading=False` and `concern_id` matching the input.

**AC-5.3:** The agent factory instantiates `MockEvidenceFindingAgent` when `agent_name == "mock"`.

## Constraints

### Technical
- Extends `BaseSpecialistAgent`
- Receives `PerplexityClient` and `InstitutionalMemoryStore` via constructor injection
- Perplexity API requires HTTPS (enforced by `PerplexityHTTPClient`)
- Perplexity API key is a required non-empty string in config

### Configuration
- Prompt file under `article_generation.editor.prompts.specialists.evidence_finding_prompt_file`
- LLM parameters under `article_generation.agents.specialists.evidence_finding`
- Temperature: 0.2
- Perplexity config under `article_generation.perplexity`: api_base, api_key, model, timeout_seconds

### Perplexity API
- OpenAI-compatible HTTPS API at `api.perplexity.ai`
- Model: configured in `perplexity.model` (e.g., `"sonar"`)
- Returns search results with a `citations` array of URLs
- Raw HTTP client (not using litellm or any SDK)

## Edge Cases

### EC-1: Perplexity returns no citations
`_extract_citations()` returns `[]`. The LLM evaluates based on the response content alone. The verdict may have `citations=null` or `citations=[]`.

### EC-2: Perplexity API timeout
`perplexity_client.search()` raises. The exception propagates (fail-fast).

### EC-3: Perplexity API returns non-JSON response
`perplexity_client.search()` raises during JSON parsing. Fail-fast.

### EC-4: Perplexity API key missing or empty
Validated at config load time. The pipeline fails before any agent is constructed.

### EC-5: Cache hit from a different article
Valid behavior — the same query/model produces the same evidence regardless of article context. The cached verdict (including citations) is reused.

### EC-6: Citations contain non-URL strings
`_extract_citations()` filters to strings but doesn't validate URL format. Non-URL citation strings are passed through and may appear in footnotes.

## Non-Goals

- **The evidence finding agent does not modify the article.** It produces a verdict.
- **The evidence finding agent does not query the knowledge base.** That's the Fact Check Agent.
- **The evidence finding agent does not evaluate style or attribution.** Only evidence availability.
- **The evidence finding agent does not decide how evidence appears in the article.** The writer uses citations to format footnotes.

## Prompt Gap Analysis

The current prompt (29 lines) has the same structure as fact check but lacks:
1. Detailed scenario handling (AC-3.1) — supporting vs contradicting vs absent evidence
2. External evidence usage rules (AC-3.2) — footnotes only, no silent injection
3. Citation handling instructions
4. Guidance on `misleading` threshold

**The prompt needs to be rewritten.** The agent code requires no changes.

## Open Questions

None.
