# Fact Check Agent — Behavioral Specification

## Objective

The Fact Check Agent validates factual claims in a generated article against a pre-indexed knowledge base (RAG). When the Article Review Agent flags a concern that may involve a verifiable factual claim, the Concern Mapping Agent routes it to this specialist. The Fact Check Agent queries the knowledge base for relevant evidence, then uses an LLM to evaluate whether the article's claim is accurate, misleading, or unverifiable. It also integrates with the institutional memory cache to avoid redundant knowledge base queries across runs.

## Existing Implementation

- **Real agent:** `src/agents/article_generation/specialists/fact_check/agent.py` — fully implemented (code complete)
- **Mock agent:** `src/agents/article_generation/specialists/fact_check/mock_agent.py` — returns static KEEP verdict
- **Prompt template:** `prompts/article_editor/specialists/fact_check.md` — **STUB (29 lines)**. The prompt lacks detailed guidance on how to evaluate KB evidence against the concern.
- **Knowledge base:** `src/agents/article_generation/knowledge_base/` — fully implemented (indexer, retriever, embedding client)
- **Institutional memory:** `src/agents/article_generation/chief_editor/institutional_memory.py` — fully implemented

## User Stories & Acceptance Criteria

### US-1: As the orchestrator, I want the fact check agent to evaluate factual concerns against the knowledge base, so that verifiable claims are validated with evidence.

**AC-1.1:** Given a `Concern`, `ArticleResponse`, source_text, source_metadata, and style_requirements, when `evaluate()` is called, then the agent returns an `AgentResult[Verdict]`.

**AC-1.2:** Before any KB query, the agent checks institutional memory for a cached result using the cache key: `(agent_name="fact_check", normalized_query, model_name, kb_index_version)`. If a cache hit occurs, the cached `Verdict` is returned immediately without KB or LLM calls. The `AgentResult.prompt` is set to `"[cache-hit]"`.

**AC-1.3:** On cache miss, the agent queries the knowledge base with `top_k=5` and the configured `kb_timeout_seconds`.

**AC-1.4:** The KB results (list of `{source_path, chunk_id, snippet, score}` dicts) are JSON-serialized and injected into the prompt as `{kb_evidence}`.

**AC-1.5:** The agent loads the prompt template from `prompts/article_editor/specialists/fact_check.md` via `PromptLoader.load_specialist_prompt()`.

**AC-1.6:** The prompt is formatted with: `{style_requirements}`, `{concern}` (review_note), `{article_excerpt}` (excerpt), `{source_text}`, `{source_metadata}` (JSON), `{kb_evidence}` (JSON).

**AC-1.7:** The LLM response is parsed into a `Verdict` object. Invalid JSON or schema mismatch raises `ValidationError`.

**AC-1.8:** After a successful LLM evaluation, the agent persists a `FactCheckRecord` to institutional memory containing: timestamp, article_id, concern_id, prompt, query, normalized_query, model_name, kb_index_version, cache_key_hash, kb_response, and verdict.

---

### US-2: As the orchestrator, I want the fact check verdict to clearly indicate whether the claim is misleading and what action to take.

**AC-2.1:** The `Verdict` object returned contains:
- `concern_id` (int) — echoes the input concern's ID
- `misleading` (bool) — `True` if the factual claim is incorrect, contradicted by evidence, or unverifiable and presented as fact; `False` if supported or reasonably inferable
- `status` — one of:
  - `"KEEP"` — claim is accurate or reasonably supported; no changes needed
  - `"REWRITE"` — claim needs qualification, correction, or additional context
  - `"REMOVE"` — claim is factually incorrect and cannot be salvaged by rewriting
- `rationale` (str) — 1-3 sentences explaining the reasoning
- `suggested_fix` (str | None) — minimal correction or rewrite instruction; `None` if KEEP
- `evidence` (str | None) — brief summary of KB evidence supporting the verdict
- `citations` (list[str] | None) — relevant citations from KB; these can be used by the writer for footnotes

---

### US-3: As the orchestrator, I want the fact check prompt to provide detailed evaluation guidance, so that the LLM makes well-reasoned factual judgments.

**AC-3.1:** The fact check prompt MUST instruct the LLM to:
1. Read the KB evidence carefully and assess its relevance to the concern
2. Compare the article's factual claim against the KB evidence AND the source text
3. Determine if the claim is: (a) directly supported by KB evidence, (b) contradicted by KB evidence, (c) not addressed by KB evidence (unverifiable from this KB)
4. Consider whether the claim could be a reasonable inference from the source text even if KB evidence is absent
5. Set `misleading=true` only when the claim is demonstrably false, contradicted by evidence, or presents unverifiable speculation as established fact
6. Set `misleading=false` when the claim is supported by evidence, is a standard journalistic descriptor (e.g., "appearance at an event"), or is a reasonable inference clearly attributable to the source

**AC-3.2:** The fact check prompt MUST instruct the LLM to handle these specific scenarios:
- **KB confirms the claim:** `status=KEEP`, `misleading=false`, cite the KB evidence
- **KB contradicts the claim:** `status=REWRITE` or `REMOVE`, `misleading=true`, explain the contradiction, provide corrected information in `suggested_fix`
- **KB is silent (no relevant evidence):** `status` depends on whether the claim is reasonable from source context alone. If it's a specific factual assertion with no support, `status=REWRITE` with suggestion to add qualification ("according to..."). If it's a standard inference, `status=KEEP`.
- **KB evidence is ambiguous:** `status=REWRITE`, suggest adding qualification/hedging language

**AC-3.3:** The fact check prompt MUST instruct the LLM to provide `evidence` as a brief, human-readable summary of the relevant KB findings (not raw JSON).

**AC-3.4:** The fact check prompt MUST instruct the LLM to include `citations` only when KB evidence directly supports or refutes the claim. Citations should reference KB source documents (not invented URLs).

**AC-3.5:** The fact check prompt MUST instruct the LLM to output strict JSON matching the Verdict schema. No preamble, no commentary.

---

### US-4: As a developer, I want institutional memory to prevent redundant KB queries across runs.

**AC-4.1:** Cache key hash is computed as `sha256("|".join(["fact_check", normalized_query, model_name, kb_index_version])).hexdigest()[:16]`.

**AC-4.2:** Query normalization: `re.sub(r"\s+", " ", query.strip().lower())`. This ensures minor whitespace/casing differences don't cause cache misses.

**AC-4.3:** Cache lookup searches all date-partitioned directories (`YYYY-MM-DD/`) under the fact-checking subdirectory and returns the most recent match.

**AC-4.4:** Cache records are persisted at `{institutional_memory_dir}/fact_checking/{YYYY-MM-DD}/{cache_key_hash}.json`.

**AC-4.5:** The `kb_index_version` is included in the cache key so that when the KB is re-indexed (new documents, changed embedding model), old cache entries are effectively invalidated.

---

### US-5: As a developer, I want a mock fact check agent for testing without KB or LLM calls.

**AC-5.1:** `MockFactCheckAgent` implements `SpecialistAgentProtocol` structurally.

**AC-5.2:** `MockFactCheckAgent.evaluate()` returns a static KEEP verdict with `misleading=False` for any input concern. The `concern_id` in the verdict matches the input concern's ID.

**AC-5.3:** The agent factory instantiates `MockFactCheckAgent` when `agent_name == "mock"`.

## Constraints

### Technical
- Extends `BaseSpecialistAgent` (which extends `BaseAgent`)
- Receives `KnowledgeBaseRetriever` and `InstitutionalMemoryStore` via constructor injection
- KB queries use `top_k=5` (hardcoded in the agent)
- Cache key includes `kb_index_version` — changes when KB is re-indexed
- `_normalize_query()` and `_build_article_id()` are inherited from `BaseSpecialistAgent`

### Configuration
- Prompt file under `article_generation.editor.prompts.specialists.fact_check_prompt_file`
- LLM parameters under `article_generation.agents.specialists.fact_check`
- Temperature: 0.2 (deterministic evaluation preferred)
- KB timeout from `article_generation.knowledge_base.timeout_seconds`

### Knowledge Base
- Source documents: `data/knowledgebase/*.txt` and `*.md`
- Index format: `chunks.json` + `embeddings.npy` + `manifest.json`
- Embedding: `text-embedding-bge-large-en-v1.5` via LM Studio
- Chunking: 512 tokens, 50 token overlap
- Retrieval: cosine similarity, top-k results

## Edge Cases

### EC-1: KB returns zero results
The `{kb_evidence}` is `"[]"`. The LLM must evaluate the concern based on source text alone, without KB support. The prompt should handle this gracefully.

### EC-2: KB evidence contradicts source text
The KB may contain information that contradicts the source transcript. The fact check agent should prioritize KB evidence for factual accuracy but note the discrepancy.

### EC-3: Cache hit returns verdict from a previous article
This is correct behavior — if the same query/model/KB version produces the same result, the cached verdict is valid regardless of which article triggered it.

### EC-4: KB timeout
The `knowledge_base_retriever.search()` raises on timeout. The exception propagates to the orchestrator (fail-fast per article).

### EC-5: Source metadata missing `source_file` or `topic_slug`
`_build_article_id()` raises `ValueError`. This happens after the LLM call but before cache persistence, so the verdict is lost. The orchestrator fails the article.

## Non-Goals

- **The fact check agent does not modify the article.** It only produces a verdict.
- **The fact check agent does not search the web.** Web search is the Evidence Finding Agent's job.
- **The fact check agent does not evaluate writing style.** Only factual accuracy.
- **The fact check agent does not re-index the KB.** Indexing is done at startup by the agent factory.

## Prompt Gap Analysis

The current prompt (29 lines) provides the task description and output schema but lacks:
1. Detailed evaluation guidance (AC-3.1) — how to weigh KB evidence
2. Scenario-specific instructions (AC-3.2) — what to do when KB confirms/contradicts/is silent
3. Instructions for handling ambiguous evidence
4. Instructions for citation formatting
5. Guidance on when `misleading=true` vs `misleading=false`

**The prompt needs to be rewritten.** The agent code requires no changes.

## Open Questions

None.
