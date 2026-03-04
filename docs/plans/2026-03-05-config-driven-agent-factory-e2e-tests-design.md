# Config-Driven AgentFactory & E2E Tests

**Created:** 2026-03-05
**Status:** Approved

## Problem

The article generation system has no end-to-end test coverage. All agents are hardcoded in `build_chief_editor_orchestrator()`, making it impossible to swap implementations for testing without code changes.

## Design

### Config-Driven Agent Selection

Each agent slot in `config.yaml` gains an `implementation` field that the `AgentFactory` uses to select the concrete class. The LLM config moves under a `llm:` sub-key.

**Production config:**

```yaml
article_generation:
  agents:
    writer:
      implementation: default
      llm: { model: ..., api_base: ..., ... }
    article_review:
      implementation: default
      llm: { model: ..., api_base: ..., ... }
    concern_mapping:
      implementation: default
      llm: { model: ..., api_base: ..., ... }
    specialists:
      fact_check:
        implementation: default
        llm: { model: ..., api_base: ..., ... }
      evidence_finding:
        implementation: default
        llm: { model: ..., api_base: ..., ... }
      opinion:
        implementation: default
        llm: { model: ..., api_base: ..., ... }
      attribution:
        implementation: default
        llm: { model: ..., api_base: ..., ... }
      style_review:
        implementation: default
        llm: { model: ..., api_base: ..., ... }
```

**Test config (`config/config.test.yaml`):**

Same structure but with `implementation: mock` for agents that depend on external services (fact_check, evidence_finding). All other agents use `implementation: default` with a real LLM (LM Studio).

### AgentFactory

A single factory function replaces the current `build_chief_editor_orchestrator()`. For each agent slot it:

1. Reads `implementation` from config
2. Maps the name to a concrete class via a registry
3. Constructs the agent with appropriate dependencies

The registry maps implementation names to classes:

- `"default"` → production agent class (e.g., `FactCheckAgent`, `EvidenceFindingAgent`)
- `"mock"` → mock agent class (e.g., `MockFactCheckAgent`, `MockEvidenceFindingAgent`)

### Mock Agents

Mock agents are concrete classes that implement the same interface as their production counterparts but skip external dependencies entirely and return static verdicts.

**Naming convention:** All mock agents use the prefix `Mock` (e.g., `MockFactCheckAgent`, `MockEvidenceFindingAgent`).

**Behavior:** `evaluate()` returns a static `Verdict` with `misleading=False`, `status="KEEP"`, and a rationale indicating it's a mock. No LLM calls, no KB queries, no Perplexity searches.

**Location:** `src/agents/article_generation/specialists/fact_check/mock_agent.py` and `src/agents/article_generation/specialists/evidence_finding/mock_agent.py`.

### E2E Test

**File:** `tests/e2e/test_full_pipeline.py`

**Setup:**
- Uses `config/config.test.yaml` (real LLM, mock specialists for KB/Perplexity-dependent agents)
- Creates a temp directory with a minimal article input bundle (short transcript, manifest.json, topics.json)

**Execution:**
- Builds orchestrator via `AgentFactory` using test config
- Calls `orchestrator.generate_article()` with the sample bundle

**Assertions:**
- `ArticleGenerationResult` is structurally valid
- `result.article` is not None and has non-empty fields
- `result.editor_report` is not None with at least one iteration
- Output files exist in the artifacts directory
- Canonical output JSON exists

### Breaking Changes

The `agents:` config structure changes:
- LLM fields move under a `llm:` sub-key per agent
- New `implementation` field per agent

Per project rules: no backwards compatibility. Both `config.yaml` and `config.yaml.template` must be updated.

## Components Affected

- `src/config.py` — config models for new agent config shape
- `src/agents/article_generation/agent.py` — becomes `AgentFactory`
- `src/agents/article_generation/specialists/fact_check/mock_agent.py` — new
- `src/agents/article_generation/specialists/evidence_finding/mock_agent.py` — new
- `config/config.yaml` — restructured agents section
- `config/config.yaml.template` — restructured agents section
- `config/config.test.yaml` — new test config
- `tests/e2e/test_full_pipeline.py` — new E2E test
- Existing tests that reference agent config — updated for new shape
