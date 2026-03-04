# Config-Driven AgentFactory & E2E Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add config-driven agent factory so agent implementations (default vs Mock*) are selected via config.yaml, then write an E2E test that exercises the full orchestrator against a real LLM with mock specialists for KB/Perplexity.

**Architecture:** Each agent config block gains an `implementation` field. An `AgentFactory` reads this field and instantiates the correct class. `MockFactCheckAgent` and `MockEvidenceFindingAgent` return static KEEP verdicts. The E2E test uses `config/config.test.yaml` pointing at LM Studio with mock specialists.

**Tech Stack:** Python 3.12, Pydantic, pytest, uv, LM Studio (real LLM for E2E)

---

### Task 1: Restructure agent config models

**Files:**
- Modify: `src/config.py` (lines 170-191, the `ArticleGenerationSpecialistLLMConfig` and `ArticleGenerationAgentsConfig` classes)

**Step 1: Write the failing test**

Add to `tests/test_article_generation_config.py`:

```python
class TestAgentConfigWithImplementation:
    """Tests for agent config with implementation field."""

    def test_agent_config_accepts_implementation_field(self) -> None:
        """Agent config block must accept implementation + llm sub-key."""
        from src.config import AgentSlotConfig

        payload = {
            "implementation": "default",
            "llm": get_valid_llm_config(),
        }
        slot = AgentSlotConfig.model_validate(payload)
        assert slot.implementation == "default"
        assert slot.llm.model == "test-model"

    def test_agent_config_implementation_required(self) -> None:
        """Missing implementation field must fail."""
        from src.config import AgentSlotConfig

        payload = {"llm": get_valid_llm_config()}
        with pytest.raises(ValidationError):
            AgentSlotConfig.model_validate(payload)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_article_generation_config.py::TestAgentConfigWithImplementation -v`
Expected: FAIL — `AgentSlotConfig` does not exist yet

**Step 3: Write minimal implementation**

In `src/config.py`, add a new model and replace the flat LLM fields with the new nested structure:

```python
class AgentSlotConfig(BaseModel):
    """Configuration for a single agent slot with swappable implementation."""

    implementation: str = Field(..., min_length=1, description="Agent implementation name (e.g., 'default', 'mock')")
    llm: LLMConfig = Field(..., description="LLM configuration for this agent")

    model_config = ConfigDict(frozen=True, extra="forbid")
```

Replace `ArticleGenerationSpecialistLLMConfig`:

```python
class ArticleGenerationSpecialistAgentsConfig(BaseModel):
    """Specialist agent configurations."""

    fact_check: AgentSlotConfig = Field(..., description="Fact-check specialist")
    evidence_finding: AgentSlotConfig = Field(..., description="Evidence-finding specialist")
    opinion: AgentSlotConfig = Field(..., description="Opinion specialist")
    attribution: AgentSlotConfig = Field(..., description="Attribution specialist")
    style_review: AgentSlotConfig = Field(..., description="Style-review specialist")

    model_config = ConfigDict(frozen=True, extra="forbid")
```

Replace `ArticleGenerationAgentsConfig`:

```python
class ArticleGenerationAgentsConfig(BaseModel):
    """Agent configurations for article generation."""

    writer: AgentSlotConfig = Field(..., description="Writer agent")
    article_review: AgentSlotConfig = Field(..., description="Article-review agent")
    concern_mapping: AgentSlotConfig = Field(..., description="Concern-mapping agent")
    specialists: ArticleGenerationSpecialistAgentsConfig = Field(..., description="Specialist agents")

    model_config = ConfigDict(frozen=True, extra="forbid")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_article_generation_config.py::TestAgentConfigWithImplementation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/config.py tests/test_article_generation_config.py
git commit -m "feat(config): add AgentSlotConfig with implementation field"
```

---

### Task 2: Update existing config tests for new agent config shape

**Files:**
- Modify: `tests/test_article_generation_config.py`

**Step 1: Update `get_valid_article_generation_config_dict()` helper**

The helper must produce the new nested shape. All field references like `agents.writer_llm` become `agents.writer.llm`:

```python
def get_valid_agent_slot(implementation: str = "default") -> dict[str, object]:
    """Return a valid agent slot config payload."""
    return {
        "implementation": implementation,
        "llm": get_valid_llm_config(),
    }


def get_valid_article_generation_config_dict() -> dict[str, object]:
    """Return valid article_generation section payload."""
    return {
        "editor": {
            "editor_max_rounds": 3,
            "output": {
                "final_articles_dir": "./data/output/articles",
                "run_artifacts_dir": "./data/output/article_editor_runs",
                "save_intermediate_results": True,
            },
            "prompts": {
                "root_dir": "./prompts/article_editor",
                "writer_prompt_file": "writer.md",
                "revision_prompt_file": "revision.md",
                "article_review_prompt_file": "article_review.md",
                "concern_mapping_prompt_file": "concern_mapping.md",
                "specialists_dir": "specialists",
            },
        },
        "agents": {
            "writer": get_valid_agent_slot(),
            "article_review": get_valid_agent_slot(),
            "concern_mapping": get_valid_agent_slot(),
            "specialists": {
                "fact_check": get_valid_agent_slot(),
                "evidence_finding": get_valid_agent_slot(),
                "opinion": get_valid_agent_slot(),
                "attribution": get_valid_agent_slot(),
                "style_review": get_valid_agent_slot(),
            },
        },
        "knowledge_base": {
            "data_dir": "./data/knowledgebase",
            "index_dir": "./data/knowledgebase_index",
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 50,
            "timeout_seconds": 30,
            "embedding": {
                "provider": "lmstudio",
                "model_name": "text-embedding-bge-large-en-v1.5",
                "api_base": "http://127.0.0.1:1234/v1",
                "api_key": "lm-studio",
                "timeout_seconds": 30,
            },
        },
        "perplexity": {
            "api_base": "https://api.perplexity.ai",
            "api_key": "key",
            "model": "sonar",
            "timeout_seconds": 45,
        },
        "institutional_memory": {
            "data_dir": "./data/institutional_memory",
            "fact_checking_subdir": "fact_checking",
            "evidence_finding_subdir": "evidence_finding",
        },
        "allowed_styles": ["NATURE_NEWS", "SCIAM_MAGAZINE"],
        "default_style_mode": "SCIAM_MAGAZINE",
    }
```

**Step 2: Update assertions that reference old field paths**

In `TestArticleGenerationConfig.test_valid_article_generation_config`:
```python
# Old: config.agents.writer_llm.timeout_seconds
# New: config.agents.writer.llm.timeout_seconds
assert config.agents.writer.llm.timeout_seconds == 60
```

In `TestConfigIntegration.test_config_loads_article_generation`:
```python
# Old: config.get_article_timeout_seconds() which reads agents.writer_llm.timeout_seconds
# Update Config.get_article_timeout_seconds() to read agents.writer.llm.timeout_seconds
```

**Step 3: Run all tests**

Run: `uv run pytest tests/test_article_generation_config.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add tests/test_article_generation_config.py
git commit -m "test: update article generation config tests for new agent slot shape"
```

---

### Task 3: Update Config class getters for new agent config shape

**Files:**
- Modify: `src/config.py` — `Config.get_article_timeout_seconds()` (line 618-629)

**Step 1: Update getter**

```python
def get_article_timeout_seconds(self) -> int:
    if not hasattr(self, "_article_generation"):
        raise KeyError("Missing required key 'article_generation' in config file")
    return self._article_generation.agents.writer.llm.timeout_seconds
```

**Step 2: Run tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/config.py
git commit -m "refactor(config): update getters for new agent slot config shape"
```

---

### Task 4: Create MockFactCheckAgent and MockEvidenceFindingAgent

**Files:**
- Create: `src/agents/article_generation/specialists/fact_check/mock_agent.py`
- Create: `src/agents/article_generation/specialists/evidence_finding/mock_agent.py`

**Step 1: Write tests for mock agents**

Create `tests/test_mock_agents.py`:

```python
"""Tests for mock specialist agents."""

from src.agents.article_generation.models import ArticleResponse, Concern, Verdict
from src.agents.article_generation.specialists.evidence_finding.mock_agent import MockEvidenceFindingAgent
from src.agents.article_generation.specialists.fact_check.mock_agent import MockFactCheckAgent


class TestMockFactCheckAgent:
    """Tests for MockFactCheckAgent."""

    def test_returns_keep_verdict(self) -> None:
        """Mock agent returns a static KEEP verdict."""
        agent = MockFactCheckAgent()
        concern = Concern(concern_id=1, excerpt="some claim", review_note="review note")
        article = ArticleResponse(
            headline="Test",
            alternative_headline="Alt",
            article_body="Body text",
            description="Desc",
        )
        verdict = agent.evaluate(
            concern=concern,
            article=article,
            source_text="source",
            source_metadata={"source_file": "f", "topic_slug": "s"},
            style_requirements="SCIAM_MAGAZINE",
        )
        assert isinstance(verdict, Verdict)
        assert verdict.concern_id == 1
        assert verdict.misleading is False
        assert verdict.status == "KEEP"

    def test_concern_id_matches_input(self) -> None:
        """Verdict concern_id must match the input concern."""
        agent = MockFactCheckAgent()
        concern = Concern(concern_id=42, excerpt="claim", review_note="note")
        article = ArticleResponse(
            headline="H", alternative_headline="AH", article_body="B", description="D",
        )
        verdict = agent.evaluate(
            concern=concern, article=article, source_text="s",
            source_metadata={"source_file": "f", "topic_slug": "s"},
            style_requirements="SCIAM_MAGAZINE",
        )
        assert verdict.concern_id == 42


class TestMockEvidenceFindingAgent:
    """Tests for MockEvidenceFindingAgent."""

    def test_returns_keep_verdict(self) -> None:
        """Mock agent returns a static KEEP verdict."""
        agent = MockEvidenceFindingAgent()
        concern = Concern(concern_id=3, excerpt="some claim", review_note="review note")
        article = ArticleResponse(
            headline="Test",
            alternative_headline="Alt",
            article_body="Body text",
            description="Desc",
        )
        verdict = agent.evaluate(
            concern=concern,
            article=article,
            source_text="source",
            source_metadata={"source_file": "f", "topic_slug": "s"},
            style_requirements="SCIAM_MAGAZINE",
        )
        assert isinstance(verdict, Verdict)
        assert verdict.concern_id == 3
        assert verdict.misleading is False
        assert verdict.status == "KEEP"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mock_agents.py -v`
Expected: FAIL — mock agent modules do not exist

**Step 3: Implement MockFactCheckAgent**

Create `src/agents/article_generation/specialists/fact_check/mock_agent.py`:

```python
"""Mock fact-check specialist agent for testing."""

from __future__ import annotations

import logging

from src.agents.article_generation.models import ArticleResponse, Concern, Verdict

logger = logging.getLogger(__name__)


class MockFactCheckAgent:
    """Returns static KEEP verdicts without KB or LLM calls."""

    def evaluate(
        self,
        *,
        concern: Concern,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_requirements: str,
    ) -> Verdict:
        """Return a static KEEP verdict."""
        logger.info("MockFactCheckAgent: returning static KEEP for concern #%d", concern.concern_id)
        return Verdict(
            concern_id=concern.concern_id,
            misleading=False,
            status="KEEP",
            rationale="Mock fact-check: no knowledge base available in test configuration",
            suggested_fix=None,
            evidence=None,
            citations=None,
        )
```

**Step 4: Implement MockEvidenceFindingAgent**

Create `src/agents/article_generation/specialists/evidence_finding/mock_agent.py`:

```python
"""Mock evidence-finding specialist agent for testing."""

from __future__ import annotations

import logging

from src.agents.article_generation.models import ArticleResponse, Concern, Verdict

logger = logging.getLogger(__name__)


class MockEvidenceFindingAgent:
    """Returns static KEEP verdicts without Perplexity or LLM calls."""

    def evaluate(
        self,
        *,
        concern: Concern,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_requirements: str,
    ) -> Verdict:
        """Return a static KEEP verdict."""
        logger.info("MockEvidenceFindingAgent: returning static KEEP for concern #%d", concern.concern_id)
        return Verdict(
            concern_id=concern.concern_id,
            misleading=False,
            status="KEEP",
            rationale="Mock evidence-finding: no Perplexity API available in test configuration",
            suggested_fix=None,
            evidence=None,
            citations=None,
        )
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_mock_agents.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/agents/article_generation/specialists/fact_check/mock_agent.py src/agents/article_generation/specialists/evidence_finding/mock_agent.py tests/test_mock_agents.py
git commit -m "feat: add MockFactCheckAgent and MockEvidenceFindingAgent"
```

---

### Task 5: Create AgentFactory

**Files:**
- Modify: `src/agents/article_generation/agent.py` — refactor `build_chief_editor_orchestrator()` to use factory pattern

**Step 1: Write the failing test**

Create `tests/test_agent_factory.py`:

```python
"""Tests for config-driven AgentFactory."""

import tempfile
from pathlib import Path

import yaml

from src.agents.article_generation.agent import build_chief_editor_orchestrator
from src.agents.article_generation.specialists.evidence_finding.mock_agent import MockEvidenceFindingAgent
from src.agents.article_generation.specialists.fact_check.mock_agent import MockFactCheckAgent
from src.config import Config


def _write_test_config(tmp_dir: Path, specialist_overrides: dict[str, str] | None = None) -> Path:
    """Write a minimal config.yaml for factory tests.

    Args:
        tmp_dir: Temporary directory for config and data.
        specialist_overrides: Dict of specialist_name -> implementation to override.
    """
    llm = {
        "model": "test-model",
        "api_base": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
        "context_window": 32768,
        "max_tokens": 2048,
        "temperature": 0.3,
        "context_window_threshold": 90,
        "max_retries": 0,
        "retry_delay": 2.0,
        "timeout_seconds": 60,
    }

    def agent_slot(impl: str = "default") -> dict:
        return {"implementation": impl, "llm": llm}

    overrides = specialist_overrides or {}

    # Create required directories
    kb_data_dir = tmp_dir / "knowledgebase"
    kb_data_dir.mkdir(parents=True, exist_ok=True)
    kb_index_dir = tmp_dir / "knowledgebase_index"
    kb_index_dir.mkdir(parents=True, exist_ok=True)
    im_dir = tmp_dir / "institutional_memory"
    im_dir.mkdir(parents=True, exist_ok=True)
    articles_dir = tmp_dir / "output" / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = tmp_dir / "output" / "article_editor_runs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "paths": {
            "data_dir": str(tmp_dir),
            "data_models_dir": str(tmp_dir / "models"),
            "data_downloads_dir": str(tmp_dir / "downloads"),
            "data_downloads_videos_dir": str(tmp_dir / "downloads" / "videos"),
            "data_downloads_transcripts_dir": str(tmp_dir / "downloads" / "transcripts"),
            "data_downloads_transcripts_hallucinations_dir": str(tmp_dir / "downloads" / "transcripts-hallucinations"),
            "data_downloads_transcripts_cleaned_dir": str(tmp_dir / "downloads" / "transcripts_cleaned"),
            "data_transcripts_topics_dir": str(tmp_dir / "downloads" / "transcripts-topics"),
            "data_downloads_audio_dir": str(tmp_dir / "downloads" / "audio"),
            "data_downloads_metadata_dir": str(tmp_dir / "downloads" / "metadata"),
            "data_output_dir": str(tmp_dir / "output"),
            "data_input_dir": str(tmp_dir / "input"),
            "data_temp_dir": str(tmp_dir / "temp"),
            "data_archive_dir": str(tmp_dir / "archive"),
            "data_archive_videos_dir": str(tmp_dir / "archive" / "videos"),
            "data_logs_dir": str(tmp_dir / "logs"),
            "data_output_articles_dir": str(articles_dir),
            "data_articles_input_dir": str(tmp_dir / "articles" / "input"),
            "reports_dir": str(tmp_dir / "reports"),
        },
        "channels": [],
        "defaults": {
            "encoding_name": "o200k_base",
            "repetition_min_k": 1,
            "repetition_min_repetitions": 5,
            "detect_min_k": 3,
        },
        "article_generation": {
            "editor": {
                "editor_max_rounds": 3,
                "output": {
                    "final_articles_dir": str(articles_dir),
                    "run_artifacts_dir": str(artifacts_dir),
                    "save_intermediate_results": True,
                },
                "prompts": {
                    "root_dir": "./prompts/article_editor",
                    "writer_prompt_file": "writer.md",
                    "revision_prompt_file": "revision.md",
                    "article_review_prompt_file": "article_review.md",
                    "concern_mapping_prompt_file": "concern_mapping.md",
                    "specialists_dir": "specialists",
                },
            },
            "agents": {
                "writer": agent_slot(),
                "article_review": agent_slot(),
                "concern_mapping": agent_slot(),
                "specialists": {
                    "fact_check": agent_slot(overrides.get("fact_check", "default")),
                    "evidence_finding": agent_slot(overrides.get("evidence_finding", "default")),
                    "opinion": agent_slot(overrides.get("opinion", "default")),
                    "attribution": agent_slot(overrides.get("attribution", "default")),
                    "style_review": agent_slot(overrides.get("style_review", "default")),
                },
            },
            "knowledge_base": {
                "data_dir": str(kb_data_dir),
                "index_dir": str(kb_index_dir),
                "chunk_size_tokens": 512,
                "chunk_overlap_tokens": 50,
                "timeout_seconds": 30,
                "embedding": {
                    "provider": "lmstudio",
                    "model_name": "text-embedding-bge-large-en-v1.5",
                    "api_base": "http://127.0.0.1:1234/v1",
                    "api_key": "lm-studio",
                    "timeout_seconds": 30,
                },
            },
            "perplexity": {
                "api_base": "https://api.perplexity.ai",
                "api_key": "test-key",
                "model": "sonar",
                "timeout_seconds": 45,
            },
            "institutional_memory": {
                "data_dir": str(im_dir),
                "fact_checking_subdir": "fact_checking",
                "evidence_finding_subdir": "evidence_finding",
            },
            "allowed_styles": ["NATURE_NEWS", "SCIAM_MAGAZINE"],
            "default_style_mode": "SCIAM_MAGAZINE",
        },
    }

    config_path = tmp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


class TestAgentFactory:
    """Tests for config-driven agent factory."""

    def test_mock_specialists_selected_by_config(self, tmp_path: Path) -> None:
        """When config says implementation=mock, factory returns Mock* agents."""
        config_path = _write_test_config(
            tmp_path,
            specialist_overrides={
                "fact_check": "mock",
                "evidence_finding": "mock",
            },
        )
        config = Config(config_path)
        orchestrator = build_chief_editor_orchestrator(config=config)

        assert isinstance(orchestrator._fact_check_agent, MockFactCheckAgent)
        assert isinstance(orchestrator._evidence_finding_agent, MockEvidenceFindingAgent)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_factory.py -v`
Expected: FAIL — factory does not read `implementation` yet

**Step 3: Refactor `build_chief_editor_orchestrator()`**

Modify `src/agents/article_generation/agent.py` to read `implementation` from each agent slot config and dispatch to the correct class. For `"mock"` fact_check/evidence_finding, instantiate the Mock agents (which need no constructor args). For `"default"`, use the existing real agent construction. All other agents only support `"default"` for now; raise `ValueError` for unknown implementations.

Key changes:
- Read `article_generation_config.agents.writer.llm` instead of `article_generation_config.agents.writer_llm`
- Same for all other agents
- For `specialists.fact_check.implementation == "mock"`: instantiate `MockFactCheckAgent()`
- For `specialists.evidence_finding.implementation == "mock"`: instantiate `MockEvidenceFindingAgent()`
- Skip KB indexer/retriever construction entirely when fact_check is mock
- Skip Perplexity client construction entirely when evidence_finding is mock

**Step 4: Run tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/agents/article_generation/agent.py tests/test_agent_factory.py
git commit -m "feat: config-driven AgentFactory with mock agent dispatch"
```

---

### Task 6: Update config.yaml and config.yaml.template

**Files:**
- Modify: `config/config.yaml` — restructure agents section with `implementation` + `llm` nesting
- Modify: `config/config.yaml.template` — same restructuring

**Step 1: Update config.yaml**

Change every agent block from flat LLM fields to the new nested shape. Example for writer:

```yaml
# Before:
agents:
  writer_llm:
    model: openai/qwen3-30b-a3b-thinking-2507-mlx@8bit
    ...

# After:
agents:
  writer:
    implementation: default
    llm:
      model: openai/qwen3-30b-a3b-thinking-2507-mlx@8bit
      ...
```

Same for `article_review`, `concern_mapping`, and all specialists.

**Step 2: Update config.yaml.template**

Same structural change with placeholder values.

**Step 3: Run config validation tests**

Run: `uv run pytest tests/test_article_generation_config.py tests/test_config.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add config/config.yaml config/config.yaml.template
git commit -m "refactor(config): restructure agent configs with implementation + llm nesting"
```

---

### Task 7: Update generate-articles.py for new config paths

**Files:**
- Modify: `scripts/generate-articles.py` (line 75, 103) — update field access paths

**Step 1: Update field references**

```python
# Line ~75: api_base access
api_base = article_config.agents.writer.llm.api_base

# Line ~103: writer_model log line
article_config.agents.writer.llm.model,
```

**Step 2: Run tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add scripts/generate-articles.py
git commit -m "fix(scripts): update generate-articles for new agent config paths"
```

---

### Task 8: Create config.test.yaml

**Files:**
- Create: `config/config.test.yaml`

**Step 1: Create test config**

This file mirrors `config.yaml` but:
- Uses temp/relative paths for output directories
- Sets `implementation: mock` for `fact_check` and `evidence_finding`
- Points at real LM Studio for all LLM calls
- Sets `editor_max_rounds: 1` to keep E2E test fast

Note: The E2E test will create its own temp config at runtime (overriding output paths to tmp_dir), so `config.test.yaml` serves as a human-runnable reference and template. The test itself writes a programmatic config.

**Step 2: Commit**

```bash
git add config/config.test.yaml
git commit -m "feat(config): add config.test.yaml with mock specialists"
```

---

### Task 9: Create E2E test with sample bundle fixture

**Files:**
- Create: `tests/e2e/__init__.py`
- Create: `tests/e2e/test_full_pipeline.py`
- Create: `tests/fixtures/article_generation/sample_bundle/manifest.json`
- Create: `tests/fixtures/article_generation/sample_bundle/transcript.txt`
- Create: `tests/fixtures/article_generation/sample_bundle/topics.json`

**Step 1: Create sample bundle fixture**

`tests/fixtures/article_generation/sample_bundle/manifest.json`:
```json
{
    "article_title": "New Advances in Large Language Model Efficiency",
    "slug": "llm-efficiency-advances",
    "publish_date": "2026-03-01",
    "source_text_file": "transcript.txt",
    "topics_file": "topics.json",
    "references": [
        {
            "type": "video",
            "title": "LLM Efficiency Breakthrough",
            "url": "https://www.youtube.com/watch?v=test123",
            "author": "Test Author",
            "channel": "Test Channel",
            "date": "2026-03-01"
        }
    ]
}
```

`tests/fixtures/article_generation/sample_bundle/transcript.txt`:
A short (~300 word) transcript about LLM efficiency research. Real enough for the LLM to produce a coherent article.

`tests/fixtures/article_generation/sample_bundle/topics.json`:
```json
[{"title": "LLM Efficiency", "summary": "Advances in making large language models more efficient"}]
```

**Step 2: Write the E2E test**

`tests/e2e/test_full_pipeline.py`:

```python
"""End-to-end test for the full article generation pipeline.

Requires a running LM Studio instance. Mock specialists are used for
KB (fact-check) and Perplexity (evidence-finding) to avoid external deps.

Run with: uv run pytest tests/e2e/ -v -s
"""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from src.agents.article_generation.agent import build_chief_editor_orchestrator
from src.agents.article_generation.bundle_loader import bundle_to_source_metadata, load_bundle
from src.config import Config

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "article_generation" / "sample_bundle"

# Mark all tests in this module as E2E (slow, requires LM Studio)
pytestmark = pytest.mark.e2e


def _build_e2e_config(tmp_path: Path) -> Config:
    """Build a Config pointing at tmp_path for outputs, real LLM, mock specialists."""
    # ... (programmatic config similar to test_agent_factory._write_test_config
    #      but with real LLM model name from LM Studio and mock fact_check + evidence_finding)
    ...


class TestFullPipeline:
    """E2E test: orchestrator produces an article from a sample bundle."""

    def test_orchestrator_produces_article(self, tmp_path: Path) -> None:
        """Full pipeline produces a valid ArticleGenerationResult."""
        config = _build_e2e_config(tmp_path)
        orchestrator = build_chief_editor_orchestrator(config=config)

        bundle = load_bundle(FIXTURES_DIR)
        source_metadata = bundle_to_source_metadata(bundle)

        result = orchestrator.generate_article(
            source_text=bundle.source_text,
            source_metadata=source_metadata,
            style_mode="SCIAM_MAGAZINE",
            reader_preference="",
        )

        # Structural assertions
        assert result.article is not None, f"Expected article, got error: {result.error}"
        assert result.article.headline != ""
        assert result.article.article_body != ""
        assert result.editor_report is not None
        assert result.editor_report.total_iterations >= 1
        assert result.artifacts_dir is not None

        # Output files exist
        artifacts = Path(result.artifacts_dir)
        assert artifacts.exists()
        assert (artifacts / "iter1_writer_draft.json").exists()
```

**Step 3: Register the `e2e` pytest marker**

In `pyproject.toml` or `pytest.ini`, add:
```ini
[tool.pytest.ini_options]
markers = [
    "e2e: end-to-end tests requiring LM Studio",
]
```

**Step 4: Run E2E test (requires LM Studio running)**

Run: `uv run pytest tests/e2e/ -v -s -m e2e`
Expected: PASS (takes 30-120s depending on model)

**Step 5: Commit**

```bash
git add tests/e2e/ tests/fixtures/article_generation/ pyproject.toml
git commit -m "feat(tests): add E2E test for full article generation pipeline"
```

---

### Task 10: Run full test suite and verify nothing is broken

**Step 1: Run all tests (excluding E2E by default)**

Run: `uv run pytest tests/ -v --ignore=tests/e2e`
Expected: ALL PASS (303+ tests)

**Step 2: Run E2E separately**

Run: `uv run pytest tests/e2e/ -v -s -m e2e`
Expected: PASS

**Step 3: Final commit if any fixups needed**

---

## Summary of deliverables

| # | What | Files |
|---|------|-------|
| 1 | `AgentSlotConfig` model | `src/config.py` |
| 2 | Updated config test helpers | `tests/test_article_generation_config.py` |
| 3 | Updated Config getters | `src/config.py` |
| 4 | `MockFactCheckAgent` + `MockEvidenceFindingAgent` | `src/agents/.../mock_agent.py` (x2) |
| 5 | Config-driven `AgentFactory` | `src/agents/article_generation/agent.py` |
| 6 | Updated config files | `config/config.yaml`, `config/config.yaml.template` |
| 7 | Updated scripts | `scripts/generate-articles.py` |
| 8 | Test config | `config/config.test.yaml` |
| 9 | E2E test + fixtures | `tests/e2e/`, `tests/fixtures/` |
| 10 | Full suite verification | — |
