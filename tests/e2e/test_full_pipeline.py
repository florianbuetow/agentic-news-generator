"""End-to-end test for the full article generation pipeline.

Uses config/config.test.yaml and the real test bundle in data/test/articles/input/.

Run with: uv run pytest tests/e2e/ -v -s -m e2e
"""

from pathlib import Path

import pytest

from src.agents.article_generation.agent import build_chief_editor_orchestrator
from src.agents.article_generation.bundle_loader import bundle_to_source_metadata, load_bundle
from src.config import Config

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.test.yaml"
ARTICLES_INPUT_DIR = PROJECT_ROOT / "data" / "test" / "articles" / "input"

pytestmark = pytest.mark.e2e


class TestFullPipeline:
    """E2E test: orchestrator produces an article from a real test bundle."""

    @pytest.mark.timeout(600)
    def test_orchestrator_produces_article(self) -> None:
        """Full pipeline produces a valid ArticleGenerationResult."""
        config = Config(CONFIG_PATH)
        orchestrator = build_chief_editor_orchestrator(config=config)

        bundles = sorted(ARTICLES_INPUT_DIR.iterdir())
        assert len(bundles) > 0, f"No bundles found in {ARTICLES_INPUT_DIR}"
        bundle_dir = bundles[0]

        bundle = load_bundle(bundle_dir)
        source_metadata = bundle_to_source_metadata(bundle)

        result = orchestrator.generate_article(
            source_text=bundle.source_text,
            source_metadata=source_metadata,
            style_mode=config.get_article_generation_config().default_style_mode,
            reader_preference="",
        )

        assert result.success, f"Expected success, got error: {result.error}"
        assert result.article is not None
        assert result.article.headline != ""
        assert result.article.article_body != ""
        assert result.editor_report is not None
        assert result.editor_report.total_iterations >= 1
        assert result.artifacts_dir is not None

        artifacts = Path(result.artifacts_dir)
        assert artifacts.exists()

        print(f"\n{'=' * 60}")
        print(f"Article artifacts: {artifacts}")
        print("Files:")
        for f in sorted(artifacts.iterdir()):
            print(f"  {f.name}")
        print(f"{'=' * 60}")
