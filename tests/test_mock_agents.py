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
