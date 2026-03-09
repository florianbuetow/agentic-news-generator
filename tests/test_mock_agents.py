"""Tests for mock agent implementations."""

from src.agents.article_generation.article_review.mock_agent import MockArticleReviewAgent
from src.agents.article_generation.concern_mapping.mock_agent import MockConcernMappingAgent
from src.agents.article_generation.models import (
    AgentResult,
    ArticleResponse,
    Concern,
    Verdict,
    WriterFeedback,
)
from src.agents.article_generation.specialists.attribution.mock_agent import MockAttributionAgent
from src.agents.article_generation.specialists.evidence_finding.mock_agent import MockEvidenceFindingAgent
from src.agents.article_generation.specialists.fact_check.mock_agent import MockFactCheckAgent
from src.agents.article_generation.specialists.opinion.mock_agent import MockOpinionAgent
from src.agents.article_generation.specialists.style_review.mock_agent import MockStyleReviewAgent
from src.agents.article_generation.writer.mock_agent import MockWriterAgent


def _sample_article() -> ArticleResponse:
    return ArticleResponse(
        headline="Test headline",
        alternative_headline="Alt headline",
        article_body="Test body content.",
        description="Test description.",
    )


def _sample_concern(concern_id: int = 1) -> Concern:
    return Concern(
        concern_id=concern_id,
        excerpt="some excerpt",
        review_note="some review note",
    )


class TestMockFactCheckAgent:
    """Tests for MockFactCheckAgent."""

    def test_returns_keep_verdict(self) -> None:
        """Mock agent returns a static KEEP verdict."""
        agent = MockFactCheckAgent()
        concern = Concern(concern_id=1, excerpt="some claim", review_note="review note")
        article = _sample_article()
        agent_result = agent.evaluate(
            concern=concern,
            article=article,
            source_text="source",
            source_metadata={"source_file": "f", "topic_slug": "s"},
            style_requirements="SCIAM_MAGAZINE",
        )
        assert isinstance(agent_result, AgentResult)
        assert agent_result.prompt == "[mock]"
        verdict = agent_result.output
        assert isinstance(verdict, Verdict)
        assert verdict.concern_id == 1
        assert verdict.misleading is False
        assert verdict.status == "KEEP"

    def test_concern_id_matches_input(self) -> None:
        """Verdict concern_id must match the input concern."""
        agent = MockFactCheckAgent()
        concern = Concern(concern_id=42, excerpt="claim", review_note="note")
        article = _sample_article()
        agent_result = agent.evaluate(
            concern=concern, article=article, source_text="s",
            source_metadata={"source_file": "f", "topic_slug": "s"},
            style_requirements="SCIAM_MAGAZINE",
        )
        assert agent_result.output.concern_id == 42


class TestMockEvidenceFindingAgent:
    """Tests for MockEvidenceFindingAgent."""

    def test_returns_keep_verdict(self) -> None:
        """Mock agent returns a static KEEP verdict."""
        agent = MockEvidenceFindingAgent()
        concern = Concern(concern_id=3, excerpt="some claim", review_note="review note")
        article = _sample_article()
        agent_result = agent.evaluate(
            concern=concern,
            article=article,
            source_text="source",
            source_metadata={"source_file": "f", "topic_slug": "s"},
            style_requirements="SCIAM_MAGAZINE",
        )
        assert isinstance(agent_result, AgentResult)
        assert agent_result.prompt == "[mock]"
        verdict = agent_result.output
        assert isinstance(verdict, Verdict)
        assert verdict.concern_id == 3
        assert verdict.misleading is False
        assert verdict.status == "KEEP"


class TestMockWriterAgent:
    """Tests for MockWriterAgent."""

    def test_generate_returns_article_response(self) -> None:
        agent = MockWriterAgent()
        agent_result = agent.generate(
            source_text="source",
            source_metadata={"key": "value"},
            style_mode="SCIAM_MAGAZINE",
            reader_preference="",
        )
        assert isinstance(agent_result, AgentResult)
        assert agent_result.prompt == "[mock]"
        result = agent_result.output
        assert isinstance(result, ArticleResponse)
        assert result.headline != ""
        assert result.article_body != ""

    def test_revise_returns_article_response(self) -> None:
        agent = MockWriterAgent()
        feedback = WriterFeedback(
            iteration=1,
            rating=5,
            passed=False,
            reasoning="needs work",
            improvement_suggestions=[],
            todo_list=[],
            verdicts=[],
        )
        agent_result = agent.revise(context="some context", feedback=feedback)
        assert isinstance(agent_result, AgentResult)
        assert agent_result.prompt == "[mock]"
        result = agent_result.output
        assert isinstance(result, ArticleResponse)
        assert result.headline != ""


class TestMockArticleReviewAgent:
    """Tests for MockArticleReviewAgent."""

    def test_review_returns_empty_bullets(self) -> None:
        agent = MockArticleReviewAgent()
        agent_result = agent.review(
            article=_sample_article(),
            source_text="source",
            source_metadata={"key": "value"},
        )
        assert isinstance(agent_result, AgentResult)
        assert agent_result.prompt == "[mock]"
        assert agent_result.output.markdown_bullets == ""


class TestMockConcernMappingAgent:
    """Tests for MockConcernMappingAgent."""

    def test_map_concerns_returns_empty_mappings(self) -> None:
        agent = MockConcernMappingAgent()
        agent_result = agent.map_concerns(
            style_requirements="SCIAM_MAGAZINE",
            source_text="source",
            generated_article_json="{}",
            concerns=[_sample_concern()],
        )
        assert isinstance(agent_result, AgentResult)
        assert agent_result.prompt == "[mock]"
        assert agent_result.output.mappings == []


class TestMockOpinionAgent:
    """Tests for MockOpinionAgent."""

    def test_evaluate_returns_keep_verdict(self) -> None:
        agent = MockOpinionAgent()
        concern = _sample_concern(concern_id=7)
        agent_result = agent.evaluate(
            concern=concern,
            article=_sample_article(),
            source_text="source",
            source_metadata={},
            style_requirements="SCIAM_MAGAZINE",
        )
        assert isinstance(agent_result, AgentResult)
        assert agent_result.prompt == "[mock]"
        assert agent_result.output.concern_id == 7
        assert agent_result.output.status == "KEEP"
        assert agent_result.output.misleading is False


class TestMockAttributionAgent:
    """Tests for MockAttributionAgent."""

    def test_evaluate_returns_keep_verdict(self) -> None:
        agent = MockAttributionAgent()
        concern = _sample_concern(concern_id=3)
        agent_result = agent.evaluate(
            concern=concern,
            article=_sample_article(),
            source_text="source",
            source_metadata={},
            style_requirements="SCIAM_MAGAZINE",
        )
        assert isinstance(agent_result, AgentResult)
        assert agent_result.prompt == "[mock]"
        assert agent_result.output.concern_id == 3
        assert agent_result.output.status == "KEEP"
        assert agent_result.output.misleading is False


class TestMockStyleReviewAgent:
    """Tests for MockStyleReviewAgent."""

    def test_evaluate_returns_keep_verdict(self) -> None:
        agent = MockStyleReviewAgent()
        concern = _sample_concern(concern_id=5)
        agent_result = agent.evaluate(
            concern=concern,
            article=_sample_article(),
            source_text="source",
            source_metadata={},
            style_requirements="SCIAM_MAGAZINE",
        )
        assert isinstance(agent_result, AgentResult)
        assert agent_result.prompt == "[mock]"
        assert agent_result.output.concern_id == 5
        assert agent_result.output.status == "KEEP"
        assert agent_result.output.misleading is False
