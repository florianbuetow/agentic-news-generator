"""Mock writer agent for testing."""

from __future__ import annotations

import logging

from src.agents.article_generation.models import ArticleResponse, WriterFeedback

logger = logging.getLogger(__name__)


class MockWriterAgent:
    """Returns static article drafts without LLM calls."""

    def generate(
        self,
        *,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_mode: str,
        reader_preference: str,
    ) -> ArticleResponse:
        """Return a static article draft."""
        logger.info("MockWriterAgent: returning static draft")
        return ArticleResponse(
            headline="Mock Article: Advances in LLM Efficiency",
            alternative_headline="How Researchers Are Making Language Models Faster and Smaller",
            article_body=(
                "Recent advances in large language model efficiency have opened new possibilities "
                "for deploying AI systems at scale. Researchers have developed several key techniques "
                "including quantization, mixture of experts architectures, knowledge distillation, "
                "and speculative decoding that collectively reduce computational requirements while "
                "maintaining model quality.\n\n"
                "Quantization reduces the precision of model weights from 32-bit floating point to "
                "8-bit or even 4-bit integers, dramatically reducing memory footprint and inference "
                "latency. Mixture of experts models activate only a subset of parameters for each "
                "input, achieving the capacity of much larger models at a fraction of the cost.\n\n"
                "These innovations suggest a future where powerful language models can run on consumer "
                "hardware, democratizing access to advanced AI capabilities."
            ),
            description="A summary of recent breakthroughs in making large language models more efficient.",
        )

    def revise(
        self,
        *,
        context: str,
        feedback: WriterFeedback,
    ) -> ArticleResponse:
        """Return the same static draft (mock never changes)."""
        logger.info("MockWriterAgent: returning static revision for iteration %d", feedback.iteration)
        return self.generate(
            source_text="",
            source_metadata={},
            style_mode="",
            reader_preference="",
        )
