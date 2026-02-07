"""Writer agent for initial draft generation and revisions."""

from __future__ import annotations

import json
import logging

from src.agents.article_generation.base import BaseAgent, LLMClient
from src.agents.article_generation.models import ArticleResponse, WriterFeedback
from src.agents.article_generation.prompts.loader import PromptLoader
from src.config import Config, LLMConfig

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Generates and revises article drafts."""

    def __init__(
        self,
        *,
        llm_config: LLMConfig,
        config: Config,
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
        writer_prompt_file: str,
        revision_prompt_file: str,
    ) -> None:
        super().__init__(llm_config=llm_config, config=config, llm_client=llm_client)
        self._prompt_loader = prompt_loader
        self._writer_prompt_file = writer_prompt_file
        self._revision_prompt_file = revision_prompt_file

    def generate(
        self,
        *,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_mode: str,
        reader_preference: str,
    ) -> ArticleResponse:
        """Generate an initial article draft."""
        logger.info(
            "Generating initial draft: style=%s source_chars=%d reader_preference=%s",
            style_mode,
            len(source_text),
            reader_preference if reader_preference else "(none)",
        )
        metadata_payload = json.dumps(source_metadata, ensure_ascii=False)

        prompt_template = self._prompt_loader.load_prompt(prompt_file=self._writer_prompt_file)
        user_prompt = prompt_template.format(
            style_mode=style_mode,
            reader_preference=reader_preference,
            source_text=source_text,
            source_metadata=metadata_payload,
        )
        logger.info("Writer prompt assembled: %d chars", len(user_prompt))

        response = self._call_llm(messages=[{"role": "user", "content": user_prompt}])
        article = self._parse_json_response(response=response, model_class=ArticleResponse)
        logger.info("Draft generated: headline=%r body_chars=%d", article.headline, len(article.article_body))
        return article

    def revise(
        self,
        *,
        context: str,
        feedback: WriterFeedback,
    ) -> ArticleResponse:
        """Revise an article draft using deterministic feedback fields."""
        logger.info(
            "Revising draft: iteration=%d rating=%d todos=%d suggestions=%d",
            feedback.iteration,
            feedback.rating,
            len(feedback.todo_list),
            len(feedback.improvement_suggestions),
        )
        revision_template = self._prompt_loader.load_prompt(prompt_file=self._revision_prompt_file)
        user_prompt = revision_template.format(
            rating=feedback.rating,
            pass_status=str(feedback.passed),
            reasoning=feedback.reasoning,
            improvement_suggestions="\n".join(feedback.improvement_suggestions),
            context=context,
        )
        logger.info("Revision prompt assembled: %d chars", len(user_prompt))

        response = self._call_llm(messages=[{"role": "user", "content": user_prompt}])
        article = self._parse_json_response(response=response, model_class=ArticleResponse)
        logger.info("Revised draft: headline=%r body_chars=%d", article.headline, len(article.article_body))
        return article
