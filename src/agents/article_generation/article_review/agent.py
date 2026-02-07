"""Article-review agent for unsupported additions extraction."""

from __future__ import annotations

import json
import logging

from src.agents.article_generation.base import BaseAgent, LLMClient
from src.agents.article_generation.models import ArticleResponse, ArticleReviewRaw
from src.agents.article_generation.prompts.loader import PromptLoader
from src.config import Config, LLMConfig

logger = logging.getLogger(__name__)


class ArticleReviewAgent(BaseAgent):
    """Extracts unsupported additions from a generated article draft."""

    def __init__(
        self,
        *,
        llm_config: LLMConfig,
        config: Config,
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
        prompt_file: str,
    ) -> None:
        super().__init__(llm_config=llm_config, config=config, llm_client=llm_client)
        self._prompt_loader = prompt_loader
        self._prompt_file = prompt_file

    def review(
        self,
        *,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
    ) -> ArticleReviewRaw:
        """Run review and return raw markdown bullets."""
        logger.info("Reviewing article: headline=%r source_chars=%d", article.headline, len(source_text))
        prompt_template = self._prompt_loader.load_prompt(prompt_file=self._prompt_file)
        user_prompt = prompt_template.format(
            source_text=source_text,
            source_metadata=json.dumps(source_metadata, ensure_ascii=False),
            generated_article=json.dumps(article.model_dump(), ensure_ascii=False),
        )
        logger.info("Review prompt assembled: %d chars", len(user_prompt))
        response = self._call_llm(messages=[{"role": "user", "content": user_prompt}])
        review = ArticleReviewRaw(markdown_bullets=response.strip())
        bullet_count = response.count("\n- ") + response.count("\n* ") + (1 if response.lstrip().startswith(("- ", "* ")) else 0)
        logger.info("Review completed: ~%d bullets, response_chars=%d", bullet_count, len(response))
        return review
