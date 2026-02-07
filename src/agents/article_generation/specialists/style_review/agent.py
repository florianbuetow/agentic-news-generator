"""Style-review specialist agent."""

from __future__ import annotations

import json
import logging

from src.agents.article_generation.base import BaseSpecialistAgent, LLMClient
from src.agents.article_generation.models import ArticleResponse, Concern, Verdict
from src.agents.article_generation.prompts.loader import PromptLoader
from src.config import Config, LLMConfig

logger = logging.getLogger(__name__)


class StyleReviewAgent(BaseSpecialistAgent):
    """Evaluates style adherence and misleading tone risks."""

    def __init__(
        self,
        *,
        llm_config: LLMConfig,
        config: Config,
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
        specialists_dir: str,
        prompt_file: str,
    ) -> None:
        super().__init__(llm_config=llm_config, config=config, llm_client=llm_client)
        self._prompt_loader = prompt_loader
        self._specialists_dir = specialists_dir
        self._prompt_file = prompt_file

    def evaluate(
        self,
        *,
        concern: Concern,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_requirements: str,
    ) -> Verdict:
        """Evaluate concern using style-review prompt."""
        logger.info("Evaluating concern #%d: excerpt=%r", concern.concern_id, concern.excerpt[:80])
        prompt_template = self._prompt_loader.load_specialist_prompt(
            specialists_dir=self._specialists_dir,
            prompt_file=self._prompt_file,
        )
        user_prompt = prompt_template.format(
            style_requirements=style_requirements,
            concern=concern.review_note,
            article_excerpt=concern.excerpt,
            source_text=source_text,
            source_metadata=json.dumps(source_metadata, ensure_ascii=False),
        )
        logger.info("Style review prompt assembled: %d chars", len(user_prompt))
        response = self._call_llm(messages=[{"role": "user", "content": user_prompt}])
        verdict = self._parse_json_response(response=response, model_class=Verdict)
        logger.info("Style review verdict: concern_id=%d status=%s misleading=%s", verdict.concern_id, verdict.status, verdict.misleading)
        return verdict
