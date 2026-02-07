"""Concern-mapping agent for specialist delegation planning."""

from __future__ import annotations

import json
import logging
from typing import cast

from src.agents.article_generation.base import BaseAgent, LLMClient
from src.agents.article_generation.models import Concern, ConcernMapping, ConcernMappingResult
from src.agents.article_generation.prompts.loader import PromptLoader
from src.config import Config, LLMConfig

logger = logging.getLogger(__name__)


class ConcernMappingAgent(BaseAgent):
    """Maps each concern to exactly one specialist agent."""

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

    def map_concerns(
        self,
        *,
        style_requirements: str,
        source_text: str,
        generated_article_json: str,
        concerns: list[Concern],
    ) -> ConcernMappingResult:
        """Map concerns to specialists and validate strict output schema."""
        logger.info("Mapping %d concerns to specialists", len(concerns))
        prompt_template = self._prompt_loader.load_prompt(prompt_file=self._prompt_file)
        user_prompt = prompt_template.format(
            style_requirements=style_requirements,
            source_text=source_text,
            generated_article=generated_article_json,
            concerns=json.dumps([concern.model_dump() for concern in concerns], ensure_ascii=False),
        )
        logger.info("Concern mapping prompt assembled: %d chars", len(user_prompt))
        response = self._call_llm(messages=[{"role": "user", "content": user_prompt}])

        text = response.strip()
        if text.startswith("["):
            raw_items = json.loads(text)
            if not isinstance(raw_items, list):
                raise ValueError("Concern mapping array response must be a JSON list")

            typed_items: list[dict[str, object]] = []
            typed_raw_items = cast(list[object], raw_items)
            for item in typed_raw_items:
                if not isinstance(item, dict):
                    raise ValueError("Concern mapping array items must be JSON objects")
                typed_item_obj = cast(dict[object, object], item)
                typed_item: dict[str, object] = {}
                for key, value in typed_item_obj.items():
                    if not isinstance(key, str):
                        raise ValueError("Concern mapping object keys must be strings")
                    typed_item[key] = value
                typed_items.append(typed_item)

            mappings = [ConcernMapping.model_validate(item) for item in typed_items]
            logger.info("Concern mapping completed: %d mappings (array format)", len(mappings))
            return ConcernMappingResult(mappings=mappings)

        result = self._parse_json_response(response=text, model_class=ConcernMappingResult)
        logger.info("Concern mapping completed: %d mappings (object format)", len(result.mappings))
        return result
