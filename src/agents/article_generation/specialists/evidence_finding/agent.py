"""Evidence-finding specialist agent."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import cast

from src.agents.article_generation.base import BaseSpecialistAgent, LLMClient, PerplexityClient
from src.agents.article_generation.chief_editor.institutional_memory import InstitutionalMemoryStore
from src.agents.article_generation.models import ArticleResponse, Concern, EvidenceRecord, Verdict
from src.agents.article_generation.prompts.loader import PromptLoader
from src.config import Config, LLMConfig

logger = logging.getLogger(__name__)


class EvidenceFindingAgent(BaseSpecialistAgent):
    """Searches for supporting/refuting evidence with cache-aware calls."""

    def __init__(
        self,
        *,
        llm_config: LLMConfig,
        config: Config,
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
        specialists_dir: str,
        prompt_file: str,
        perplexity_client: PerplexityClient,
        perplexity_model: str,
        institutional_memory: InstitutionalMemoryStore,
    ) -> None:
        super().__init__(llm_config=llm_config, config=config, llm_client=llm_client)
        self._prompt_loader = prompt_loader
        self._specialists_dir = specialists_dir
        self._prompt_file = prompt_file
        self._perplexity_client = perplexity_client
        self._perplexity_model = perplexity_model
        self._institutional_memory = institutional_memory

    def evaluate(
        self,
        *,
        concern: Concern,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_requirements: str,
    ) -> Verdict:
        """Evaluate evidence support with cached external search."""
        logger.info("Evaluating concern #%d: excerpt=%r", concern.concern_id, concern.excerpt[:80])
        normalized_query = self._normalize_query(concern.review_note)
        cached_record = self._institutional_memory.lookup_evidence(
            agent_name="evidence_finding",
            normalized_query=normalized_query,
            model_name=self._llm_config.model,
        )
        if cached_record is not None:
            logger.info("Evidence cache hit for concern #%d", concern.concern_id)
            return cached_record.verdict

        logger.info("Evidence cache miss for concern #%d, querying Perplexity model=%s", concern.concern_id, self._perplexity_model)
        search_response = self._perplexity_client.search(
            query=concern.review_note,
            model=self._perplexity_model,
            timeout_seconds=self._llm_config.timeout_seconds,
        )
        citations = self._extract_citations(search_response=search_response)
        search_payload = json.dumps(search_response, ensure_ascii=False)
        logger.info("Perplexity returned %d citations, payload=%d chars", len(citations), len(search_payload))

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
            web_evidence=search_payload,
        )
        logger.info("Evidence finding prompt assembled: %d chars", len(user_prompt))
        response = self._call_llm(messages=[{"role": "user", "content": user_prompt}])
        verdict = self._parse_json_response(response=response, model_class=Verdict)
        logger.info("Evidence verdict: concern_id=%d status=%s misleading=%s", verdict.concern_id, verdict.status, verdict.misleading)

        article_id = self._build_article_id(source_metadata)
        cache_key_hash = self._institutional_memory.build_evidence_cache_key_hash(
            agent_name="evidence_finding",
            normalized_query=normalized_query,
            model_name=self._llm_config.model,
        )
        record = EvidenceRecord(
            timestamp=datetime.now(UTC).isoformat(),
            article_id=article_id,
            concern_id=concern.concern_id,
            prompt=user_prompt,
            query=concern.review_note,
            normalized_query=normalized_query,
            model_name=self._llm_config.model,
            cache_key_hash=cache_key_hash,
            perplexity_response=search_payload,
            citations=citations,
            verdict=verdict,
        )
        self._institutional_memory.persist_evidence(record=record)
        logger.info("Evidence record persisted: cache_key=%s", cache_key_hash)

        return verdict

    def _normalize_query(self, query: str) -> str:
        """Normalize query for stable cache keys."""
        return re.sub(r"\s+", " ", query.strip().lower())

    def _extract_citations(self, *, search_response: dict[str, object]) -> list[str]:
        """Extract citations from a Perplexity-compatible response payload."""
        citations_value = search_response.get("citations")
        if isinstance(citations_value, list):
            typed_citations = cast(list[object], citations_value)
            return [item for item in typed_citations if isinstance(item, str)]
        return []

    def _build_article_id(self, source_metadata: dict[str, str | None]) -> str:
        """Build deterministic article identifier."""
        source_file = source_metadata.get("source_file")
        topic_slug = source_metadata.get("topic_slug")
        if source_file is None:
            raise ValueError("Missing required metadata field: source_file")
        if topic_slug is None:
            raise ValueError("Missing required metadata field: topic_slug")
        return f"{source_file}:{topic_slug}"
