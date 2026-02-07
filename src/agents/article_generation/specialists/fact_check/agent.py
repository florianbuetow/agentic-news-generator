"""Fact-check specialist agent."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime

from src.agents.article_generation.base import BaseSpecialistAgent, KnowledgeBaseRetriever, LLMClient
from src.agents.article_generation.chief_editor.institutional_memory import InstitutionalMemoryStore
from src.agents.article_generation.models import ArticleResponse, Concern, FactCheckRecord, Verdict
from src.agents.article_generation.prompts.loader import PromptLoader
from src.config import Config, LLMConfig

logger = logging.getLogger(__name__)


class FactCheckAgent(BaseSpecialistAgent):
    """Validates concerns against the configured knowledge base."""

    def __init__(
        self,
        *,
        llm_config: LLMConfig,
        config: Config,
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
        specialists_dir: str,
        prompt_file: str,
        knowledge_base_retriever: KnowledgeBaseRetriever,
        institutional_memory: InstitutionalMemoryStore,
        kb_index_version: str,
        kb_timeout_seconds: int,
    ) -> None:
        super().__init__(llm_config=llm_config, config=config, llm_client=llm_client)
        self._prompt_loader = prompt_loader
        self._specialists_dir = specialists_dir
        self._prompt_file = prompt_file
        self._knowledge_base_retriever = knowledge_base_retriever
        self._institutional_memory = institutional_memory
        self._kb_index_version = kb_index_version
        self._kb_timeout_seconds = kb_timeout_seconds

    def evaluate(
        self,
        *,
        concern: Concern,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_requirements: str,
    ) -> Verdict:
        """Evaluate factual accuracy with cache-aware KB retrieval."""
        logger.info("Evaluating concern #%d: excerpt=%r", concern.concern_id, concern.excerpt[:80])
        normalized_query = self._normalize_query(concern.review_note)
        cached_record = self._institutional_memory.lookup_fact_check(
            agent_name="fact_check",
            normalized_query=normalized_query,
            model_name=self._llm_config.model,
            kb_index_version=self._kb_index_version,
        )
        if cached_record is not None:
            logger.info("Fact-check cache hit for concern #%d", concern.concern_id)
            return cached_record.verdict

        logger.info("Fact-check cache miss for concern #%d, querying KB", concern.concern_id)
        kb_results = self._knowledge_base_retriever.search(
            query=concern.review_note,
            top_k=5,
            timeout_seconds=self._kb_timeout_seconds,
        )
        kb_evidence = json.dumps(kb_results, ensure_ascii=False)
        logger.info("KB returned %d results (%d chars)", len(kb_results), len(kb_evidence))

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
            kb_evidence=kb_evidence,
        )
        logger.info("Fact-check prompt assembled: %d chars", len(user_prompt))
        response = self._call_llm(messages=[{"role": "user", "content": user_prompt}])
        verdict = self._parse_json_response(response=response, model_class=Verdict)
        logger.info("Fact-check verdict: concern_id=%d status=%s misleading=%s", verdict.concern_id, verdict.status, verdict.misleading)

        article_id = self._build_article_id(source_metadata)
        cache_key_hash = self._institutional_memory.build_fact_check_cache_key_hash(
            agent_name="fact_check",
            normalized_query=normalized_query,
            model_name=self._llm_config.model,
            kb_index_version=self._kb_index_version,
        )
        record = FactCheckRecord(
            timestamp=datetime.now(UTC).isoformat(),
            article_id=article_id,
            concern_id=concern.concern_id,
            prompt=user_prompt,
            query=concern.review_note,
            normalized_query=normalized_query,
            model_name=self._llm_config.model,
            kb_index_version=self._kb_index_version,
            cache_key_hash=cache_key_hash,
            kb_response=kb_evidence,
            verdict=verdict,
        )
        self._institutional_memory.persist_fact_check(record=record)
        logger.info("Fact-check record persisted: cache_key=%s", cache_key_hash)

        return verdict

    def _normalize_query(self, query: str) -> str:
        """Normalize query for stable cache keys."""
        return re.sub(r"\s+", " ", query.strip().lower())

    def _build_article_id(self, source_metadata: dict[str, str | None]) -> str:
        """Build deterministic article identifier."""
        source_file = source_metadata.get("source_file")
        topic_slug = source_metadata.get("topic_slug")
        if source_file is None:
            raise ValueError("Missing required metadata field: source_file")
        if topic_slug is None:
            raise ValueError("Missing required metadata field: topic_slug")
        return f"{source_file}:{topic_slug}"
