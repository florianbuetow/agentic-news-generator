"""Institutional memory store for specialist cache persistence and lookup."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from src.agents.article_generation.models import EvidenceRecord, FactCheckRecord

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class InstitutionalMemoryStore:
    """Filesystem-backed cache for fact-check and evidence-finding records."""

    def __init__(
        self,
        *,
        data_dir: Path,
        fact_checking_subdir: str,
        evidence_finding_subdir: str,
    ) -> None:
        self._data_dir = data_dir
        self._fact_checking_dir = data_dir / fact_checking_subdir
        self._evidence_finding_dir = data_dir / evidence_finding_subdir

    def build_fact_check_cache_key_hash(
        self,
        *,
        agent_name: str,
        normalized_query: str,
        model_name: str,
        kb_index_version: str,
    ) -> str:
        """Build cache hash for fact-check records."""
        cache_key = "|".join([agent_name, normalized_query, model_name, kb_index_version])
        return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]

    def build_evidence_cache_key_hash(
        self,
        *,
        agent_name: str,
        normalized_query: str,
        model_name: str,
    ) -> str:
        """Build cache hash for evidence records."""
        cache_key = "|".join([agent_name, normalized_query, model_name])
        return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]

    def lookup_fact_check(
        self,
        *,
        agent_name: str,
        normalized_query: str,
        model_name: str,
        kb_index_version: str,
    ) -> FactCheckRecord | None:
        """Lookup cached fact-check record by hash across date partitions."""
        cache_key_hash = self.build_fact_check_cache_key_hash(
            agent_name=agent_name,
            normalized_query=normalized_query,
            model_name=model_name,
            kb_index_version=kb_index_version,
        )
        return self._lookup_record(
            base_dir=self._fact_checking_dir,
            cache_key_hash=cache_key_hash,
            model_class=FactCheckRecord,
        )

    def lookup_evidence(
        self,
        *,
        agent_name: str,
        normalized_query: str,
        model_name: str,
    ) -> EvidenceRecord | None:
        """Lookup cached evidence record by hash across date partitions."""
        cache_key_hash = self.build_evidence_cache_key_hash(
            agent_name=agent_name,
            normalized_query=normalized_query,
            model_name=model_name,
        )
        return self._lookup_record(
            base_dir=self._evidence_finding_dir,
            cache_key_hash=cache_key_hash,
            model_class=EvidenceRecord,
        )

    def persist_fact_check(self, *, record: FactCheckRecord) -> Path:
        """Persist fact-check record under date partition."""
        return self._persist_record(base_dir=self._fact_checking_dir, cache_key_hash=record.cache_key_hash, payload=record.model_dump())

    def persist_evidence(self, *, record: EvidenceRecord) -> Path:
        """Persist evidence record under date partition."""
        return self._persist_record(base_dir=self._evidence_finding_dir, cache_key_hash=record.cache_key_hash, payload=record.model_dump())

    def _persist_record(self, *, base_dir: Path, cache_key_hash: str, payload: dict[str, object]) -> Path:
        """Persist a JSON payload in a date-partitioned directory."""
        date_dir = datetime.now(UTC).strftime("%Y-%m-%d")
        output_dir = base_dir / date_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{cache_key_hash}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        logger.info("Persisted record: %s", output_path)
        return output_path

    def _lookup_record(
        self,
        *,
        base_dir: Path,
        cache_key_hash: str,
        model_class: type[T],
    ) -> T | None:
        """Lookup a record by hash across date subdirectories."""
        if not base_dir.exists():
            logger.debug("Cache base dir does not exist: %s", base_dir)
            return None

        candidate_paths = sorted(base_dir.glob(f"*/{cache_key_hash}.json"), reverse=True)
        for candidate_path in candidate_paths:
            logger.info("Cache hit: %s", candidate_path)
            with open(candidate_path, encoding="utf-8") as handle:
                raw_data = json.load(handle)
            return model_class.model_validate(raw_data)

        logger.debug("Cache miss: hash=%s in %s", cache_key_hash, base_dir)
        return None
