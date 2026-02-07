"""Tests for institutional memory storage and retrieval."""

from datetime import UTC, datetime
from pathlib import Path

from src.agents.article_generation.chief_editor.institutional_memory import InstitutionalMemoryStore
from src.agents.article_generation.models import EvidenceRecord, FactCheckRecord, Verdict


def _build_verdict() -> Verdict:
    return Verdict(
        concern_id=1,
        misleading=False,
        status="KEEP",
        rationale="ok",
        suggested_fix=None,
        evidence="evidence",
        citations=["https://example.com"],
    )


def test_institutional_memory_persist_and_lookup_fact_check(tmp_path: Path) -> None:
    """Fact-check records persist and can be looked up by cache key."""
    store = InstitutionalMemoryStore(
        data_dir=tmp_path,
        fact_checking_subdir="fact_checking",
        evidence_finding_subdir="evidence_finding",
    )

    cache_key_hash = store.build_fact_check_cache_key_hash(
        agent_name="fact_check",
        normalized_query="query",
        model_name="model",
        kb_index_version="v1",
    )

    record = FactCheckRecord(
        timestamp=datetime.now(UTC).isoformat(),
        article_id="article",
        concern_id=1,
        prompt="prompt",
        query="query",
        normalized_query="query",
        model_name="model",
        kb_index_version="v1",
        cache_key_hash=cache_key_hash,
        kb_response="[]",
        verdict=_build_verdict(),
    )

    store.persist_fact_check(record=record)
    loaded = store.lookup_fact_check(
        agent_name="fact_check",
        normalized_query="query",
        model_name="model",
        kb_index_version="v1",
    )

    assert loaded is not None
    assert loaded.cache_key_hash == cache_key_hash
    assert loaded.verdict.status == "KEEP"


def test_institutional_memory_persist_and_lookup_evidence(tmp_path: Path) -> None:
    """Evidence records persist and can be looked up by cache key."""
    store = InstitutionalMemoryStore(
        data_dir=tmp_path,
        fact_checking_subdir="fact_checking",
        evidence_finding_subdir="evidence_finding",
    )

    cache_key_hash = store.build_evidence_cache_key_hash(
        agent_name="evidence_finding",
        normalized_query="query",
        model_name="model",
    )

    record = EvidenceRecord(
        timestamp=datetime.now(UTC).isoformat(),
        article_id="article",
        concern_id=1,
        prompt="prompt",
        query="query",
        normalized_query="query",
        model_name="model",
        cache_key_hash=cache_key_hash,
        perplexity_response="{}",
        citations=["https://example.com"],
        verdict=_build_verdict(),
    )

    store.persist_evidence(record=record)
    loaded = store.lookup_evidence(
        agent_name="evidence_finding",
        normalized_query="query",
        model_name="model",
    )

    assert loaded is not None
    assert loaded.cache_key_hash == cache_key_hash
    assert loaded.citations == ["https://example.com"]


def test_institutional_memory_lookup_miss_returns_none(tmp_path: Path) -> None:
    """Lookup for absent key returns none."""
    store = InstitutionalMemoryStore(
        data_dir=tmp_path,
        fact_checking_subdir="fact_checking",
        evidence_finding_subdir="evidence_finding",
    )

    assert (
        store.lookup_fact_check(
            agent_name="fact_check",
            normalized_query="missing",
            model_name="model",
            kb_index_version="v1",
        )
        is None
    )
