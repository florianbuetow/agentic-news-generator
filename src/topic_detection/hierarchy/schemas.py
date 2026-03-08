"""Pydantic schemas for deterministic hierarchical topic tree outputs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HierarchicalSegmentationConfigData(BaseModel):
    """Serialized hierarchical segmentation configuration."""

    enabled: bool
    method: str
    context_window_entries: int = Field(..., gt=0)
    max_depth: int = Field(..., ge=0)
    min_leaf_entries: int = Field(..., gt=0)
    min_leaf_seconds: float = Field(..., ge=0)
    min_gain: float = Field(..., ge=0)

    model_config = ConfigDict(frozen=True, extra="forbid")


class TaxonomyConfigData(BaseModel):
    """Serialized taxonomy mapping configuration."""

    enabled: bool
    taxonomy_name: str
    acm_ccs_2012_xml_path: str
    cache_dir: str
    top_k_per_node: int = Field(..., gt=0)
    min_similarity: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True, extra="forbid")


class TFIDFKeyphrasesConfigData(BaseModel):
    """Serialized TF-IDF keyphrase extraction configuration."""

    enabled: bool
    top_k_per_node: int = Field(..., gt=0)
    ngram_range_min: int = Field(..., ge=1)
    ngram_range_max: int = Field(..., ge=1)
    min_df: int = Field(..., ge=1)
    max_df: float = Field(..., gt=0.0, le=1.0)
    stop_words: str | None
    max_features: int = Field(..., gt=0)
    lowercase: bool

    model_config = ConfigDict(frozen=True, extra="forbid")


class YAKEKeyphrasesConfigData(BaseModel):
    """Serialized YAKE keyphrase extraction configuration."""

    enabled: bool
    top_k_per_node: int = Field(..., gt=0)
    max_ngram_size: int = Field(..., ge=1)
    deduplication_threshold: float = Field(..., ge=0.0, le=1.0)
    deduplication_algo: str
    window_size: int = Field(..., ge=1)

    model_config = ConfigDict(frozen=True, extra="forbid")


class KeyBERTKeyphrasesConfigData(BaseModel):
    """Serialized KeyBERT keyphrase extraction configuration."""

    enabled: bool
    top_k_per_node: int = Field(..., gt=0)
    keyphrase_ngram_range_min: int = Field(..., ge=1)
    keyphrase_ngram_range_max: int = Field(..., ge=1)
    use_mmr: bool
    mmr_diversity: float = Field(..., ge=0.0, le=1.0)
    stop_words: str | None

    model_config = ConfigDict(frozen=True, extra="forbid")


class KeyphrasesConfigData(BaseModel):
    """Serialized keyphrase extraction configuration (all methods)."""

    tfidf: TFIDFKeyphrasesConfigData
    yake: YAKEKeyphrasesConfigData
    keybert: KeyBERTKeyphrasesConfigData

    model_config = ConfigDict(frozen=True, extra="forbid")


class TaxonomyLabelData(BaseModel):
    """A single taxonomy label match for a node."""

    taxonomy: str
    concept_id: str
    label: str
    path: list[str]
    level: int = Field(..., ge=1)
    score: float = Field(..., ge=-1.0, le=1.0)

    model_config = ConfigDict(frozen=True, extra="forbid")


class KeyphraseData(BaseModel):
    """A single keyphrase and its score."""

    phrase: str
    score: float = Field(..., ge=0)

    model_config = ConfigDict(frozen=True, extra="forbid")


class MethodKeyphrasesData(BaseModel):
    """Keyphrases from a single extraction method."""

    method: str
    keyphrases: list[KeyphraseData]
    skipped_reason: str | None

    model_config = ConfigDict(frozen=True, extra="forbid")


class LLMTopicLabelData(BaseModel):
    """LLM-generated topic label for a node."""

    summary: str
    about: str
    topic_labels: list[str]

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicTreeNodeData(BaseModel):
    """A node in the exported topic tree."""

    node_id: str
    parent_id: str | None
    depth: int = Field(..., ge=0)
    start_timestamp: str
    end_timestamp: str
    duration_seconds: float = Field(..., ge=0)
    start_entry_idx: int = Field(..., ge=0)
    end_entry_idx_exclusive: int = Field(..., ge=0)
    children_ids: list[str]
    taxonomy_labels: list[TaxonomyLabelData]
    keyphrases: list[MethodKeyphrasesData]
    llm_label: LLMTopicLabelData | None

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicTreeOutput(BaseModel):
    """Complete deterministic topic tree output for a single transcript."""

    source_file: str
    generated_at: str
    embedding_model: str
    hierarchical_segmentation: HierarchicalSegmentationConfigData
    taxonomy: TaxonomyConfigData
    keyphrases: KeyphrasesConfigData
    root_node_id: str
    total_nodes: int = Field(..., ge=1)
    nodes: list[TopicTreeNodeData]

    model_config = ConfigDict(frozen=True, extra="forbid")
