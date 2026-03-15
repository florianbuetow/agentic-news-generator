"""Tests for topic detection configuration models."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import (
    Config,
    TopicDetectionConfig,
    TopicDetectionEmbeddingConfig,
    TopicDetectionHierarchicalSegmentationConfig,
    TopicDetectionKeyBERTKeyphrasesConfig,
    TopicDetectionKeyphrasesConfig,
    TopicDetectionSlidingWindowConfig,
    TopicDetectionTaxonomyConfig,
    TopicDetectionTFIDFKeyphrasesConfig,
    TopicDetectionYAKEKeyphrasesConfig,
)


def get_valid_paths_config() -> dict[str, str]:
    """Return a valid paths configuration dictionary for tests."""
    return {
        "data_dir": "data",
        "data_models_dir": "data/models",
        "data_downloads_dir": "data/downloads",
        "data_downloads_videos_dir": "data/downloads/videos",
        "data_downloads_transcripts_dir": "data/downloads/transcripts",
        "data_downloads_transcripts_hallucinations_dir": "data/downloads/transcripts-hallucinations",
        "data_downloads_transcripts_cleaned_dir": "data/downloads/transcripts_cleaned",
        "data_transcripts_topics_dir": "data/downloads/transcripts-topics",
        "data_downloads_audio_dir": "data/downloads/audio",
        "data_downloads_metadata_dir": "data/downloads/metadata",
        "data_output_dir": "data/output",
        "data_input_dir": "data/input",
        "data_temp_dir": "data/temp",
        "data_archive_dir": "data/archive",
        "data_archive_videos_dir": "data/archive/videos",
        "data_logs_dir": "logs",
        "data_output_articles_dir": "data/output/articles",
        "data_articles_input_dir": "data/articles/input",
        "reports_dir": "reports",
        "data_article_generation_output_dir": "data/output/articles",
        "data_article_generation_artifacts_dir": "data/output/article_editor_runs",
        "data_article_generation_kb_dir": "data/knowledgebase",
        "data_article_generation_kb_index_dir": "data/knowledgebase_index",
        "data_article_generation_institutional_memory_dir": "data/institutional_memory",
        "data_article_generation_prompts_dir": "prompts/article_editor",
        "data_topic_detection_output_dir": "data/output/topics",
        "data_topic_detection_taxonomies_dir": "data/input/taxonomies",
        "data_topic_detection_taxonomy_cache_dir": "data/input/taxonomies/cache",
    }


def get_valid_channels_config() -> list[dict[str, str | int]]:
    """Return a valid channels configuration list for tests."""
    return [
        {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test",
            "description": "Test channel description",
            "download-limiter": 10,
            "language": "en",
        }
    ]


def get_valid_topic_detection_config() -> dict[str, object]:
    """Return a valid topic detection configuration dictionary for tests."""
    return {
        "embedding": {
            "provider": "lmstudio",
            "model_name": "multilingual-e5-base-mlx",
            "api_base": "http://127.0.0.1:1234/v1",
            "api_key": "LMSTUDIO_API_KEY",
        },
        "sliding_window": {
            "window_size": 50,
            "stride": 25,
            "threshold_method": "relative",
            "threshold_value": 0.4,
            "smoothing_passes": 1,
        },
        "topic_detection_llm": {
            "model": "openai/test-model",
            "api_base": "http://127.0.0.1:1234/v1",
            "api_key": "LMSTUDIO_API_KEY",
            "context_window": 262144,
            "max_tokens": 4096,
            "temperature": 0.3,
            "context_window_threshold": 90,
            "max_retries": 3,
            "retry_delay": 2.0,
            "timeout_seconds": 30,
        },
        "hierarchical_segmentation": {
            "enabled": True,
            "method": "treeseg_divisive_sse",
            "context_window_entries": 8,
            "max_depth": 4,
            "min_leaf_entries": 12,
            "min_leaf_seconds": 45.0,
            "min_gain": 0.02,
        },
        "taxonomy": {
            "enabled": True,
            "taxonomy_name": "acm_ccs_2012",
            "acm_ccs_2012_xml_file": "acm_ccs2012.xml",
            "top_k_per_node": 3,
            "min_similarity": 0.3,
        },
        "keyphrases": {
            "tfidf": {
                "enabled": True,
                "top_k_per_node": 12,
                "ngram_range_min": 1,
                "ngram_range_max": 3,
                "min_df": 2,
                "max_df": 0.9,
                "stop_words": "english",
                "max_features": 20000,
                "lowercase": True,
            },
            "yake": {
                "enabled": True,
                "top_k_per_node": 12,
                "max_ngram_size": 3,
                "deduplication_threshold": 0.9,
                "deduplication_algo": "seqm",
                "window_size": 1,
            },
            "keybert": {
                "enabled": True,
                "top_k_per_node": 12,
                "keyphrase_ngram_range_min": 1,
                "keyphrase_ngram_range_max": 3,
                "use_mmr": True,
                "mmr_diversity": 0.5,
                "min_score": 0.0,
                "stop_words": "english",
            },
        },
        "llm_label": {
            "enabled": True,
            "llm": {
                "model": "openai/gemma-3-1b-it",
                "api_base": "http://127.0.0.1:1234/v1",
                "api_key": "lm-studio",
                "context_window": 8192,
                "max_tokens": 512,
                "temperature": 0.3,
                "context_window_threshold": 90,
                "max_retries": 3,
                "retry_delay": 2.0,
                "timeout_seconds": 30,
            },
        },
    }


class TestTopicDetectionEmbeddingConfig:
    """Tests for TopicDetectionEmbeddingConfig model."""

    def test_valid_embedding_config(self) -> None:
        """Test valid embedding configuration."""
        config = TopicDetectionEmbeddingConfig(
            provider="lmstudio",
            model_name="multilingual-e5-base-mlx",
            api_base="http://127.0.0.1:1234/v1",
            api_key="LMSTUDIO_API_KEY",
        )
        assert config.provider == "lmstudio"
        assert config.model_name == "multilingual-e5-base-mlx"
        assert config.api_base == "http://127.0.0.1:1234/v1"
        assert config.api_key == "LMSTUDIO_API_KEY"

    def test_embedding_config_with_none_api_key(self) -> None:
        """Test embedding config with None api_key."""
        config = TopicDetectionEmbeddingConfig(
            provider="lmstudio",
            model_name="test-model",
            api_base="http://localhost:1234/v1",
            api_key=None,
        )
        assert config.api_key is None

    def test_missing_provider_field(self) -> None:
        """Test embedding config with missing provider field."""
        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionEmbeddingConfig.model_validate(
                {
                    "model_name": "test-model",
                    "api_base": "http://localhost:1234/v1",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("provider",) and err["type"] == "missing" for err in errors)

    def test_missing_model_name_field(self) -> None:
        """Test embedding config with missing model_name field."""
        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionEmbeddingConfig.model_validate(
                {
                    "provider": "lmstudio",
                    "api_base": "http://localhost:1234/v1",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("model_name",) and err["type"] == "missing" for err in errors)


class TestTopicDetectionSlidingWindowConfig:
    """Tests for TopicDetectionSlidingWindowConfig model."""

    def test_valid_sliding_window_config(self) -> None:
        """Test valid sliding window configuration."""
        config = TopicDetectionSlidingWindowConfig(
            window_size=50,
            stride=25,
            threshold_method="relative",
            threshold_value=0.4,
            smoothing_passes=1,
        )
        assert config.window_size == 50
        assert config.stride == 25
        assert config.threshold_method == "relative"
        assert config.threshold_value == 0.4
        assert config.smoothing_passes == 1

    def test_window_size_must_be_positive(self) -> None:
        """Test that window_size must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionSlidingWindowConfig(
                window_size=0,
                stride=25,
                threshold_method="relative",
                threshold_value=0.4,
                smoothing_passes=1,
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("window_size",) for err in errors)

    def test_stride_must_be_positive(self) -> None:
        """Test that stride must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionSlidingWindowConfig(
                window_size=50,
                stride=0,
                threshold_method="relative",
                threshold_value=0.4,
                smoothing_passes=1,
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("stride",) for err in errors)


class TestTopicDetectionHierarchicalSegmentationConfig:
    """Tests for TopicDetectionHierarchicalSegmentationConfig model."""

    def test_valid_hierarchical_segmentation_config(self) -> None:
        """Test valid hierarchical segmentation configuration."""
        config = TopicDetectionHierarchicalSegmentationConfig(
            enabled=True,
            method="treeseg_divisive_sse",
            context_window_entries=8,
            max_depth=4,
            min_leaf_entries=12,
            min_leaf_seconds=45.0,
            min_gain=0.02,
        )
        assert config.enabled is True
        assert config.method == "treeseg_divisive_sse"
        assert config.context_window_entries == 8
        assert config.max_depth == 4

    def test_invalid_method_raises(self) -> None:
        """Test invalid method raises ValidationError."""
        with pytest.raises(ValidationError):
            TopicDetectionHierarchicalSegmentationConfig(
                enabled=True,
                method="unknown",
                context_window_entries=8,
                max_depth=4,
                min_leaf_entries=12,
                min_leaf_seconds=45.0,
                min_gain=0.02,
            )


class TestTopicDetectionTaxonomyConfig:
    """Tests for TopicDetectionTaxonomyConfig model."""

    def test_valid_taxonomy_config(self) -> None:
        """Test valid taxonomy configuration."""
        config = TopicDetectionTaxonomyConfig(
            enabled=True,
            taxonomy_name="acm_ccs_2012",
            acm_ccs_2012_xml_file="acm_ccs2012.xml",
            top_k_per_node=3,
            min_similarity=0.3,
        )
        assert config.enabled is True
        assert config.top_k_per_node == 3


class TestTopicDetectionKeyphrasesConfig:
    """Tests for TopicDetectionKeyphrasesConfig container and sub-models."""

    def test_valid_tfidf_config(self) -> None:
        """Test valid TF-IDF keyphrases configuration."""
        config = TopicDetectionTFIDFKeyphrasesConfig(
            enabled=True,
            top_k_per_node=12,
            ngram_range_min=1,
            ngram_range_max=3,
            min_df=2,
            max_df=0.9,
            stop_words="english",
            max_features=20000,
            lowercase=True,
        )
        assert config.enabled is True
        assert config.top_k_per_node == 12

    def test_valid_yake_config(self) -> None:
        """Test valid YAKE keyphrases configuration."""
        config = TopicDetectionYAKEKeyphrasesConfig(
            enabled=True,
            top_k_per_node=12,
            max_ngram_size=3,
            deduplication_threshold=0.9,
            deduplication_algo="seqm",
            window_size=1,
        )
        assert config.enabled is True
        assert config.top_k_per_node == 12
        assert config.max_ngram_size == 3

    def test_valid_keybert_config(self) -> None:
        """Test valid KeyBERT keyphrases configuration."""
        config = TopicDetectionKeyBERTKeyphrasesConfig(
            enabled=True,
            top_k_per_node=12,
            keyphrase_ngram_range_min=1,
            keyphrase_ngram_range_max=3,
            use_mmr=True,
            mmr_diversity=0.5,
            min_score=0.0,
            stop_words="english",
        )
        assert config.enabled is True
        assert config.use_mmr is True
        assert config.min_score == 0.0

    def test_valid_container_config(self) -> None:
        """Test valid container keyphrases configuration."""
        config_dict = get_valid_topic_detection_config()["keyphrases"]
        config = TopicDetectionKeyphrasesConfig.model_validate(config_dict)
        assert config.tfidf.enabled is True
        assert config.yake.enabled is True
        assert config.keybert.enabled is True


class TestTopicDetectionConfig:
    """Tests for TopicDetectionConfig model."""

    def test_valid_topic_detection_config(self) -> None:
        """Test valid complete topic detection configuration."""
        config_dict = get_valid_topic_detection_config()
        config = TopicDetectionConfig.model_validate(config_dict)

        assert config.embedding.provider == "lmstudio"
        assert config.sliding_window.window_size == 50
        assert config.topic_detection_llm.model == "openai/test-model"
        assert config.hierarchical_segmentation.method == "treeseg_divisive_sse"
        assert config.taxonomy.taxonomy_name == "acm_ccs_2012"
        assert config.keyphrases.tfidf.top_k_per_node == 12

    def test_missing_embedding_section(self) -> None:
        """Test config with missing embedding section."""
        config_dict = get_valid_topic_detection_config()
        del config_dict["embedding"]

        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionConfig.model_validate(config_dict)
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("embedding",) and err["type"] == "missing" for err in errors)

    def test_missing_sliding_window_section(self) -> None:
        """Test config with missing sliding_window section."""
        config_dict = get_valid_topic_detection_config()
        del config_dict["sliding_window"]

        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionConfig.model_validate(config_dict)
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("sliding_window",) and err["type"] == "missing" for err in errors)

    def test_missing_hierarchical_segmentation_section(self) -> None:
        """Test config with missing hierarchical_segmentation section."""
        config_dict = get_valid_topic_detection_config()
        del config_dict["hierarchical_segmentation"]

        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionConfig.model_validate(config_dict)
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("hierarchical_segmentation",) and err["type"] == "missing" for err in errors)

    def test_missing_taxonomy_section(self) -> None:
        """Test config with missing taxonomy section."""
        config_dict = get_valid_topic_detection_config()
        del config_dict["taxonomy"]

        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionConfig.model_validate(config_dict)
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("taxonomy",) and err["type"] == "missing" for err in errors)

    def test_missing_keyphrases_section(self) -> None:
        """Test config with missing keyphrases section."""
        config_dict = get_valid_topic_detection_config()
        del config_dict["keyphrases"]

        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionConfig.model_validate(config_dict)
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("keyphrases",) and err["type"] == "missing" for err in errors)


class TestConfigTopicDetectionIntegration:
    """Integration tests for Config.get_topic_detection_config."""

    def test_config_get_topic_detection_config(self, tmp_path: Path) -> None:
        """Test loading topic detection config from Config class."""
        config_dict = {
            "channels": get_valid_channels_config(),
            "paths": get_valid_paths_config(),
            "topic_detection": get_valid_topic_detection_config(),
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = Config(config_file)
        td_config = config.get_topic_detection_config()

        assert td_config.embedding.provider == "lmstudio"
        assert td_config.sliding_window.window_size == 50
        assert td_config.topic_detection_llm.model == "openai/test-model"
        assert td_config.hierarchical_segmentation.method == "treeseg_divisive_sse"

    def test_config_get_topic_detection_config_when_missing(self, tmp_path: Path) -> None:
        """Test that get_topic_detection_config raises KeyError when section is missing."""
        config_dict = {
            "channels": get_valid_channels_config(),
            "paths": get_valid_paths_config(),
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = Config(config_file)

        with pytest.raises(KeyError, match="topic_detection"):
            config.get_topic_detection_config()

    def test_config_validates_topic_detection_on_load(self, tmp_path: Path) -> None:
        """Test that invalid topic_detection config raises error on Config load."""
        config_dict = {
            "channels": get_valid_channels_config(),
            "paths": get_valid_paths_config(),
            "topic_detection": {
                "embedding": {
                    # Missing required fields
                    "provider": "lmstudio",
                },
                "sliding_window": get_valid_topic_detection_config()["sliding_window"],
                "topic_detection_llm": get_valid_topic_detection_config()["topic_detection_llm"],
                "hierarchical_segmentation": get_valid_topic_detection_config()["hierarchical_segmentation"],
                "taxonomy": get_valid_topic_detection_config()["taxonomy"],
                "keyphrases": get_valid_topic_detection_config()["keyphrases"],
                "llm_label": get_valid_topic_detection_config()["llm_label"],
            },
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        with pytest.raises(ValueError, match="Topic detection configuration validation failed"):
            Config(config_file)
