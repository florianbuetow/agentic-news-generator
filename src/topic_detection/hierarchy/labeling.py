"""Labeling utilities for hierarchical transcript topic trees."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import litellm
import numpy as np
import yake
from keybert._model import KeyBERT
from litellm.exceptions import BadRequestError
from numpy.typing import NDArray
from pydantic import ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TopicDetectionKeyBERTKeyphrasesConfig,
    TopicDetectionLLMLabelConfig,
    TopicDetectionTaxonomyConfig,
    TopicDetectionTFIDFKeyphrasesConfig,
    TopicDetectionYAKEKeyphrasesConfig,
)
from src.topic_detection.hierarchy.schemas import LLMTopicLabelData
from src.topic_detection.taxonomy.data_types import TaxonomyConcept
from src.topic_detection.taxonomy.embedding_cache import EmbeddingGenerator


@dataclass(frozen=True)
class TaxonomyMatch:
    """A single taxonomy match for a node embedding."""

    taxonomy: str
    concept_id: str
    label: str
    path: list[str]
    level: int
    score: float


@dataclass(frozen=True)
class Keyphrase:
    """A single keyphrase and its score."""

    phrase: str
    score: float


class TaxonomyMatcher:
    """Match node embeddings to taxonomy concepts by cosine similarity."""

    def __init__(
        self,
        *,
        taxonomy_config: TopicDetectionTaxonomyConfig,
        concepts: dict[str, TaxonomyConcept],
        concept_embeddings: dict[str, NDArray[np.float32]],
        max_level: int,
    ) -> None:
        self._cfg = taxonomy_config
        self._concepts = concepts
        self._max_level = max_level

        if max_level <= 0:
            raise ValueError("max_level must be > 0")

        self._concept_ids_by_level: dict[int, list[str]] = {}
        self._concept_matrix_by_level: dict[int, NDArray[np.float32]] = {}

        for level in range(1, max_level + 1):
            ids = sorted([cid for cid, c in concepts.items() if c.level == level])
            if not ids:
                continue

            missing = [cid for cid in ids if cid not in concept_embeddings]
            if missing:
                raise ValueError(f"Missing embeddings for {len(missing)} concepts at level {level}")

            mat = np.vstack([concept_embeddings[cid] for cid in ids]).astype(np.float32, copy=False)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat_normed = mat / (norms + 1e-10)

            self._concept_ids_by_level[level] = ids
            self._concept_matrix_by_level[level] = mat_normed

    def match(
        self,
        *,
        node_embedding: NDArray[np.float32],
        target_level: int,
        top_k: int,
        min_similarity: float,
    ) -> list[TaxonomyMatch]:
        """Return top-k taxonomy matches for a node embedding."""
        if not self._cfg.enabled:
            return []

        if target_level < 1 or target_level > self._max_level:
            raise ValueError(f"target_level must be in [1, {self._max_level}], got: {target_level}")

        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        if min_similarity < 0.0 or min_similarity > 1.0:
            raise ValueError("min_similarity must be in [0.0, 1.0]")

        if target_level not in self._concept_ids_by_level or target_level not in self._concept_matrix_by_level:
            return []

        concept_ids = self._concept_ids_by_level[target_level]
        mat = self._concept_matrix_by_level[target_level]
        if not concept_ids:
            return []

        node_vec = node_embedding.astype(np.float32, copy=False)
        node_norm = node_vec / (float(np.linalg.norm(node_vec)) + 1e-10)

        scores = mat @ node_norm
        order = np.argsort(-scores, kind="stable")

        matches: list[TaxonomyMatch] = []
        for idx in order:
            score = float(scores[int(idx)])
            if score < min_similarity:
                break

            cid = concept_ids[int(idx)]
            concept = self._concepts[cid]
            matches.append(
                TaxonomyMatch(
                    taxonomy=self._cfg.taxonomy_name,
                    concept_id=cid,
                    label=concept.pref_label,
                    path=list(concept.path_labels),
                    level=concept.level,
                    score=score,
                )
            )
            if len(matches) >= top_k:
                break

        return matches


class TFIDFKeyphraseExtractor:
    """Extract deterministic TF-IDF keyphrases from node texts."""

    def __init__(self, *, config: TopicDetectionTFIDFKeyphrasesConfig) -> None:
        self._cfg = config

    def fit_vectorizer(self, *, leaf_texts: list[str]) -> tuple[TfidfVectorizer | None, str | None]:
        """Fit a TF-IDF vectorizer on leaf texts.

        Returns:
            (vectorizer, skipped_reason)
        """
        if not self._cfg.enabled:
            return None, "TF-IDF keyphrase extraction disabled"

        leaf_count = len(leaf_texts)
        if leaf_count == 0:
            return None, "No leaf texts available"

        if self._cfg.min_df > leaf_count:
            return None, f"min_df={self._cfg.min_df} exceeds leaf document count={leaf_count}"

        vectorizer = TfidfVectorizer(
            input="content",
            encoding="utf-8",
            decode_error="strict",
            strip_accents=None,
            lowercase=self._cfg.lowercase,
            preprocessor=None,
            tokenizer=None,
            analyzer="word",
            stop_words=self._cfg.stop_words,
            token_pattern=r"(?u)\b\w\w+\b",  # nosec B106
            ngram_range=(self._cfg.ngram_range_min, self._cfg.ngram_range_max),
            max_df=self._cfg.max_df,
            min_df=self._cfg.min_df,
            max_features=self._cfg.max_features,
            vocabulary=None,
            binary=False,
            dtype=np.float64,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
        )

        try:
            vectorizer.fit(leaf_texts)
        except ValueError as e:
            return None, f"TF-IDF fit failed: {e}"

        return vectorizer, None

    def extract(self, *, vectorizer: TfidfVectorizer | None, text: str) -> list[Keyphrase]:
        """Extract top-k TF-IDF keyphrases for a single text."""
        if not self._cfg.enabled:
            return []

        if vectorizer is None:
            return []

        matrix = vectorizer.transform([text])
        row = matrix[0]

        if row.nnz == 0:
            return []

        features = vectorizer.get_feature_names_out()
        pairs: list[tuple[str, float]] = [(features[int(i)], float(s)) for i, s in zip(row.indices, row.data, strict=True)]
        pairs_sorted = sorted(pairs, key=lambda p: (-p[1], p[0]))

        top = pairs_sorted[: self._cfg.top_k_per_node]
        return [Keyphrase(phrase=phrase, score=score) for phrase, score in top]


class YAKEKeyphraseExtractor:
    """Extract keyphrases using YAKE (Yet Another Keyword Extractor)."""

    def __init__(self, *, config: TopicDetectionYAKEKeyphrasesConfig) -> None:
        self._cfg = config

    def extract(self, *, text: str) -> list[Keyphrase]:
        """Extract top-k YAKE keyphrases for a single text."""
        if not self._cfg.enabled:
            return []

        if not text.strip():
            return []

        extractor = yake.KeywordExtractor(
            lan="en",
            n=self._cfg.max_ngram_size,
            dedupLim=self._cfg.deduplication_threshold,
            dedupFunc=self._cfg.deduplication_algo,
            windowsSize=self._cfg.window_size,
            top=self._cfg.top_k_per_node,
        )

        keywords: list[tuple[str, float]] = extractor.extract_keywords(text)

        return [Keyphrase(phrase=kw, score=1.0 / (1.0 + yake_score)) for kw, yake_score in keywords]


class _EmbeddingGeneratorBackend:
    """KeyBERT backend adapter wrapping EmbeddingGenerator."""

    def __init__(self, *, embedding_generator: EmbeddingGenerator) -> None:
        self._generator = embedding_generator

    def embed(self, documents: list[str], verbose: bool) -> NDArray[np.float32]:
        """Embed documents (verbose is required by KeyBERT interface)."""
        _ = verbose
        return self._generator.generate(documents)


class KeyBERTKeyphraseExtractor:
    """Extract keyphrases using KeyBERT with a custom embedding backend."""

    def __init__(
        self,
        *,
        config: TopicDetectionKeyBERTKeyphrasesConfig,
        embedding_generator: EmbeddingGenerator,
    ) -> None:
        self._cfg = config
        self._model = KeyBERT(model=_EmbeddingGeneratorBackend(embedding_generator=embedding_generator))

    def extract(self, *, text: str) -> list[Keyphrase]:
        """Extract top-k KeyBERT keyphrases for a single text."""
        if not self._cfg.enabled:
            return []

        if not text.strip():
            return []

        keywords: list[tuple[str, float]] = self._model.extract_keywords(
            text,
            keyphrase_ngram_range=(self._cfg.keyphrase_ngram_range_min, self._cfg.keyphrase_ngram_range_max),
            stop_words=self._cfg.stop_words,
            top_n=self._cfg.top_k_per_node,
            use_mmr=self._cfg.use_mmr,
            diversity=self._cfg.mmr_diversity,
        )

        return [Keyphrase(phrase=kw, score=score) for kw, score in keywords]


_LLM_TOPIC_LABEL_SYSTEM_PROMPT = (
    "You are a topic labeler. Given a text segment from a video transcript, "
    "produce a JSON object with exactly these fields:\n"
    "\n"
    '- "summary": A 1-2 sentence summary of the segment.\n'
    '- "about": A single sentence answering "What is this text about?"\n'
    '- "topic_labels": A list of 2-5 canonical single-word topic labels '
    "(lowercase, no phrases, no multi-word terms).\n"
    "\n"
    "Respond with ONLY the JSON object. No markdown, no explanation."
)

_LLM_TOPIC_LABEL_USER_PROMPT = """Label the following transcript segment:

{text}"""


class LLMTopicLabeler:
    """Label topic tree nodes using an LLM to generate summaries and topic labels."""

    def __init__(self, *, config: TopicDetectionLLMLabelConfig) -> None:
        self._cfg = config
        self._llm = config.llm

    def label(self, *, text: str) -> LLMTopicLabelData | None:
        """Generate an LLM topic label for a text segment."""
        if not self._cfg.enabled:
            return None

        if not text.strip():
            return None

        last_error: Exception | None = None
        for attempt in range(self._llm.max_retries):
            try:
                return self._call_llm(text)
            except ValueError as e:
                last_error = e
                if attempt < self._llm.max_retries - 1:
                    print(f"      LLM label retry {attempt + 1}/{self._llm.max_retries - 1} after error: {e}")
                    time.sleep(self._llm.retry_delay)

        raise ValueError(f"LLM topic labeling failed after {self._llm.max_retries} attempts. Last error: {last_error}")

    def _call_llm(self, text: str) -> LLMTopicLabelData:
        """Make a single LLM call and parse the response."""
        messages = [
            {"role": "system", "content": _LLM_TOPIC_LABEL_SYSTEM_PROMPT},
            {"role": "user", "content": _LLM_TOPIC_LABEL_USER_PROMPT.format(text=text)},
        ]

        try:
            response = litellm.completion(
                model=self._llm.model,
                messages=messages,
                api_base=self._llm.api_base,
                api_key=self._llm.api_key,
                max_tokens=self._llm.max_tokens,
                temperature=self._llm.temperature,
            )
        except BadRequestError as e:
            error_msg = str(e)
            if "No models loaded" in error_msg:
                raise BadRequestError(
                    message=f"No models loaded in LM Studio. Expected model: {self._llm.model}. "
                    f"Load it with: lms load {self._llm.model.split('/')[-1]}",
                    model=self._llm.model,
                    llm_provider="openai",
                ) from e
            raise

        response_text = response.choices[0].message.content
        if response_text is None or response_text.strip() == "":
            raise ValueError("LLM returned empty response")

        return self.parse_response(response_text)

    def parse_response(self, response_text: str) -> LLMTopicLabelData:
        """Parse LLM response into LLMTopicLabelData."""
        cleaned = response_text.strip()

        # Strip <think> tags from reasoning models
        think_match = re.search(r"</think>\s*(.*)$", cleaned, re.DOTALL)
        if think_match:
            cleaned = think_match.group(1).strip()

        # Strip markdown code fences
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        if not cleaned:
            raise ValueError(f"No JSON content after cleaning response. Raw (first 500 chars): {response_text[:500]}")

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}. Response: {response_text[:500]}") from e

        try:
            return LLMTopicLabelData.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"LLM response does not match expected schema: {e}. Data: {data}") from e
