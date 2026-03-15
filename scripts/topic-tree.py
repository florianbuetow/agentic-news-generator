#!/usr/bin/env python3
"""Build deterministic hierarchical topic trees for SRT transcripts.

This replaces the flat boundary detection + LLM topic extraction pipeline with:
- TreeSeg-style hierarchical segmentation (divisive SSE splitting)
- Deterministic taxonomy labeling (ACM CCS 2012) with cosine similarity scores
- Deterministic keyphrases via TF-IDF, YAKE, and KeyBERT with scores

Usage:
    uv run python scripts/topic-tree.py
    uv run python scripts/topic-tree.py --file path/to/transcript.srt
    uv run python scripts/topic-tree.py --force
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.config import (
    Config,
    TopicDetectionHierarchicalSegmentationConfig,
    TopicDetectionKeyphrasesConfig,
    TopicDetectionTaxonomyConfig,
)
from src.topic_detection.embedding.factory import EmbeddingGeneratorFactory
from src.topic_detection.hierarchy.data_types import TopicTreeNode
from src.topic_detection.hierarchy.labeling import (
    KeyBERTKeyphraseExtractor,
    LLMTopicLabeler,
    TaxonomyMatcher,
    TFIDFKeyphraseExtractor,
    YAKEKeyphraseExtractor,
)
from src.topic_detection.hierarchy.schemas import (
    HierarchicalSegmentationConfigData,
    KeyBERTKeyphrasesConfigData,
    KeyphraseData,
    KeyphrasesConfigData,
    MethodKeyphrasesData,
    TaxonomyConfigData,
    TaxonomyLabelData,
    TFIDFKeyphrasesConfigData,
    TopicTreeNodeData,
    TopicTreeOutput,
    YAKEKeyphrasesConfigData,
)
from src.topic_detection.hierarchy.treeseg_divisive_sse import TreeSegDivisiveSSETopicSegmenter
from src.topic_detection.taxonomy.acm_ccs2012 import ACMCCS2012Loader
from src.topic_detection.taxonomy.embedding_cache import (
    EmbeddingGenerator,
    build_cache,
    get_cache_path,
    load_cache,
    write_cache,
)
from src.util.srt_util import SRTEntry, SRTUtil


def format_timedelta(td: timedelta) -> str:
    """Format a timedelta as SRT timestamp string (HH:MM:SS,mmm)."""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def discover_srt_files(args: argparse.Namespace, config: Config) -> tuple[list[Path], Path] | tuple[None, str]:
    """Discover SRT files to process.

    Returns:
        (files, base_dir) on success, or (None, error_message) on failure.
    """
    if args.file:
        file_path = args.file
        if not file_path.exists():
            return None, f"Error: File not found: {file_path}"
        if file_path.suffix.lower() != ".srt":
            return None, f"Error: Expected .srt file, got: {file_path}"
        return [file_path], file_path.parent

    cleaned_dir = config.getDataDownloadsTranscriptsCleanedDir()
    if not cleaned_dir.exists():
        return None, f"Error: Cleaned transcripts directory not found: {cleaned_dir}"

    srt_files = sorted(cleaned_dir.rglob("*.srt"))
    srt_files = [f for f in srt_files if not f.name.startswith("._")]
    return srt_files, cleaned_dir


def build_context_blocks(*, utterances: list[str], context_window_entries: int) -> list[str]:
    """Build deterministic context blocks for per-entry embeddings."""
    if context_window_entries <= 0:
        raise ValueError("context_window_entries must be > 0")

    blocks: list[str] = []
    for i in range(len(utterances)):
        start = max(0, i - context_window_entries + 1)
        blocks.append(" ".join(utterances[start : i + 1]))
    return blocks


def traverse_preorder(root: TopicTreeNode) -> list[TopicTreeNode]:
    """Return nodes in deterministic pre-order traversal (root, left, right)."""
    nodes: list[TopicTreeNode] = []

    def _visit(node: TopicTreeNode) -> None:
        nodes.append(node)
        if node.left is not None and node.right is not None:
            _visit(node.left)
            _visit(node.right)

    _visit(root)
    return nodes


def node_text(*, entries: list[SRTEntry], node: TopicTreeNode) -> str:
    """Build text for a node by joining its SRT entry contents."""
    parts = [e.content for e in entries[node.start_entry_idx : node.end_entry_idx_exclusive]]
    return " ".join(parts)


def node_embedding(*, entry_embeddings: np.ndarray, node: TopicTreeNode) -> np.ndarray:
    """Compute mean embedding for a node's entry range."""
    start = node.start_entry_idx
    end = node.end_entry_idx_exclusive
    return np.mean(entry_embeddings[start:end], axis=0, dtype=np.float32)


def validate_taxonomy_name(taxonomy_name: str) -> None:
    """Validate supported taxonomy names."""
    if taxonomy_name != "acm_ccs_2012":
        raise ValueError(f"Unsupported taxonomy_name='{taxonomy_name}'. Only 'acm_ccs_2012' is implemented.")


def build_taxonomy_matcher(
    *,
    data_dir: Path,
    taxonomy_cfg: TopicDetectionTaxonomyConfig,
    embedding_model: str,
    embedding_generator: EmbeddingGenerator,
) -> TaxonomyMatcher | None:
    """Build a taxonomy matcher or return None if taxonomy is disabled."""
    if not taxonomy_cfg.enabled:
        return None

    validate_taxonomy_name(taxonomy_cfg.taxonomy_name)

    taxonomy_xml_path = data_dir / taxonomy_cfg.acm_ccs_2012_xml_path
    concepts = ACMCCS2012Loader().load(xml_path=taxonomy_xml_path)

    cache_path = get_cache_path(
        data_dir=data_dir,
        taxonomy_name=taxonomy_cfg.taxonomy_name,
        embedding_model=embedding_model,
        cache_dir=taxonomy_cfg.cache_dir,
    )

    cache = load_cache(cache_path=cache_path) if cache_path.exists() else None
    required_concept_ids = sorted([cid for cid, c in concepts.items() if c.level <= 4])
    cache_is_valid = (
        cache is not None
        and cache.taxonomy_name == taxonomy_cfg.taxonomy_name
        and cache.embedding_model == embedding_model
        and all(cid in cache.concept_embeddings for cid in required_concept_ids)
    )

    if not cache_is_valid:
        cache = build_cache(
            taxonomy_name=taxonomy_cfg.taxonomy_name,
            embedding_model=embedding_model,
            concepts=concepts,
            embedding_generator=embedding_generator,
            max_level=4,
            batch_size=128,
        )
        write_cache(cache_path=cache_path, cache=cache)

    concept_embeddings_np = {cid: np.array(vec, dtype=np.float32) for cid, vec in cache.concept_embeddings.items() if cid in concepts}

    return TaxonomyMatcher(
        taxonomy_config=taxonomy_cfg,
        concepts=concepts,
        concept_embeddings=concept_embeddings_np,
        max_level=4,
    )


def build_tree_output(
    *,
    srt_path: Path,
    data_dir: Path,
    seg_cfg: TopicDetectionHierarchicalSegmentationConfig,
    taxonomy_cfg: TopicDetectionTaxonomyConfig,
    keyphrases_cfg: TopicDetectionKeyphrasesConfig,
    embedding_generator: EmbeddingGenerator,
    embedding_model: str,
    matcher: TaxonomyMatcher | None,
    tfidf_extractor: TFIDFKeyphraseExtractor,
    yake_extractor: YAKEKeyphraseExtractor,
    keybert_extractor: KeyBERTKeyphraseExtractor,
    llm_labeler: LLMTopicLabeler | None,
) -> TopicTreeOutput:
    """Build a TopicTreeOutput for a single SRT file."""
    entries = SRTUtil.parse_srt_file(srt_path)
    if not entries:
        raise ValueError("No SRT entries found")

    utterances = [e.content for e in entries]
    blocks = build_context_blocks(utterances=utterances, context_window_entries=seg_cfg.context_window_entries)
    entry_embeddings = embedding_generator.generate(blocks)

    entry_start_seconds = [float(e.start.total_seconds()) for e in entries]
    entry_end_seconds = [float(e.end.total_seconds()) for e in entries]

    segmenter = TreeSegDivisiveSSETopicSegmenter(
        max_depth=seg_cfg.max_depth,
        min_leaf_entries=seg_cfg.min_leaf_entries,
        min_leaf_seconds=seg_cfg.min_leaf_seconds,
        min_gain=seg_cfg.min_gain,
    )
    root = segmenter.build_tree(
        embeddings=entry_embeddings,
        entry_start_seconds=entry_start_seconds,
        entry_end_seconds=entry_end_seconds,
    )

    nodes = traverse_preorder(root)
    leaf_texts = [node_text(entries=entries, node=n) for n in nodes if n.is_leaf()]
    vectorizer, tfidf_skipped_reason = tfidf_extractor.fit_vectorizer(leaf_texts=leaf_texts)

    node_data_list: list[TopicTreeNodeData] = []
    for n in nodes:
        start_idx = n.start_entry_idx
        end_idx_excl = n.end_entry_idx_exclusive
        if end_idx_excl <= start_idx:
            raise ValueError(f"Invalid node span: {start_idx}:{end_idx_excl}")

        start_ts = format_timedelta(entries[start_idx].start)
        end_ts = format_timedelta(entries[end_idx_excl - 1].end)
        duration_seconds = float(entry_end_seconds[end_idx_excl - 1] - entry_start_seconds[start_idx])

        child_ids = [n.left.node_id(), n.right.node_id()] if n.left is not None and n.right is not None else []

        labels: list[TaxonomyLabelData] = []
        if matcher is not None:
            target_level = min(n.depth + 1, 4)
            matches = matcher.match(
                node_embedding=node_embedding(entry_embeddings=entry_embeddings, node=n),
                target_level=target_level,
                top_k=taxonomy_cfg.top_k_per_node,
                min_similarity=taxonomy_cfg.min_similarity,
            )
            labels = [
                TaxonomyLabelData(
                    taxonomy=m.taxonomy,
                    concept_id=m.concept_id,
                    label=m.label,
                    path=m.path,
                    level=m.level,
                    score=m.score,
                )
                for m in matches
            ]

        text = node_text(entries=entries, node=n)

        tfidf_keyphrases = [
            KeyphraseData(phrase=k.phrase, score=k.score) for k in tfidf_extractor.extract(vectorizer=vectorizer, text=text)
        ]
        yake_keyphrases = [KeyphraseData(phrase=k.phrase, score=k.score) for k in yake_extractor.extract(text=text)]
        keybert_keyphrases = [KeyphraseData(phrase=k.phrase, score=k.score) for k in keybert_extractor.extract(text=text)]

        yake_skipped = None if keyphrases_cfg.yake.enabled else "YAKE keyphrase extraction disabled"
        keybert_skipped = None if keyphrases_cfg.keybert.enabled else "KeyBERT keyphrase extraction disabled"

        method_keyphrases = [
            MethodKeyphrasesData(method="tfidf", keyphrases=tfidf_keyphrases, skipped_reason=tfidf_skipped_reason),
            MethodKeyphrasesData(method="yake", keyphrases=yake_keyphrases, skipped_reason=yake_skipped),
            MethodKeyphrasesData(method="keybert", keyphrases=keybert_keyphrases, skipped_reason=keybert_skipped),
        ]

        llm_label = llm_labeler.label(text=text) if llm_labeler is not None else None

        node_data_list.append(
            TopicTreeNodeData(
                node_id=n.node_id(),
                parent_id=n.parent.node_id() if n.parent is not None else None,
                depth=n.depth,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                duration_seconds=duration_seconds,
                start_entry_idx=start_idx,
                end_entry_idx_exclusive=end_idx_excl,
                children_ids=child_ids,
                taxonomy_labels=labels,
                keyphrases=method_keyphrases,
                llm_label=llm_label,
            )
        )

    relative_source = srt_path.relative_to(data_dir)
    return TopicTreeOutput(
        source_file=str(relative_source),
        generated_at=datetime.now().isoformat(),
        embedding_model=embedding_model,
        hierarchical_segmentation=HierarchicalSegmentationConfigData(**seg_cfg.model_dump()),
        taxonomy=TaxonomyConfigData(**taxonomy_cfg.model_dump()),
        keyphrases=KeyphrasesConfigData(
            tfidf=TFIDFKeyphrasesConfigData(**keyphrases_cfg.tfidf.model_dump()),
            yake=YAKEKeyphrasesConfigData(**keyphrases_cfg.yake.model_dump()),
            keybert=KeyBERTKeyphrasesConfigData(**keyphrases_cfg.keybert.model_dump()),
        ),
        root_node_id=root.node_id(),
        total_nodes=len(node_data_list),
        nodes=node_data_list,
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build deterministic hierarchical topic trees for SRT transcripts")
    parser.add_argument(
        "--file",
        type=Path,
        help="Process a single SRT file instead of all files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if _topic_tree.json already exists",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config(config_path)

    td_config = config.get_topic_detection_config()
    data_dir = config.getDataDir()
    output_dir = config.getTopicDetectionOutputDir()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = discover_srt_files(args, config)
    if result[0] is None:
        print(result[1], file=sys.stderr)
        return 1
    srt_files, base_dir = result

    if not srt_files:
        print("No SRT files found to process.")
        return 0

    seg_cfg = td_config.hierarchical_segmentation
    taxonomy_cfg = td_config.taxonomy
    keyphrases_cfg = td_config.keyphrases

    if not seg_cfg.enabled:
        print("Error: topic_detection.hierarchical_segmentation.enabled is false; cannot build topic trees.", file=sys.stderr)
        return 1

    embedding_generator = EmbeddingGeneratorFactory.create(td_config.embedding)
    try:
        matcher = build_taxonomy_matcher(
            data_dir=data_dir,
            taxonomy_cfg=taxonomy_cfg,
            embedding_model=td_config.embedding.model_name,
            embedding_generator=embedding_generator,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    tfidf_extractor = TFIDFKeyphraseExtractor(config=keyphrases_cfg.tfidf)
    yake_extractor = YAKEKeyphraseExtractor(config=keyphrases_cfg.yake)
    keybert_extractor = KeyBERTKeyphraseExtractor(config=keyphrases_cfg.keybert, embedding_generator=embedding_generator)

    llm_label_cfg = td_config.llm_label
    llm_labeler = LLMTopicLabeler(config=llm_label_cfg) if llm_label_cfg.enabled else None

    print(f"Found {len(srt_files)} SRT file(s)")
    print(f"Output directory: {output_dir}")
    print()

    succeeded = 0
    skipped = 0
    failed = 0

    for srt_path in srt_files:
        relative_path = srt_path.relative_to(base_dir)
        output_subdir = output_dir / relative_path.parent
        output_filename = relative_path.stem + "_topic_tree.json"
        output_path = output_subdir / output_filename

        if output_path.exists() and not args.force:
            print(f"Skipping: {relative_path} (_topic_tree.json already exists)")
            skipped += 1
            continue

        print(f"Processing: {relative_path}")

        try:
            output = build_tree_output(
                srt_path=srt_path,
                data_dir=data_dir,
                seg_cfg=seg_cfg,
                taxonomy_cfg=taxonomy_cfg,
                keyphrases_cfg=keyphrases_cfg,
                embedding_generator=embedding_generator,
                embedding_model=td_config.embedding.model_name,
                matcher=matcher,
                tfidf_extractor=tfidf_extractor,
                yake_extractor=yake_extractor,
                keybert_extractor=keybert_extractor,
                llm_labeler=llm_labeler,
            )
            output_subdir.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output.model_dump(), f, indent=2, ensure_ascii=False)

            print(f"  → {output.total_nodes} nodes → {output_path}")
            succeeded += 1
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            failed += 1
        print()

    print("=" * 50)
    print(f"Completed: {succeeded} succeeded, {skipped} skipped, {failed} failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
