# Topic Detection Pipeline

This document explains how this repo segments transcripts into topic sections.

## Overview

### Recommended: deterministic hierarchical topic trees

The recommended pipeline builds a **deterministic, per-transcript topic tree** (coarse → fine) and avoids non-deterministic LLM topic strings:

- **Atomic unit:** SRT entries (timestamps stay stable)
- **Hierarchical segmentation:** TreeSeg-style divisive splitting on an embedding sequence (binary tree)
- **Deterministic labeling:** taxonomy multi-labels (**ACM CCS 2012**) + TF‑IDF keyphrases (both with scores)

Outputs per transcript:
- `*_topic_tree.json` under `data/output/topics/...`
- (Optional) mini-rag export of **leaf nodes** as `.txt + .json` pairs

### Legacy: flat boundary detection + LLM topic strings

The older pipeline segments continuous text into flat sections and then uses an LLM to generate topic strings. It is kept for comparison and may be removed.

## Prerequisites (ACM CCS 2012 taxonomy)

If taxonomy labeling is enabled in `config/config.yaml`, you must download the ACM CCS 2012 SKOS XML and place it at:

- `data/input/taxonomies/acm_ccs2012.xml` (gitignored)

Official source:
- <https://dl.acm.org/pb-assets/dl_ccs/acm_ccs2012-1626988337597.xml>

The pipeline will also create an embedding cache (gitignored) at:
- `data/input/taxonomies/cache/acm_ccs_2012.<embedding_model>.json`

## Running the Pipeline

### Hierarchical topic trees (recommended)

| Target | Script | Behavior |
|--------|--------|----------|
| `just topics-tree` | `scripts/topic-tree.py` | Builds `*_topic_tree.json` deterministically. Skips existing files unless `--force`. |
| `just topics-all` | Runs `topics-tree` | Current “default” topic detection pipeline. |

### Export to mini-rag (leaf nodes)

This exports **leaf nodes** from `*_topic_tree.json` files into `.txt + .json` pairs:

```bash
just export-to-minirag -- --export-dir /path/to/mini-rag/data/input/knowledgebase/txt
```

### Legacy pipeline (flat segmentation + LLM topics)

| Target | Script | Behavior |
|--------|--------|----------|
| `just topics-embed` | `scripts/generate-embeddings.py` | Creates `_embeddings.json` files. Skips existing files (incremental). |
| `just topics-boundaries` | `scripts/detect-boundaries.py` | Creates `_segmentation.json` files. Overwrites existing files (fast operation). |
| `just topics-extract` | `scripts/extract-topics.py` | Extracts topics from segments using an LLM. |

## Legacy Pipeline Architecture (flat segmentation)

> Note: The remainder of this document section describes the legacy flat segmentation pipeline. The recommended pipeline is `just topics-tree`.

```
┌─────────────┐    ┌───────────────┐    ┌──────────────────────┐
│  Raw Text   │───▶│   Tokenizer   │───▶│  List of Words       │
└─────────────┘    └───────────────┘    └──────────────────────┘
                                                   │
                                                   ▼
                   ┌───────────────┐    ┌──────────────────────┐
                   │   Embedding   │◀───│   Chunk Encoder      │
                   │   Generator   │    │   (sliding window)   │
                   └───────────────┘    └──────────────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────┐
                                        │ Similarity Calculator│
                                        │   (cosine + smooth)  │
                                        └──────────────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────┐
                                        │  Boundary Detector   │
                                        │  (depth scoring)     │
                                        └──────────────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────┐
                                        │  Segment Assembler   │
                                        └──────────────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────┐
                                        │   Final Segments     │
                                        └──────────────────────┘
```

## Stage 1: Tokenization

The text is split into word-level tokens (not BPE/subword tokens). This provides:
- Consistent position tracking for timestamp mapping
- Intuitive segment boundaries at word boundaries
- Predictable window sizes

## Stage 2: Chunk Encoding (Sliding Window)

A sliding window moves across the tokenized text, creating overlapping chunks:

```
Text:  w0  w1  w2  w3  w4  w5  w6  w7  w8  w9  w10 w11 w12 ...

       |──── window_size ────|
       ├─────────────────────┤
Chunk 0: [w0, w1, w2, w3, w4]
           |──── stride ────|
           ├─────────────────────┤
Chunk 1:   [w2, w3, w4, w5, w6]
               ├─────────────────────┤
Chunk 2:       [w4, w5, w6, w7, w8]
                   ...
```

**Parameters:**
- `window_size`: Number of words per chunk (determines context captured)
- `stride`: Words to advance between chunks (determines granularity)

Each chunk is converted to text and passed to the embedding model, producing a dense vector representation of that text window's semantic content.

## Stage 3: Similarity Calculation

For each pair of adjacent chunks, compute **cosine similarity**:

```
similarity[i] = cos(embedding[i], embedding[i+1])
              = (embedding[i] · embedding[i+1]) / (||embedding[i]|| × ||embedding[i+1]||)
```

The result is a similarity curve showing how semantically similar adjacent windows are:

```
similarity
    ^
1.0 │    ╱╲        ╱╲
    │   ╱  ╲      ╱  ╲
0.7 │  ╱    ╲    ╱    ╲  ╱╲
    │ ╱      ╲  ╱      ╲╱  ╲
0.4 │╱        ╲╱            ╲
    └──────────────────────────▶ position
              ▲             ▲
           valley        valley
        (low similarity = potential boundary)
```

**Optional Smoothing:** A 3-point moving average can be applied to reduce noise in the similarity curve. Multiple passes create progressively smoother curves.

## Stage 4: Boundary Detection

Topic boundaries are detected where the similarity curve shows significant "valleys". Three methods are supported:

### Method 1: Relative (Depth Scoring) — Recommended

This method, inspired by the TextTiling algorithm, finds **local minima** and scores them by their "depth":

```
                    left_peak
                        ╱╲
                       ╱  ╲
                      ╱    ╲         right_peak
                     ╱      ╲           ╱╲
                    ╱        ╲         ╱  ╲
                   ╱          ╲       ╱    ╲
                  ╱            ╲     ╱      ╲
                 ╱              ╲   ╱        ╲
                ╱                ╲ ╱          ╲
               ╱                  V            ╲
              ╱                 valley          ╲
             ╱               (local minimum)     ╲

       depth = ((left_peak - valley) + (right_peak - valley)) / 2
```

**Algorithm:**
1. Find all local minima (points lower than both neighbors)
2. For each minimum at position `i`:
   - Find `left_peak` = maximum similarity from start to position `i`
   - Find `right_peak` = maximum similarity from position `i` to end
   - Compute `depth = average(left_peak - valley, right_peak - valley)`
3. Mark as boundary if `depth > threshold`

**Intuition:** A deep valley indicates the similarity dropped significantly relative to surrounding context — a strong signal of topic change.

### Method 2: Absolute

Simple threshold: mark boundary wherever `similarity < threshold`.

- Pro: Easy to understand and tune
- Con: Doesn't account for varying baseline similarity across documents

### Method 3: Percentile

Mark boundaries where similarity is in the bottom N percentile of all scores.

- Pro: Adapts to document characteristics
- Con: Always creates some boundaries even in single-topic documents

## Stage 5: Segment Assembly

Detected boundaries are converted into final text segments:

1. **Boundary validation:** Ensure boundaries are within valid word range
2. **Text reconstruction:** Words are joined back into readable text

## Key Insight: Why This Works

When the topic changes in a transcript:

1. The speaker starts using different vocabulary and concepts
2. Adjacent windows now contain semantically different content
3. Their embedding vectors point in different directions
4. Cosine similarity between them drops
5. This creates a "valley" in the similarity curve
6. The boundary detector identifies this valley as a topic change

```
Topic A                          Topic B
(discussing sports)              (discussing weather)
        │                               │
        ▼                               ▼
┌─────────────────┐             ┌─────────────────┐
│ "the game was   │             │ "temperatures   │
│  exciting with  │             │  will drop and  │
│  the final goal"│             │  rain expected" │
└─────────────────┘             └─────────────────┘
        │                               │
        ▼                               ▼
   embedding A                     embedding B
   (sports vector)                 (weather vector)
        │                               │
        └───────────┬───────────────────┘
                    │
                    ▼
            low cosine similarity
                    │
                    ▼
            BOUNDARY DETECTED
```

## Configuration Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `window_size` | Words per embedding chunk | 50-200 |
| `stride` | Words between chunk starts | 10-50 |
| `threshold_method` | Boundary detection method | "relative" |
| `threshold_value` | Method-specific threshold | 0.1-0.5 (relative) |
| `smoothing_passes` | Similarity curve smoothing | 0-3 |

**Trade-offs:**
- Larger `window_size` → more context, but boundaries less precise
- Smaller `stride` → finer granularity, but more computation
- Higher `threshold_value` → fewer boundaries (only major topic shifts)

## Output

The pipeline produces:

1. **Segments:** List of topically coherent text sections with:
   - Text content
   - Start/end word positions
   - Word count

2. **Boundary indices:** Word positions where topic changes were detected

3. **Similarity scores:** The full similarity curve (useful for visualization/debugging)

4. **Embeddings data:** (Optional) Per-window embeddings with timestamps for downstream analysis

## References

This approach is based on unsupervised topic segmentation techniques, particularly:

- **TextTiling** (Hearst, 1997): Original depth-scoring algorithm for topic segmentation
- Embedding-based improvements that replace lexical similarity with neural semantic similarity
