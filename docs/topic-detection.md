# Topic Detection Pipeline

This document explains the embedding-based topic segmentation approach used to automatically detect topic boundaries in transcripts.

## Overview

The pipeline segments continuous text (e.g., podcast transcripts) into topically coherent sections without requiring any labeled training data. It works by:

1. Sliding a window across the text
2. Embedding each window into a semantic vector
3. Measuring similarity between adjacent windows
4. Detecting boundaries where similarity drops significantly

## Pipeline Architecture

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

## Running the Pipeline

The pipeline is executed in three steps via justfile targets:

| Target | Script | Behavior |
|--------|--------|----------|
| `just topics-embed` | `scripts/generate-embeddings.py` | Creates `_embeddings.json` files. **Skips existing files** (incremental). |
| `just topics-boundaries` | `scripts/detect-boundaries.py` | Creates `_segmentation.json` files. **Overwrites existing files** (fast operation). |
| `just topics-extract` | `scripts/extract-topics.py` | Extracts topics from segments using LLM. |
| `just topics-all` | Runs all three steps | Complete pipeline execution. |

**Incremental Processing:**
- `topics-embed` is the slowest step (requires embedding model inference). It skips files that already exist, so you can safely re-run it to process only new transcripts.
- `topics-boundaries` is fast (pure computation on pre-computed embeddings). It always regenerates all segmentation files, allowing you to experiment with different threshold parameters without re-computing embeddings.
- `topics-extract` uses LLM inference to extract topic titles and summaries from each segment.

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
