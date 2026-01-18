# Magnetic Segment Merging

## Problem

Small segments (< 30 seconds) from boundary detection can be too short for meaningful topic extraction. These micro-segments often represent transitions, brief asides, or detection noise rather than standalone topics.

## Proposed Solution

"Magnetic" merging where small segments attach to adjacent large segments on both sides. The approach treats large segments as "magnets" that pull in neighboring small segments, creating overlapping merged segments that preserve context.

## Visual Example

```
BEFORE: [LARGE1] [s1] [s2] [s3] [LARGE2] [s4]

AFTER:
  Merged 1: [LARGE1 + s1 + s2 + s3]
  Merged 2: [s1 + s2 + s3 + LARGE2 + s4]
```

Each large segment becomes a merged segment that includes:
- The large segment itself
- All consecutive small segments to the right (until the next large segment)
- All consecutive small segments from the left (that were also pulled by the previous large segment)

## Key Behaviors

1. **Small segments duplicate into both neighbors** - This is intentional, not a bug. A transitional segment between two topics may be relevant to understanding both.

2. **Consecutive small segments all merge together** - If you have `[s1] [s2] [s3]` between two large segments, all three attach to both sides.

3. **One merged segment per original large segment** - The final output contains exactly as many segments as there were large segments (plus edge cases).

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Small segment at start | Attach to the right neighbor only |
| Small segment at end | Attach to the left neighbor only |
| All segments are small | Merge everything into a single segment |
| No small segments | Return original segments unchanged |
| Only one large segment | Merge all small segments into it |

## Example Walkthrough

Given segments with durations:
```
[60s] [10s] [5s] [90s] [8s] [120s]
```

With a 30-second threshold:
- Large segments: 60s, 90s, 120s
- Small segments: 10s, 5s, 8s

Result:
```
Merged 1: [60s + 10s + 5s]         # 75 seconds total
Merged 2: [10s + 5s + 90s + 8s]    # 113 seconds total
Merged 3: [8s + 120s]               # 128 seconds total
```

Note how the 10s and 5s segments appear in both Merged 1 and Merged 2, and the 8s segment appears in both Merged 2 and Merged 3.

## Future Implementation Notes

### Class Structure

Add a `SegmentMerger` class to handle the merging logic:

```python
class SegmentMerger:
    def __init__(self, min_segment_duration: float):
        self.min_segment_duration = min_segment_duration

    def merge_magnetic(self, segments: list[Segment]) -> list[MergedSegment]:
        """Merge small segments into adjacent large segments."""
        ...
```

### Integration Point

Call from `detect-boundaries.py` after boundary detection but before output:

```python
# After detecting boundaries
boundaries = detect_boundaries(audio_file)
segments = create_segments(boundaries)

# Apply magnetic merging
merger = SegmentMerger(min_segment_duration=config.min_segment_duration)
merged_segments = merger.merge_magnetic(segments)
```

### Configuration

Make the threshold configurable in `config.yaml`:

```yaml
boundary_detection:
  min_segment_duration: 30.0  # seconds - segments shorter than this get merged
```

## Rationale

This approach is preferred over simpler alternatives because:

1. **Preserves context** - Short segments often contain important transitional information that provides context for adjacent topics.

2. **No information loss** - By duplicating small segments into both neighbors, we ensure nothing is discarded.

3. **Natural topic boundaries** - Large segments naturally correspond to distinct topics; small segments are usually transitions or sub-points.

4. **Deterministic** - The algorithm produces consistent results without requiring ML or heuristics beyond the duration threshold.
