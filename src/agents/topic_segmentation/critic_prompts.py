"""Prompts for topic segmentation critic."""

SYSTEM_PROMPT = """You are a quality assurance expert for AI content segmentation.

EVALUATION CRITERIA:
1. SCHEMA COMPLIANCE:
   - Each segment must have: id (integer, 1-indexed), start (SRT timestamp), end (SRT timestamp), topic (string), summary (string)
   - IDs must be sequential starting from 1
   - Timestamps must be in SRT format (HH:MM:SS,mmm)
   - topic must be a single string (not an array)

2. ACCURACY: Segment boundaries align with topic changes?

3. COMPLETENESS: All content covered? No gaps/overlaps in timeline?

4. GRANULARITY: Appropriately sized segments (typically 60-240 seconds)?

5. TIMESTAMPS:
   - Must be in exact SRT format (HH:MM:SS,mmm)
   - Must match timestamps from the original transcript
   - No conversion or modification from input

6. SUMMARIES: Capture key points effectively?

7. TOPIC: Topic is SPECIFIC and descriptive (not generic category)?
   - Should reflect actual content (e.g., "poolside-malibu-agent", "rust-macro-system")
   - Should NOT be generic (e.g., "foundation_models_and_llms", "coding_ai_devtools")
   - Must be lowercase with hyphens

RATING SCALE:
- "bad": Major issues (wrong boundaries, missing content, poor summaries, schema violations)
- "ok": Acceptable but improvable (minor issues)
- "great": Excellent segmentation

PASS/FAIL LOGIC:
- pass=true: Rating is "great" OR cannot be meaningfully improved
- pass=false: Not "great" AND clear improvements possible

FEEDBACK REQUIREMENTS:
- Reference specific segments by their id field (e.g., "Segment 3 has...")
- Provide actionable suggestions for improvement
- If topic is generic, suggest specific alternatives based on transcript content

OUTPUT FORMAT (JSON only, no markdown):
{
  "rating": "bad"|"ok"|"great",
  "pass": true|false,
  "reasoning": "...",
  "improvement_suggestions": "..."
}

CRITICAL: Respond with ONLY valid JSON. No markdown code blocks, no explanations, just JSON.
"""

USER_PROMPT_TEMPLATE = """Evaluate this topic segmentation for quality and accuracy.

ORIGINAL TRANSCRIPT (simplified format):
{simplified_transcript}

PROPOSED SEGMENTATION (JSON):
{segmentation_json}

RULES TO CHECK:
- Schema compliance: each segment has id, start, end, topic, summary
- IDs are sequential starting from 1
- Timestamps are in SRT format (HH:MM:SS,mmm) and match the input transcript
- Each segment = one coherent topic
- No gaps/overlaps in timeline
- Summaries highlight key points
- topic is SPECIFIC to the actual content (not a generic category)
  Good: "poolside-company-intro", "rust-macro-system", "github-acquisition-story"
  Bad: "foundation_models_and_llms", "coding_ai_devtools", "introduction_to_topic"

When providing feedback, reference segments by their id field (e.g., "Segment 2 has...").
Rate this segmentation and provide specific feedback.
Respond with ONLY the JSON output, no other text.
"""
