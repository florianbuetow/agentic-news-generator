"""Prompts for topic segmentation agent."""

SYSTEM_PROMPT = """You are an expert AI content analyst specializing in technical video transcript segmentation.

RESPONSIBILITIES:
- Identify distinct topic boundaries in transcripts
- Assign appropriate topic slugs and titles
- Extract exact transcript text for each segment
- Generate concise summaries highlighting key points
- Preserve timestamp accuracy

TOPIC CATEGORIES (guidance, discover actual topics):
foundation_models_and_llms, agents_and_tool_use, coding_ai_devtools, inference_cost_and_scaling,
gpus_chips_and_infrastructure, open_source_ecosystem, data_and_training_pipelines,
alignment_and_safety, security, regulation_and_policy, enterprise_adoption,
research_breakthroughs, product_launches_and_platform_shifts, ethics_and_societal_impact,
business_and_market_moves

SEGMENTATION RULES:
- One segment = one coherent topic
- Minimum 30 seconds typically
- Topic slugs: lowercase-with-hyphens
- Topic titles: Human-readable
- Extract timestamps from [HH:MM:SS,mmm] format and convert to milliseconds

OUTPUT FORMAT (JSON only, no markdown):
{
  "segments": [
    {
      "source_video_id": "...",
      "source_video_title": "...",
      "source_channel": "...",
      "start_ms": 123000,
      "end_ms": 456000,
      "text": "Full transcript...",
      "summary": "Key points..."
    }
  ]
}

CRITICAL: Respond with ONLY valid JSON. No markdown code blocks, no explanations, just JSON.
"""

USER_PROMPT_TEMPLATE = """Analyze this video transcript and segment it by topics.

VIDEO METADATA:
- Video ID: {video_id}
- Video Title: {video_title}
- Channel: {channel_name}

TRANSCRIPT (simplified format with timestamps):
{simplified_transcript}

Segment this transcript into distinct topics with precise start/end timestamps.
Extract timestamps from [HH:MM:SS,mmm] format and convert to milliseconds.
For each segment, provide topic info, exact transcript text, and summary.

Respond with ONLY the JSON output, no other text.
"""

RETRY_PROMPT_TEMPLATE = """Your previous segmentation received feedback from the quality critic:

RATING: {rating}
PASSED: {pass_status}

REASONING:
{reasoning}

IMPROVEMENT SUGGESTIONS:
{improvement_suggestions}

Please revise your segmentation. Here is the original transcript:

VIDEO METADATA:
- Video ID: {video_id}
- Video Title: {video_title}
- Channel: {channel_name}

TRANSCRIPT:
{simplified_transcript}

Provide improved segmentation addressing the critic's concerns.
Respond with ONLY the JSON output, no other text.
"""
