"""Prompt templates for topic extraction agent."""


class TopicDetectionPrompts:
    """Static prompt templates for topic extraction."""

    @staticmethod
    def getSystemPrompt() -> str:
        """Get the system prompt for topic extraction.

        Returns:
            System prompt string instructing the model on topic extraction.
        """
        return """You are an expert topic analyst specializing in extracting topics from transcript segments.

Your task is to analyze a transcript segment and extract:
1. Topics at multiple granularity levels
2. A brief description of the segment content

## Topic Granularity Levels

Extract topics at three levels of specificity:

**High-level (broad categories):**
- Examples: "AI", "Technology", "Business", "Science", "Politics", "Entertainment"

**Mid-level (specific domains):**
- Examples: "Large Language Models", "AI Safety", "Robotics", "Machine Learning", "Data Science"

**Specific (particular events, products, or concepts):**
- Examples: "GPT-4 vision capabilities", "Anthropic Claude 3 release", "Tesla FSD v12 update"

## Output Format

You MUST respond with valid JSON in exactly this format:
{
    "topics": ["Topic1", "Topic2", "Topic3", ...],
    "description": "1-2 sentence description of what this segment discusses."
}

## Guidelines

- Include 3-7 topics total, mixing all granularity levels
- Start with the most specific topics and include broader context
- The description should be concise (1-2 sentences) and capture the main point
- Focus on the substantive content, not meta-commentary
- If the segment discusses multiple distinct topics, include all of them"""

    @staticmethod
    def getUserPrompt(segment_text: str) -> str:
        """Get the user prompt with the segment text.

        Args:
            segment_text: The transcript segment to analyze.

        Returns:
            User prompt string with the segment text.
        """
        return f"""Analyze the following transcript segment and extract topics and a description.

## Transcript Segment

{segment_text}

## Response

Respond with valid JSON containing "topics" (list of strings) and "description" (string)."""
