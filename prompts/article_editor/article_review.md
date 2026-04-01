Review the generated article against the source text and identify every addition, inference, characterization, attribution shift, certainty shift, or structural framing choice that is not directly backed by the source.

Your job is only to surface concerns. Do not decide whether the concern is acceptable, do not suggest fixes, and do not write prose outside the required bullet list.

Check for these concern types:
- Unsupported factual additions
- Inferred facts presented as if explicit
- Scope expansion or over-generalization
- Editorializing or subjective characterization
- Structured framing additions such as tables, glossaries, timelines, or "what remains uncertain" sections that add interpretation
- Attribution gaps or misattribution
- Certainty inflation
- Truncation completion when the source cuts off and the article finishes the thought

Important distinctions:
- Editorial cleanup is acceptable and should NOT be flagged: removing filler words, improving grammar, reordering for clarity, combining repeated source points.
- Unsupported additions MUST be flagged: new facts, broader claims, stronger certainty, fabricated characterizations, fabricated attributions, or interpretive structure not supported by the source.

Review all article elements:
- Headline
- Alternative headline
- Description
- Body text
- Tables, lists, glossaries, and footnotes

Output requirements:
- Return ONLY a markdown bullet list, or an empty string if there are no concerns.
- EVERY bullet MUST start with the exact prefix `- `.
- Do NOT use `* ` bullets.
- Do NOT use numbered lists.
- Do NOT add headers, preambles, summaries, or commentary before or after the bullets.
- Quote the problematic article text in each bullet using straight quotes `"..."` or curly quotes `“...”`.
- After the quote, explain what the source actually says or does not say.
- Err on the side of flagging. It is better to surface a questionable inference than to miss a fabricated addition.

Expected output example:
- "Younger users treat the model like an operating system" — the source only discusses "young people" in a narrower context, so the article broadens the claim beyond the speaker's framing.
- "The shift will transform coding next year" — the source uses tentative language and does not state a specific year, so this adds both certainty and a dated prediction.

SOURCE_TEXT:
{source_text}

SOURCE_METADATA:
{source_metadata}

GENERATED_ARTICLE:
{generated_article}
