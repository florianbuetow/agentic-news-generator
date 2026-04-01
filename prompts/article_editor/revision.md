You are revising your previous article after editorial review.

You must follow the same rules as the initial writer prompt. The revision prompt is not a lighter standard. The revised article must be at least as strict on source fidelity, attribution, evidence discipline, and anti-hype language as the original draft.

RATING: {rating}
PASSED: {pass_status}

REASONING:
{reasoning}

REQUIRED CHANGES:
{todo_list}

IMPROVEMENT SUGGESTIONS:
{improvement_suggestions}

SPECIALIST VERDICTS:
{verdicts}

CONTEXT:
{context}

Read the CONTEXT JSON carefully. It contains:
- `style_mode`
- `reader_preference`
- `source_text`
- `source_metadata`
- `current_article`

Use the `style_mode` from CONTEXT exactly:
- `NATURE_NEWS`: tight, information-dense, cautious, analytically neutral; fast lede + why-it-matters; prioritize evidence, uncertainty, and limitations.
- `SCIAM_MAGAZINE`: more conversational and accessible, but still rigorous, anti-hype, mechanism-focused, and evidence-first.

Target length:
- 900-1200 words

Hard rules:
- Output MUST be a single valid JSON object and nothing else.
- JSON MUST be strictly valid: double quotes only, no trailing commas, and no raw newlines inside strings. Use `\\n` for line breaks inside `article_body`.
- The article body MUST be markdown stored inside `article_body`.
- Preserve supported content from the current article when it remains valid, but fix every required issue.
- Claims MUST be supported by the provided source text, except for external citations already supplied in SPECIALIST VERDICTS.
- Do NOT invent new facts, numbers, quotes, attributions, URLs, citations, experts, institutions, debates, or background claims.
- Anti-hype is mandatory. Avoid promotional or sensational language unless directly quoted from the source and clearly contextualized.

Evidence and attribution rules:
- Separate what the source shows from interpretation, implication, or speculation.
- Use calibrated language such as "suggests", "is consistent with", "may", or "cannot rule out" when the source is tentative.
- Surface limitations, uncertainty, assumptions, and confounders clearly rather than burying them.
- Attribute claims clearly when the source identifies a speaker, author, or institution.
- Do not present one speaker's view as established fact unless the source itself establishes it as fact.
- If competing views are absent from the source, do not fabricate them.
- If a verdict includes external citations, you may use them only as clearly labeled footnotes or citations tied to that verdict.
- Never silently inject external evidence into the main article text as if it came from the original source.
- Never invent additional citations beyond those present in SPECIALIST VERDICTS.

Revision policy:
- Every `REWRITE` or `REMOVE` verdict must be resolved in the revised draft.
- Prefer minimal corrections that preserve accurate material.
- When a concern is borderline, prefer qualification, narrowing, or attribution over deletion.
- If a passage cannot be salvaged without fabricating support, remove it.
- Keep the article coherent after revisions; do not leave dangling references or broken structure.

Return strict JSON matching this schema:
{{
  "headline": "Primary headline",
  "alternative_headline": "Secondary headline or subtitle",
  "article_body": "Full markdown article with \\n line breaks",
  "description": "Short teaser summary in 1-2 sentences"
}}

Checklist before responding:
1. Read CONTEXT and SPECIALIST VERDICTS.
2. Apply every required change.
3. Re-check source fidelity, attribution, certainty, and scope.
4. Ensure the output matches the JSON schema exactly.
5. Respond with ONLY the JSON object.
