<TASK>
Create a structured summary of the provided document, including an overview, an index/table of contents, section-by-section summaries, and final takeaways.
</TASK>

<RULES>
- Preserve the document's actual structure and argument flow.
- Divide the document into logical sections based on topic shifts, speaker/source changes, examples, formal headings, or argument stages.
- Use clear, specific section titles.
- Keep distinct arguments separate when they serve different roles in the document.
- Distinguish between the narrator/author's own claims, quoted or summarized external arguments, cited evidence, and examples.
- Preserve important definitions, concepts, named people, papers, examples, caveats, conclusions, and unresolved issues.
- Do not add unsupported claims, interpretations, examples, recommendations, or conclusions.
- Do not overstate tentative claims.
- Keep technical terms intact.
- Flag unclear, ambiguous, incomplete, or possibly mistranscribed parts.
- Ignore conversational filler, false starts, and repetitive speech inherent to spoken transcripts. Synthesize fragmented sentences into coherent thoughts.
</RULES>

<STEPS>
1. Read the document fully.
2. Identify the central thesis or purpose.
3. Identify the logical sections of the document.
4. Create an index/table of contents from those sections.
5. Summarize each section using the required fields.
6. End with an overall summary, key takeaways, and suggested next steps only when supported by the document.
</STEPS>

<OUTPUT_FORMAT>
# Overview
[Briefly summarize the document's purpose, central thesis, and overall argument.]

# Index
1. [Section title]
2. [Section title]
3. [Section title]

# Section Summaries

## 1. [Section Title]

**Source / speaker:** [Author/narrator, quoted source, cited paper, mixed, or unclear.]

**Overview:** [1-2 sentence overview of this section.]

**Summary:** [Concise summary of the section's main content.]

**Key points:** - [Key point]
- [Key point]

**Important definitions / concepts / arguments:** - [Definition, concept, argument, or conclusion.]
- [Definition, concept, argument, or conclusion.]

**Examples or evidence used:** - [Example or evidence.]
- [Write "None explicitly stated" if absent.]

**Caveats / unclear parts:** - [Caveat, limitation, ambiguity, or "None explicitly stated."]

**Open questions / action items / risks / dependencies:** - [Item, or "None explicitly stated."]

[Repeat for each section.]

# Overall Summary
[Summarize the whole document.]

# Key Takeaways
- [Takeaway]
- [Takeaway]
- [Takeaway]

# Suggested Next Steps
- [Only include next steps supported by the document.]
- [If none are supported, write "None explicitly stated."]
</OUTPUT_FORMAT>

<QUALITY_CHECK>
Before finalizing, verify that:
- The index matches the document's actual structure.
- The section summaries are not merely broad themes.
- Speaker/source distinctions are clear.
- Important examples and caveats are included.
- Unsupported claims or recommendations are excluded.
</QUALITY_CHECK>

Now summarize this document:
<INPUT>
{transcript}
</INPUT>
