Read the subtitles from `{{INPUT_SRT_FILE}}`.

Decide where YouTube chapter boundaries should start.

Write the final result to `{{OUTPUT_JSON_FILE}}`.

The output file must be valid JSON in this format:

```json
{
  "chapters": [
    { "start_seconds": 0, "title": "Chapter title" },
    { "start_seconds": 123, "title": "Next chapter title" }
  ]
}
```

Rules:
- Create chapter boundaries only for real section/topic shifts
- Do not create a new chapter for minor sentence transitions
- Titles must be short, concrete, and YouTube-chapter style
- Return only the final JSON content for `{{OUTPUT_JSON_FILE}}`
