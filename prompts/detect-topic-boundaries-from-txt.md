You are detecting major topic boundaries in a transcript.

Input transcript file path: `{{INPUT_TXT_FILE}}`

You will receive transcript content formatted as numbered sentences:

```text
1: First sentence...
2: Second sentence...
3: Third sentence...
```

Task:
1. Read all numbered sentences.
2. Detect only meaningful topic shifts (not minor transitions).
3. Return the sentence numbers where a new topic starts.

Output requirements:
- Return valid JSON only.
- Return a JSON array of integers only.
- Do not return an object.
- Do not include markdown fences.
- Do not include prose, comments, or explanations.
- Numbers must be strictly increasing.
- The first boundary must be `1`.

Example of valid output:

```json
[1, 18, 47, 93]
```

Now detect topic boundaries for this transcript:

{{NUMBERED_SENTENCES}}
