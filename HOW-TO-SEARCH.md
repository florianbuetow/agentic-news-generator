# How to Search the Transcript Corpora

Three justfile targets search the project's two transcript corpora:

- **Cleaned transcripts** (`*.txt`) — directory from `Config.get_data_downloads_transcripts_cleaned_dir()`
- **Summaries** (`*.md`) — directory from `Config.get_data_downloads_transcripts_summaries_dir()`

Both directories are resolved at runtime from `config/config.yaml`, so the commands always target the configured data location. Run all commands from the project root.

## Quick Reference

| Goal | Command |
|------|---------|
| Browse/open a cleaned transcript interactively | `just find` |
| Rank cleaned transcripts by a query | `just find "<query>"` |
| Browse/open a summary interactively (live content search) | `just search` |
| Rank summaries by a query | `just search "<query>"` |
| Ranked report across **both** corpora for several keywords | `just research "<kw1>, <kw2>, ..."` |

## At a Glance

| | `find` | `search` | `research` |
|---|---|---|---|
| **Corpus** | cleaned transcripts (`.txt`) | summaries (`.md`) | both |
| **Argument** | optional | optional | required |
| **No-arg mode** | interactive `fzf` (filename match) | interactive `fzf` (live content match) | n/a |
| **Query matching** | case-sensitive regex | case-sensitive regex | case-insensitive literal, multi-term |
| **Opens a file?** | yes (Sublime Text) | yes (Sublime Text) | no — report only |
| **Output** | `count  channel/file` | `count  channel/file` | two ranked sections + full paths + totals |

`find` and `search` are interactive "jump to one file" tools, each scoped to a single corpus. `research` is a non-interactive, multi-keyword ranked report across both corpora.

## `find [QUERY]` — cleaned transcripts

Searches the cleaned transcripts (`*.txt`).

- **No query** → opens an interactive `fzf` picker over all transcript files, matching on the **filename** (channel/file), with a `bat` preview. Selecting an entry opens it in Sublime Text (`subl`).
- **With query** → non-interactive. Runs `rg -c "<QUERY>"` (case-sensitive regex), counts matching lines per file, sorts by count descending, and prints `count  channel/file`.

```bash
just find                 # interactive picker
just find "context window"
```

## `search [QUERY]` — summaries

Searches the transcript summaries (`*.md`).

- **No query** → interactive `fzf` with **live full-content search**: every keystroke re-runs `rg --fixed-strings` over summary contents, with a `glow`-rendered markdown preview that highlights matches. Selecting an entry opens it in Sublime Text.
- **With query** → identical to `find`'s query mode (`rg -c`, case-sensitive regex, ranked `count  channel/file`), but over summaries.

```bash
just search               # interactive live-content picker
just search "agent loop"
```

## `research KEYWORDS` — both corpora

Searches cleaned transcripts **and** summaries — the union of what `find` and `search` cover. Non-interactive; opens nothing.

- **Argument is required.** Empty input fails with `✗ research failed: no keywords provided`.
- Takes **comma-separated** keywords/phrases, trims each, and combines them as multiple `rg -c --ignore-case --fixed-strings` patterns (case-insensitive, literal, OR'd together).
- Prints two ranked sections — *Cleaned Transcripts* and *Summaries* — each file shown as `count  channel/title` with the **full path** on a second line, ending with a `✓ research completed: N transcript(s), M summary(ies)` summary.

```bash
just research "tool use, function calling, MCP"
```

## Choosing a Target

- Know roughly which file you want and want to open it → `just find` (transcripts) or `just search` (summaries).
- Want to see which files mention a single term, ranked → `just find "<query>"` or `just search "<query>"`.
- Exploring a topic across everything with several related terms → `just research "<kw1>, <kw2>, ..."`.
