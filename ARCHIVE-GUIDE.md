# How to archive code and data

Procedure for moving stale code and its on-disk data artifacts out of the active tree into a long-term archive. The archive is preserved permanently; nothing is ever deleted.

This guide is tracked in git as project documentation. The archive it describes is not: the archive directory lives outside the repo on the external data volume, and its absolute path must never appear in any tracked file (see "Where the archive lives" below).

---

## When to archive

- The code or data is no longer used by any active pipeline.
- The owner has explicitly chosen to preserve it rather than delete it.
- A future reader will need the snapshot accessible without doing git archaeology.

---

## Where the archive lives

A sibling of the project's `data/` directory on the external data volume — **not** inside `data/`, **not** in the repo.

To find it:
1. Read `data.data_dir` from `config/config.yaml` — that resolves to `<external_data_root>/agentic-news-generator/data/`.
2. Go one level up and into `archived_code_and_data/`. Create it if it does not exist.
3. Inside `archived_code_and_data/`, create a subfolder **named for the feature being removed**, in kebab-case (e.g. `topic-boundary-detection/`). One subfolder per removed feature; never mix features in a single folder. This per-feature subfolder is the `<archive>` root referred to throughout the rest of this document.

The archive path is private. It must never appear in any tracked file (README, CHANGELOG, AGENTS.md, code, commits). Refer to it abstractly — *"the external archive"*.

---

## Required layout

```
archived_code_and_data/
└── <feature-name>/            ← one subfolder per removed feature (kebab-case)
    ├── README.md              ← mandatory; describes contents and origin
    ├── code/                  ← mirrors original repo paths
    └── data/                  ← mirrors original data paths (may be empty — see step 4)
```

Files preserve their **exact original path** under `code/` or `data/`. Example: `src/topic_detection/boundary_metrics.py` → `<archive>/code/src/topic_detection/boundary_metrics.py`. Do not flatten or rename.

The exception is **partial extracts** from files that stay in the repo (see step 3) — those get a descriptive suffix instead of an exact mirror, because their parent file is not being moved.

Do not include `__pycache__/`, `.pyc`, or other regenerable build artifacts.

---

## The workflow

### 1. Identify the scope

Be **narrow**, not broad. Archive only what is unambiguously part of the feature; tangential utilities and generic infrastructure stay out. Git history covers anything omitted.

Two situations occur, and a single archival can mix both:
- **Code still live in the working tree** — the feature is being removed *now*, as part of this change. This is the common case.
- **Code already deleted by a prior commit** — enumerate candidates via:
  ```bash
  git diff --name-status <removal_commit>^ <removal_commit> | grep '^D'
  ```
  Only `D` (deleted) entries are candidates. `M` (modified) entries are not — the current version still exists at that path.

**Verify wiring, not names.** A filename that contains the feature's keyword may belong to a *different* feature. (Example: `prompts/extract-topic-boundaries.md` reads like topic-boundary detection, but it is wired to the `topics` / extract-topics feature and must stay.) Before classifying a file as in-scope, confirm which config section and which script actually reference it.

**Trace dependencies before removing anything shared.** `grep` the whole repo for every symbol, config key, prompt path, and script name in scope. For each consumer that will *remain* in the repo, confirm it degrades gracefully without the removed piece (e.g. a config reader using `.get(key, {})` rather than a hard lookup). A consumer that would raise is a blocker — resolve it before proceeding.

- **Data**: list every directory on the external data volume that the feature produced or consumed.

### 2. Move the code into the archive

Create the destination parent first, then move by origin. `git mv` **cannot** cross the repo boundary onto the external volume — so the mechanics differ from a repo-internal move.

```bash
mkdir -p "<archive>/code/$(dirname <original_path>)"
```

- **(a) Live, git-tracked file** — copy into the archive, then `git rm` the original. The content also survives in git history, but the archive holds it at its mirrored path regardless:
  ```bash
  cp -p <original_path> "<archive>/code/<original_path>"
  git rm <original_path>
  ```
- **(b) Live, untracked file** — plain move:
  ```bash
  mv <original_path> "<archive>/code/<original_path>"
  ```
- **(c) File already deleted by a prior commit** — restore from history with `git show` (not `git checkout`, which would resurrect the file at its active path):
  ```bash
  git show "<commit>^:<original_path>" > "<archive>/code/<original_path>"
  ```

Afterwards, delete any stale bytecode the moved module left behind (e.g. `src/.../__pycache__/<module>.cpython-XXX.pyc`). Never copy `__pycache__/` into the archive.

### 3. Snapshot and remove the configuration

A feature's configuration usually lives in files that **stay** in the repo — `config/config.yaml`, `config/config.yaml.template`, and a shared `src/config.py`. You cannot move these whole files; instead extract the feature's slice, snapshot it, then remove it from the live file.

⚠️ **`config/config.yaml` is git-ignored — it has NO git history.** Whatever you remove from it is unrecoverable unless you snapshot it first. Snapshot **before** editing.

For each config file with a feature-specific slice:
1. Copy the removed slice into the archive as a partial extract with a descriptive suffix:
   - YAML section → `<archive>/code/config/<file>__<feature>.snippet.yaml`
   - Code symbols → `<archive>/code/<path>__removed_<feature>_symbols.py` (a PARTIAL, non-runnable extract — the live file keeps every other symbol).
2. Remove the slice from the live file. In `config.py` that means the Pydantic model, its `__init__` validation wiring, the validator method, the getter, and any derived-path helper — all of them.

These extracts are the one place the layout does *not* mirror the original path exactly, because the parent file is not being archived.

### 4. Move data from the external volume

```bash
mkdir -p "<archive>/data/$(dirname <original_data_path>)"
mv "<external_data_root>/agentic-news-generator/data/<original_data_path>" \
   "<archive>/data/<original_data_path>"
```

The external data volume has no backup. Always `mv` into the archive; never `cp`, never `rm`. Preservation is the default — if you mean to remove something, move it to the archive. If deletion seems necessary, stop and ask the operator.

It is normal for `data/` to end up **empty**: an experimental feature may have produced no current on-disk artifacts at archival time. Re-check the produced/consumed directories at execution time and, if none exist, state so explicitly in the README rather than leaving the empty `data/` unexplained.

### 5. Write the archive README

At `<archive>/README.md`:
1. **Origin** — what was archived, when, and (if applicable) which commit removed it. State the scope boundary explicitly: name the closely-related things that were deliberately *left in the repo* and why (the name-trap files especially).
2. **Layout** — what lives under `code/` and `data/`.
3. **Content removed from files that stayed** — list each partial extract (config snippets, removed `config.py` symbols, removed justfile targets) and flag which ones have no git history (the git-ignored `config.yaml`).
4. **Why archived, not deleted** — the reason for preservation.
5. **How to restore** — `git show <commit>^:<path>` for tracked code/config; copy-back from `code/<path>` for untracked code; re-insert the snippet for the git-ignored `config.yaml`.
6. **Do not modify** — the archive is a frozen snapshot.

### 6. Update the project CHANGELOG

Add one short line under `### Removed` in `CHANGELOG.md`:

```
- <brief description of what was removed>
```

Do NOT include in the entry: filesystem paths (the repo is public, the archive path is private), commit hashes (`git log` finds them), or inventories of archive contents (that is what the archive README is for).

### 7. Update README.md and AGENTS.md — only when factually wrong

Update only when:
- The README's repository structure section names a removed directory or script.
- AGENTS.md's documentation index points to a removed doc file.
- A specific feature description in either file is now factually incorrect.

Do NOT touch aspirational or forward-looking sections (e.g., "Project Status" WIP markers) or descriptions of concepts whose data was archived but whose script can still repopulate. If nothing in either file is factually wrong, do not edit them.

### 8. Update the justfile — only when a target's script was archived

For each `just` target whose script was archived:
- If a replacement script exists → update the target to call the replacement; update the help-section description.
- If no replacement exists → remove the target, its help-section line, and any reference to it inside an **aggregate target** (e.g. drop the archived step from a `*-all` recipe so the remaining pipeline still runs).

Do NOT remove targets whose scripts still exist and work.

### 9. Verify

```bash
just ci
```

Must pass. Any new failure means the move broke an active code path — investigate, do not silently bypass.

### 10. Stage for operator review

Stage only the tracked files you modified (CHANGELOG.md, plus any minimal doc/justfile/config-template/`config.py` edits) and the `git rm` deletions. Never `git add .` or `git add -A`. The archived files live outside the repo and are never staged; `config/config.yaml` is git-ignored and will not appear. Do not commit or push — leave the staged change for the operator.

---

## Hard rules

1. **Never delete from the external data volume without explicit per-file authorization.** Move to the archive instead.
2. **Never write private filesystem paths into tracked files.** Use abstract references.
3. **Never include `__pycache__/` or other build artifacts in the archive.**
4. **Match the move mechanic to the situation.** `git mv` for repo-internal moves only — it cannot cross onto the external volume. To archive a live tracked file, `cp` it into the archive then `git rm` the original. `mv` untracked files and external-volume data. `git show <commit>^:<path>` for already-deleted code. `rm` is never the archival mechanic.
5. **Snapshot git-ignored config before removing it.** `config/config.yaml` has no history; the archived snippet is the only copy.
6. **Preserve original paths under `code/` and `data/`.** The archive mirrors; it does not redesign. The only exception is partial extracts from files that stay in the repo.
7. **The README at the archive root is mandatory.** Without it the archive is opaque.
8. **Narrow scope.** When in doubt, archive less. The git history covers what is omitted.
9. **One feature per archive subfolder.** Each removal gets its own `<feature-name>/` folder.

---

## Pitfalls observed in past archive operations

- **Over-archival**: pulling in supporting infrastructure that was tangential to the feature, on the assumption "it might be related". Stay narrow — the operator will push back if scope is too broad, and the resulting churn is wasteful.
- **Trusting the filename over the wiring**: a file whose name contains the feature keyword may belong to another feature (`extract-topic-boundaries.md` is owned by the `topics` feature, not boundary detection). Confirm the config/script that actually references it before classifying it in-scope.
- **Removing config without snapshotting the git-ignored copy**: `config/config.yaml` has no git history. Remove its section without first copying it into the archive and it is gone for good.
- **Forgetting a consumer that stays**: removing a config key or symbol that a remaining script still reads breaks it. Trace every consumer first and confirm graceful degradation before removing.
- **Leaving an aggregate target dangling**: removing a `just` target but not the line that calls it from a `*-all` recipe breaks the aggregate. Remove both.
- **Leaking private paths**: writing the absolute archive path into CHANGELOG.md, README.md, or any other tracked file. The repo is public; the path is private. Always abstract.
- **Forgetting `__pycache__/`**: a moved module leaves stale bytecode under `src/.../__pycache__/`. Remove it; it is not part of the archive.
- **Editing AGENTS.md or README.md when nothing factually changed**: aspirational language about a concept can remain accurate even after its data was snapshotted, if a script can still repopulate it. Touch tracked docs only when they are factually wrong.
- **Confusing modification vs. deletion in a removal commit**: `git diff --name-status` distinguishes `D` (deleted) from `M` (modified). Only archive `D` entries unless explicitly told otherwise — `M` entries still exist at their original path with current content.
- **Asking to delete instead of asking where to move**: the framing matters. The external volume has no backup; the operator will reject `rm` proposals on principle. Frame everything as a move.
