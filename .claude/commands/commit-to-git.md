Create clean, logical commits from current working tree changes.

## Objective
Turn working tree changes into well-organized commits with clear messages. Do NOT push or modify remotes unless explicitly asked.

## Safety Rules
- Never commit secrets, API keys, or tokens
- Never rewrite history (no rebase/reset --hard/force push)
- Never stage all files with `git add -A` or `git add .` - always stage explicitly

## Workflow

1. **Review changes**
   - Check status and diffs
   - Review recent commits for context
   - Identify change types: features, fixes, refactors, tests, docs, config

2. **Group into logical commits**
   - Each commit should have one clear purpose
   - Separate unrelated changes into different commits
   - Suggested order: refactors → features/fixes → tests → docs → chores

3. **Create commits**
   - Stage files explicitly for each commit
   - Use Conventional Commits format: `type(scope): description`
   - Types: feat, fix, refactor, test, docs, chore, ci, build
   - Keep subject line under 72 characters
   - Run tests if available and relevant

4. **Verify**
   - Confirm working tree is clean
   - Review commit log

## Example
```
fix(detector): handle empty input in repetition detector
test(detector): add test cases for edge conditions
docs: update README with hallucination detection examples
```
