# Development Rules for Agentic News Generator

## General Coding Principles
- **Never assume any default values anywhere**
- Always be explicit about values, paths, and configurations
- If a value is not provided, handle it explicitly (raise error, use null, or prompt for input)
- **NEVER MAINTAIN BACKWARDS COMPATIBILITY WHEN CHANGING CODE**
  - **ABSOLUTELY NO backwards compatibility - EVER**
  - Backwards compatibility accumulates technical debt
  - Make breaking changes cleanly rather than adding compatibility layers
  - Delete unused code completely instead of keeping it for compatibility
  - **NO legacy format support, NO fallback logic, NO "handle both old and new" code**
  - When changing a format or structure, the old format becomes immediately unsupported
  - If old data exists, it must be regenerated or migrated - never supported in-place

## Git Commit Guidelines
- **NEVER include AI attribution in commit messages**
- **NEVER add "Generated with AI assistance" or similar phrases**
- **NEVER add "Co-Authored-By: [AI Agent Name]" or similar attribution**
- **NEVER run `git add -A` or `git add .` - always stage files explicitly**
- Keep commit messages professional and focused on the changes made
- Commit messages should describe what changed and why, without mentioning AI assistance

## Testing
- After **every change** to the code, the tests must be executed
- Always verify the program runs correctly with `just run` after modifications

## Python Execution Rules
- Python code must be executed **only** via `uv run ...`
  - Example: `uv run src/main.py`
  - **Never** use: `python src/main.py` or `python3 src/main.py`
- The virtual environment must be created and updated **only** via `uv sync`
  - **Never** use: `pip install`, `python -m pip`, or `uv pip`
- All dependencies must be managed through `uv` and declared in `pyproject.toml`

## Justfile Rules
- All Python execution in the justfile uses `uv run`, never `python` directly
- Use `just init` to set up the project
- Use `just run` to execute the main program
- Use `just` (without arguments) to see all available targets

## Project Structure
- All source code lives in `src/`
- Test scripts and utilities go in `scripts/`
- Prompt templates go in `prompts/`
- **Input data**: `data/input/`
- **Output data**: `data/output/`
- **Temporary debug scripts**: `debug/`
  - Create all temporary test and debugging scripts in `debug/` subfolder
  - This makes it easy to identify and clean up scripts that are no longer needed
  - Debug scripts should not be committed to version control
- **Never create Python files in the project root directory**
  - Wrong: `./test.py`, `./helper.py`
  - Correct: `./src/helper.py`, `./scripts/test.py`, `./debug/test_something.py`

## News Generation
- Use OpenAI compatible API (LM Studio) for news generation
- Store prompt templates in `prompts/` directory
- Input topics/sources should be stored in `data/input/`
- Generated articles should be saved to `data/output/`
- Each generated article should include:
  - Title
  - Content
  - Timestamp
  - Source information (if applicable)

## Error Handling
- Scripts should continue processing other items even if one fails
- Failed/invalid outputs must be logged appropriately
- Scripts should track and report success/failure counts
- Exit with code 1 if any items failed, 0 if all succeeded

## API Key Management
- API keys must be loaded from environment variables
- Never hardcode API keys in source code
- Use `.env` file for local development (add to `.gitignore`)
- Document required environment variables in README.md

## Output Organization
- Organize output files by date or topic as appropriate
- Use clear, descriptive filenames
- Include metadata files alongside generated content when useful
- Keep raw outputs separate from processed/refined outputs

