"""Prompt loading utilities for article-generation agents."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptLoader:
    """Loads prompt templates from configured files."""

    def __init__(self, *, root_dir: Path) -> None:
        self._root_dir = root_dir

    def load_prompt(self, *, prompt_file: str) -> str:
        """Load a prompt template file."""
        prompt_path = self._root_dir / prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        with open(prompt_path, encoding="utf-8") as handle:
            content = handle.read()
        logger.info("Loaded prompt: %s (%d chars)", prompt_path, len(content))
        return content

    def load_specialist_prompt(self, *, specialists_dir: str, prompt_file: str) -> str:
        """Load a specialist prompt template file."""
        prompt_path = self._root_dir / specialists_dir / prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(f"Specialist prompt file not found: {prompt_path}")
        with open(prompt_path, encoding="utf-8") as handle:
            content = handle.read()
        logger.info("Loaded specialist prompt: %s (%d chars)", prompt_path, len(content))
        return content
