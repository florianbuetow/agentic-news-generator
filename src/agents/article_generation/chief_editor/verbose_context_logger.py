"""Per-run sequential logger that writes numbered artifact files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_VALID_FORMATS = frozenset({"json", "txt", "md"})


class VerboseContextLogger:
    """Writes sequentially numbered artifact files into a run directory.

    Files are named: ``{sequence:03d}_{agent_name}_{step}.{ext}``
    """

    def __init__(self, *, artifacts_dir: Path) -> None:
        self._artifacts_dir = artifacts_dir
        self._sequence: int = 0
        self._final_sequence: int = 900

    def log(
        self,
        *,
        agent_name: str,
        step: str,
        content: str | dict[str, object] | list[object],
        fmt: str,
    ) -> Path:
        """Write an artifact file and increment the sequence counter.

        Args:
            agent_name: The agent or processing step name.
            step: What the file represents (e.g. "context", "prompt", "output").
            content: The content to write. Strings are written directly for txt/md;
                     for json fmt, strings are parsed first. Dicts and lists are
                     serialized as JSON.
            fmt: File extension / format. Must be one of "json", "txt", "md".

        Returns:
            The path of the written file.
        """
        if self._sequence >= 900:
            raise RuntimeError(
                f"VerboseContextLogger sequence overflow: {self._sequence} >= 900. "
                f"Reduce editor rounds or agent calls per run."
            )
        path = self._write(
            sequence=self._sequence,
            agent_name=agent_name,
            step=step,
            content=content,
            fmt=fmt,
        )
        self._sequence += 1
        return path

    def log_final(
        self,
        *,
        agent_name: str,
        step: str,
        content: str | dict[str, object] | list[object],
        fmt: str,
    ) -> Path:
        """Write a final artifact file (900+ range) and increment the final sequence counter.

        Same as log() but uses sequence numbers starting at 900 so that final
        artifacts sort last in directory listings.

        Args:
            agent_name: The agent or processing step name.
            step: What the file represents (e.g. "context", "prompt", "output").
            content: The content to write.
            fmt: File extension / format. Must be one of "json", "txt", "md".

        Returns:
            The path of the written file.
        """
        path = self._write(
            sequence=self._final_sequence,
            agent_name=agent_name,
            step=step,
            content=content,
            fmt=fmt,
        )
        self._final_sequence += 1
        return path

    def _write(
        self,
        *,
        sequence: int,
        agent_name: str,
        step: str,
        content: str | dict[str, object] | list[object],
        fmt: str,
    ) -> Path:
        """Build the filename, serialize content, and write to disk."""
        if fmt not in _VALID_FORMATS:
            raise ValueError(f"fmt must be one of {sorted(_VALID_FORMATS)}, got {fmt!r}")

        filename = f"{sequence:03d}_{agent_name}_{step}.{fmt}"
        path = self._artifacts_dir / filename
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

        serialized = self._serialize(content=content, fmt=fmt)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(serialized)

        logger.info("Wrote artifact: %s", path)
        return path

    @staticmethod
    def _serialize(*, content: str | dict[str, object] | list[object], fmt: str) -> str:
        """Serialize content to a string suitable for writing."""
        if isinstance(content, (dict, list)):
            return json.dumps(content, indent=2, ensure_ascii=False)

        if fmt == "json":
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Content for artifact is not valid JSON (first 200 chars): "
                    f"{content[:200]!r}"
                ) from exc
            return json.dumps(parsed, indent=2, ensure_ascii=False)

        return content
