"""File system utility functions for common operations."""

from pathlib import Path


class FSUtil:
    """Utility class for file system operations."""

    @staticmethod
    def find_files_by_extension(
        directory: Path,
        extension: str,
        recursive: bool,
    ) -> list[Path]:
        """Find files by extension in a directory.

        Args:
            directory: Directory to search in.
            extension: File extension to search for (e.g., ".txt", ".srt").
                Can include or exclude the leading dot.
            recursive: If True, search recursively in subdirectories.
                If False, only search in the immediate directory.

        Returns:
            List of Path objects matching the extension, sorted.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        # Normalize extension (ensure it starts with a dot)
        if not extension.startswith("."):
            extension = f".{extension}"

        # Build glob pattern
        pattern = f"*{extension}"

        files = list(directory.rglob(pattern)) if recursive else list(directory.glob(pattern))

        # Filter to only files (exclude directories that might match)
        files = [f for f in files if f.is_file()]

        return sorted(files)

    @staticmethod
    def read_text_file(file_path: Path) -> str:
        """Read UTF-8 encoded text file.

        Args:
            file_path: Path to the text file.

        Returns:
            File contents as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnicodeDecodeError: If the file cannot be decoded as UTF-8.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        return file_path.read_text(encoding="utf-8")

    @staticmethod
    def write_text_file(file_path: Path, content: str, create_parents: bool) -> None:
        """Write UTF-8 encoded text file.

        Args:
            file_path: Path where the file should be written.
            content: Content to write to the file.
            create_parents: If True, create parent directories if they don't exist.

        Raises:
            OSError: If the file cannot be written.
        """
        if create_parents:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text(content, encoding="utf-8")

    @staticmethod
    def find_directories(directory: Path, exclude_hidden: bool) -> list[Path]:
        """Find all subdirectories in a directory.

        Args:
            directory: Directory to search in.
            exclude_hidden: If True, exclude directories starting with a dot.

        Returns:
            List of Path objects for subdirectories, sorted.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        dirs = [d for d in directory.iterdir() if d.is_dir()]

        if exclude_hidden:
            dirs = [d for d in dirs if not d.name.startswith(".")]

        return sorted(dirs)

    @staticmethod
    def ensure_directory_exists(directory: Path) -> None:
        """Ensure a directory exists, creating it if necessary.

        Args:
            directory: Path to the directory.
        """
        directory.mkdir(parents=True, exist_ok=True)
