"""Compile hand-authored newspaper Markdown into Nuxt content."""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import yaml


class NewspaperLayout(TypedDict):
    """Front-page layout keyed by selected article slugs."""

    hero: str
    featured: list[str]
    secondary: str
    sidebar: list[list[str]]
    briefs: list[list[str]]


@dataclass(frozen=True)
class SourceArticle:
    """Validated source article ready for selection and staging."""

    source_path: Path
    slug: str
    published_date: dt.date
    content: str


@dataclass(frozen=True)
class CompileResult:
    """Summary of generated newspaper content."""

    selected_slugs: tuple[str, ...]
    article_output_dir: Path
    layout_output_path: Path


class NewspaperArticleCompilationError(Exception):
    """Base error for newspaper article compilation failures."""


class NewspaperArticleValidationError(NewspaperArticleCompilationError):
    """Validation error that reports the failing path and violated rule."""

    def __init__(self, source_path: Path, rule: str) -> None:
        """Initialize the validation error.

        Args:
            source_path: Path where validation failed.
            rule: Violated validation rule.
        """
        self.source_path = source_path
        self.rule = rule
        super().__init__(f"{source_path}: {rule}")


def selected_article_count() -> int:
    """Return the exact number of articles required by the newspaper layout."""
    return 21


def required_frontmatter_fields() -> tuple[str, ...]:
    """Return frontmatter fields required by the source article contract."""
    return (
        "title",
        "subtitle",
        "slug",
        "author",
        "date",
        "dateline",
        "tags",
        "summary",
        "image",
    )


def slug_pattern() -> re.Pattern[str]:
    """Return the accepted source article slug pattern."""
    return re.compile(r"^[a-z0-9-]+$")


def date_pattern() -> re.Pattern[str]:
    """Return the exact source article date pattern."""
    return re.compile(r"^\d{4}-\d{2}-\d{2}$")


def compile_newspaper_articles(source_dir: Path, project_root: Path) -> CompileResult:
    """Validate source articles and stage selected Nuxt content.

    Args:
        source_dir: Directory containing manually dropped Markdown articles.
        project_root: Repository root used to resolve generated frontend paths.

    Returns:
        Summary of the generated article and layout outputs.

    Raises:
        NewspaperArticleCompilationError: If validation or staging fails.
    """
    source_articles = load_source_articles(source_dir)
    selected_articles = select_articles(source_articles)
    content_dir = project_root / "frontend" / "newspaper" / "content"
    article_output_dir = content_dir / "articles"
    layout_output_path = content_dir / "layout.json"

    article_output_dir.mkdir(parents=True, exist_ok=True)
    clear_generated_articles(article_output_dir)
    write_selected_articles(selected_articles, article_output_dir)
    write_layout(selected_articles, layout_output_path)

    return CompileResult(
        selected_slugs=tuple(article.slug for article in selected_articles),
        article_output_dir=article_output_dir,
        layout_output_path=layout_output_path,
    )


def load_source_articles(source_dir: Path) -> list[SourceArticle]:
    """Load and validate all source article files.

    Args:
        source_dir: Directory containing source Markdown articles.

    Returns:
        Validated source articles.

    Raises:
        NewspaperArticleValidationError: If any source validation rule fails.
    """
    markdown_files = validate_source_folder(source_dir)
    articles = [parse_source_article(file_path) for file_path in markdown_files]
    validate_unique_slugs(articles)
    validate_filename_slugs(articles)
    return articles


def validate_source_folder(source_dir: Path) -> list[Path]:
    """Validate source folder existence and Markdown file inventory."""
    if not source_dir.exists():
        raise NewspaperArticleValidationError(source_dir, "source folder exists")
    if not source_dir.is_dir():
        raise NewspaperArticleValidationError(source_dir, "source folder is a directory")

    source_files = sorted((path for path in source_dir.iterdir() if path.is_file()), key=lambda path: path.name)
    for source_file in source_files:
        if source_file.suffix != ".md":
            raise NewspaperArticleValidationError(source_file, "each file extension is .md")

    markdown_files = [source_file for source_file in source_files if source_file.suffix == ".md"]
    article_count = selected_article_count()
    if len(markdown_files) < article_count:
        raise NewspaperArticleValidationError(
            source_dir,
            f"source folder contains at least {article_count} .md files",
        )

    return markdown_files


def parse_source_article(source_path: Path) -> SourceArticle:
    """Parse and validate a single source article file."""
    content = source_path.read_text(encoding="utf-8")
    frontmatter_text, body = split_frontmatter(content, source_path)
    metadata = parse_frontmatter(frontmatter_text, source_path)
    validate_required_frontmatter(metadata, source_path)
    validate_body(body, source_path)

    slug = cast(str, metadata["slug"])
    validate_slug(slug, source_path)
    published_date = parse_article_date(metadata["date"], source_path)

    return SourceArticle(
        source_path=source_path,
        slug=slug,
        published_date=published_date,
        content=content,
    )


def split_frontmatter(content: str, source_path: Path) -> tuple[str, str]:
    """Split Markdown into YAML frontmatter and body."""
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise NewspaperArticleValidationError(source_path, "YAML frontmatter starts with ---")

    closing_index = next((index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if closing_index is None:
        raise NewspaperArticleValidationError(source_path, "YAML frontmatter has a closing --- delimiter")

    frontmatter_text = "".join(lines[1:closing_index])
    body = "".join(lines[closing_index + 1 :])
    return frontmatter_text, body


def parse_frontmatter(frontmatter_text: str, source_path: Path) -> dict[str, object]:
    """Parse YAML frontmatter into a string-keyed mapping."""
    try:
        parsed_frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        raise NewspaperArticleValidationError(source_path, f"YAML frontmatter parses: {exc}") from exc

    if not isinstance(parsed_frontmatter, dict):
        raise NewspaperArticleValidationError(source_path, "YAML frontmatter is a mapping")

    metadata: dict[str, object] = {}
    parsed_mapping = cast(dict[object, object], parsed_frontmatter)
    for key, value in parsed_mapping.items():
        if not isinstance(key, str):
            raise NewspaperArticleValidationError(source_path, "YAML frontmatter keys are strings")
        metadata[key] = value
    return metadata


def validate_required_frontmatter(metadata: dict[str, object], source_path: Path) -> None:
    """Validate required frontmatter fields and field types."""
    for field_name in required_frontmatter_fields():
        if field_name not in metadata:
            raise NewspaperArticleValidationError(source_path, f"required frontmatter field '{field_name}' is present")

        value = metadata[field_name]
        if field_name == "tags":
            validate_tags(value, source_path)
        elif field_name == "date":
            validate_date_field_presence(value, source_path)
        elif not isinstance(value, str) or not value.strip():
            raise NewspaperArticleValidationError(
                source_path,
                f"required frontmatter field '{field_name}' is non-empty",
            )


def validate_tags(value: object, source_path: Path) -> None:
    """Validate tags as a non-empty list of non-empty strings."""
    if not isinstance(value, list) or not value:
        raise NewspaperArticleValidationError(source_path, "tags is a non-empty list of strings")

    tags = cast(list[object], value)
    for tag in tags:
        if not isinstance(tag, str) or not tag.strip():
            raise NewspaperArticleValidationError(source_path, "tags is a non-empty list of strings")


def validate_date_field_presence(value: object, source_path: Path) -> None:
    """Validate date frontmatter is present before parsing exact date semantics."""
    if isinstance(value, str) and value.strip():
        return
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return
    raise NewspaperArticleValidationError(source_path, "required frontmatter field 'date' is non-empty")


def validate_body(body: str, source_path: Path) -> None:
    """Validate article body exists after frontmatter."""
    if not body.strip():
        raise NewspaperArticleValidationError(source_path, "body exists after the closing frontmatter delimiter")


def validate_slug(slug: str, source_path: Path) -> None:
    """Validate slug characters."""
    if not slug_pattern().fullmatch(slug):
        raise NewspaperArticleValidationError(source_path, "slug uses only lowercase a-z, 0-9, and -")


def parse_article_date(date_value: object, source_path: Path) -> dt.date:
    """Validate and parse a YYYY-MM-DD article date."""
    if isinstance(date_value, dt.date) and not isinstance(date_value, dt.datetime):
        return date_value

    if not isinstance(date_value, str) or not date_pattern().fullmatch(date_value):
        raise NewspaperArticleValidationError(source_path, "date parses as YYYY-MM-DD")

    try:
        return dt.datetime.strptime(date_value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise NewspaperArticleValidationError(source_path, "date parses as YYYY-MM-DD") from exc


def validate_unique_slugs(articles: list[SourceArticle]) -> None:
    """Validate that source article slugs are unique."""
    seen_slugs: dict[str, Path] = {}
    for article in articles:
        existing_path = seen_slugs.get(article.slug)
        if existing_path is not None:
            raise NewspaperArticleValidationError(
                article.source_path,
                f"slugs are unique; duplicate slug '{article.slug}' also used by {existing_path}",
            )
        seen_slugs[article.slug] = article.source_path


def validate_filename_slugs(articles: list[SourceArticle]) -> None:
    """Validate filename stems match frontmatter slugs."""
    for article in articles:
        if article.source_path.stem != article.slug:
            raise NewspaperArticleValidationError(
                article.source_path,
                f"filename stem equals frontmatter slug '{article.slug}'",
            )


def select_articles(articles: list[SourceArticle]) -> list[SourceArticle]:
    """Select exactly 21 articles by date descending, then slug ascending."""
    article_count = selected_article_count()
    if len(articles) < article_count:
        raise NewspaperArticleCompilationError(f"Need {article_count} valid articles, found {len(articles)}")

    return sorted(articles, key=lambda article: (-article.published_date.toordinal(), article.slug))[:article_count]


def clear_generated_articles(article_output_dir: Path) -> None:
    """Remove stale generated Markdown articles from the output folder."""
    for generated_article_path in sorted(article_output_dir.glob("*.md")):
        if generated_article_path.is_file():
            generated_article_path.unlink()


def write_selected_articles(selected_articles: list[SourceArticle], article_output_dir: Path) -> None:
    """Write selected source article content into the Nuxt article folder."""
    for article in selected_articles:
        output_path = article_output_dir / f"{article.slug}.md"
        output_path.write_text(article.content, encoding="utf-8")


def build_layout(selected_slugs: list[str]) -> NewspaperLayout:
    """Build front-page layout assignments from selected slugs."""
    article_count = selected_article_count()
    if len(selected_slugs) != article_count:
        raise NewspaperArticleCompilationError(f"layout generation requires exactly {article_count} slugs, found {len(selected_slugs)}")

    return {
        "hero": selected_slugs[0],
        "featured": selected_slugs[1:6],
        "secondary": selected_slugs[6],
        "sidebar": [
            selected_slugs[7:10],
            selected_slugs[10:13],
        ],
        "briefs": [
            selected_slugs[13:15],
            selected_slugs[15:17],
            selected_slugs[17:19],
            selected_slugs[19:21],
        ],
    }


def write_layout(selected_articles: list[SourceArticle], layout_output_path: Path) -> None:
    """Write selected article slugs to the Nuxt layout data file."""
    selected_slugs = [article.slug for article in selected_articles]
    layout = build_layout(selected_slugs)
    layout_output_path.parent.mkdir(parents=True, exist_ok=True)
    layout_output_path.write_text(json.dumps(cast(dict[str, Any], layout), indent=2) + "\n", encoding="utf-8")
