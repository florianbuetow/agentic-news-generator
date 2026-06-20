"""Tests for staging newspaper Markdown articles into Nuxt content."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from src.processing.newspaper_articles import (
    NewspaperArticleValidationError,
    compile_newspaper_articles,
    selected_article_count,
)

PRESERVED_BODY = """**SAN FRANCISCO** — The opening paragraph stays in the article body.

## Preserved Heading

![Diagram](https://example.com/diagram.jpg)

This paragraph keeps a [reference link](https://example.com/source).

- First list item
- Second list item

> A quoted paragraph remains intact.

```python
print("code block")
```
"""


def write_article(
    source_dir: Path,
    slug: str,
    published_date: str,
    *,
    filename: str | None = None,
    frontmatter_slug: str | None = None,
    missing_field: str | None = None,
    body: str = PRESERVED_BODY,
) -> Path:
    """Write one source article fixture."""
    article_slug = frontmatter_slug if frontmatter_slug is not None else slug
    frontmatter_values = {
        "title": f"Title {article_slug}",
        "subtitle": f"Subtitle {article_slug}",
        "slug": article_slug,
        "author": "The Artificial Intelligence Times",
        "date": published_date,
        "dateline": "SAN FRANCISCO",
        "summary": f"Summary for {article_slug}.",
        "image": "https://example.com/image.jpg",
    }

    lines = ["---"]
    for field_name in ("title", "subtitle", "slug", "author", "date", "dateline", "summary", "image"):
        if missing_field == field_name:
            continue
        lines.append(f"{field_name}: {frontmatter_values[field_name]}")

    if missing_field != "tags":
        lines.extend(["tags:", "  - AI Engineering"])

    lines.extend(["---", "", body])

    source_dir.mkdir(parents=True, exist_ok=True)
    file_path = source_dir / (filename if filename is not None else f"{slug}.md")
    file_path.write_text("\n".join(lines), encoding="utf-8")
    return file_path


def write_articles(source_dir: Path, count: int) -> list[str]:
    """Write date-descending source article fixtures."""
    base_date = dt.date(2026, 6, 20)
    slugs: list[str] = []

    for index in range(count):
        slug = f"article-{index:02d}"
        slugs.append(slug)
        published_date = (base_date - dt.timedelta(days=index)).isoformat()
        write_article(source_dir, slug, published_date)

    return slugs


def generated_article_dir(project_root: Path) -> Path:
    """Return the generated Nuxt article directory for a temp project."""
    return project_root / "frontend" / "newspaper" / "content" / "articles"


def generated_layout_path(project_root: Path) -> Path:
    """Return the generated Nuxt layout path for a temp project."""
    return project_root / "frontend" / "newspaper" / "content" / "layout.json"


def test_valid_source_folder_with_21_articles_generates_nuxt_articles(tmp_path: Path) -> None:
    """Valid source articles are copied into generated Nuxt content."""
    source_dir = tmp_path / "data" / "input" / "newspaper" / "articles"
    project_root = tmp_path / "repo"
    source_slugs = write_articles(source_dir, selected_article_count())

    result = compile_newspaper_articles(source_dir=source_dir, project_root=project_root)

    output_dir = generated_article_dir(project_root)
    generated_files = sorted(output_dir.glob("*.md"))
    assert len(generated_files) == selected_article_count()
    assert {path.name for path in generated_files} == {f"{slug}.md" for slug in source_slugs}
    assert result.selected_slugs == tuple(source_slugs)

    generated_content = (output_dir / f"{source_slugs[0]}.md").read_text(encoding="utf-8")
    assert "## Preserved Heading" in generated_content
    assert "![Diagram](https://example.com/diagram.jpg)" in generated_content
    assert "[reference link](https://example.com/source)" in generated_content
    assert "```python" in generated_content


def test_layout_json_uses_deterministic_slug_assignment(tmp_path: Path) -> None:
    """Layout uses date descending and slug ascending for tied dates."""
    source_dir = tmp_path / "data" / "input" / "newspaper" / "articles"
    project_root = tmp_path / "repo"
    article_specs = [("zeta-story", "2026-06-20"), ("alpha-story", "2026-06-20")]
    base_date = dt.date(2026, 6, 19)
    article_specs.extend((f"article-{index:02d}", (base_date - dt.timedelta(days=index)).isoformat()) for index in range(20))

    for slug, published_date in article_specs:
        write_article(source_dir, slug, published_date)

    compile_newspaper_articles(source_dir=source_dir, project_root=project_root)

    expected_slugs = ["alpha-story", "zeta-story", *[f"article-{index:02d}" for index in range(19)]]
    layout = json.loads(generated_layout_path(project_root).read_text(encoding="utf-8"))
    assert layout == {
        "hero": expected_slugs[0],
        "featured": expected_slugs[1:6],
        "secondary": expected_slugs[6],
        "sidebar": [
            expected_slugs[7:10],
            expected_slugs[10:13],
        ],
        "briefs": [
            expected_slugs[13:15],
            expected_slugs[15:17],
            expected_slugs[17:19],
            expected_slugs[19:21],
        ],
    }
    assert not (generated_article_dir(project_root) / "article-19.md").exists()


def test_fewer_than_21_articles_fails(tmp_path: Path) -> None:
    """Compilation fails when too few source Markdown files exist."""
    source_dir = tmp_path / "data" / "input" / "newspaper" / "articles"
    project_root = tmp_path / "repo"
    write_articles(source_dir, selected_article_count() - 1)

    with pytest.raises(NewspaperArticleValidationError, match="at least 21 .md files") as exc_info:
        compile_newspaper_articles(source_dir=source_dir, project_root=project_root)

    assert str(source_dir) in str(exc_info.value)


def test_missing_required_frontmatter_field_fails(tmp_path: Path) -> None:
    """Compilation fails when required frontmatter is missing."""
    source_dir = tmp_path / "data" / "input" / "newspaper" / "articles"
    project_root = tmp_path / "repo"
    write_articles(source_dir, selected_article_count() - 1)
    failing_path = write_article(source_dir, "missing-field", "2026-05-01", missing_field="subtitle")

    with pytest.raises(NewspaperArticleValidationError, match="subtitle") as exc_info:
        compile_newspaper_articles(source_dir=source_dir, project_root=project_root)

    assert str(failing_path) in str(exc_info.value)


def test_filename_slug_mismatch_fails(tmp_path: Path) -> None:
    """Compilation fails when filename stem and frontmatter slug differ."""
    source_dir = tmp_path / "data" / "input" / "newspaper" / "articles"
    project_root = tmp_path / "repo"
    write_articles(source_dir, selected_article_count() - 1)
    failing_path = write_article(source_dir, "actual-slug", "2026-05-01", filename="wrong-name.md")

    with pytest.raises(NewspaperArticleValidationError, match="filename stem equals frontmatter slug") as exc_info:
        compile_newspaper_articles(source_dir=source_dir, project_root=project_root)

    assert str(failing_path) in str(exc_info.value)


def test_duplicate_slug_fails(tmp_path: Path) -> None:
    """Compilation reports duplicate frontmatter slugs distinctly."""
    source_dir = tmp_path / "data" / "input" / "newspaper" / "articles"
    project_root = tmp_path / "repo"
    write_articles(source_dir, selected_article_count() - 2)
    write_article(source_dir, "duplicate-source-a", "2026-05-01", frontmatter_slug="duplicate-slug")
    failing_path = write_article(source_dir, "duplicate-source-b", "2026-05-02", frontmatter_slug="duplicate-slug")

    with pytest.raises(NewspaperArticleValidationError, match="duplicate slug") as exc_info:
        compile_newspaper_articles(source_dir=source_dir, project_root=project_root)

    assert str(failing_path) in str(exc_info.value)


def test_invalid_date_fails(tmp_path: Path) -> None:
    """Compilation fails when date is not YYYY-MM-DD."""
    source_dir = tmp_path / "data" / "input" / "newspaper" / "articles"
    project_root = tmp_path / "repo"
    write_articles(source_dir, selected_article_count() - 1)
    failing_path = write_article(source_dir, "invalid-date", "06/20/2026")

    with pytest.raises(NewspaperArticleValidationError, match="date parses as YYYY-MM-DD") as exc_info:
        compile_newspaper_articles(source_dir=source_dir, project_root=project_root)

    assert str(failing_path) in str(exc_info.value)


def test_bodyless_article_fails(tmp_path: Path) -> None:
    """Compilation fails when an article has no body after frontmatter."""
    source_dir = tmp_path / "data" / "input" / "newspaper" / "articles"
    project_root = tmp_path / "repo"
    write_articles(source_dir, selected_article_count() - 1)
    failing_path = write_article(source_dir, "empty-body", "2026-05-01", body="   \n")

    with pytest.raises(NewspaperArticleValidationError, match="body exists") as exc_info:
        compile_newspaper_articles(source_dir=source_dir, project_root=project_root)

    assert str(failing_path) in str(exc_info.value)


def test_stale_generated_markdown_files_are_removed_before_writing(tmp_path: Path) -> None:
    """Compilation removes old generated Markdown without touching non-Markdown files."""
    source_dir = tmp_path / "data" / "input" / "newspaper" / "articles"
    project_root = tmp_path / "repo"
    write_articles(source_dir, selected_article_count())

    output_dir = generated_article_dir(project_root)
    output_dir.mkdir(parents=True)
    stale_article = output_dir / "stale-article.md"
    retained_file = output_dir / "keep.txt"
    stale_article.write_text("stale", encoding="utf-8")
    retained_file.write_text("keep", encoding="utf-8")

    compile_newspaper_articles(source_dir=source_dir, project_root=project_root)

    assert not stale_article.exists()
    assert retained_file.exists()
    assert len(list(output_dir.glob("*.md"))) == selected_article_count()
