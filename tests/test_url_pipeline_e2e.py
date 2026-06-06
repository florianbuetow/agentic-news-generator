"""End-to-end URL pipeline tests with a real local HTTP server."""

from pathlib import Path

import tiktoken

from src.config import LLMConfig
from src.url_ingestion.clean_content_pipeline import CleaningErrorLog, UrlCleanContentPipeline
from src.url_ingestion.download_pipeline import InboxArchive, UrlDownloadPipeline
from src.url_ingestion.downloader import DownloaderFactory
from src.url_ingestion.formatting import FormattingAgent
from src.url_ingestion.identity import sanitize_normalized_url_to_stem
from src.url_ingestion.raw_processing import (
    HtmlRawProcessor,
    HtmlTextExtractor,
    PdfRawProcessor,
    PlainTextRawProcessor,
    PypdfTextExtractor,
    RawContentScanner,
    RawProcessorFactory,
)
from tests.test_url_download_pipeline import fixed_today, read_queue, serve_directory, write_config
from tests.test_url_formatting import make_llm_config


class FixtureFormattingClient:
    """Deterministic formatter for expected-file comparisons."""

    def complete(self, prompt: str, llm: LLMConfig) -> str:
        """Return the extracted source text unchanged."""
        return prompt


def make_text_pdf_bytes(text: str) -> bytes:
    """Create a small text PDF fixture that pypdf can extract."""
    content = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET\n".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"endstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode())
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Root 1 0 R /Size {len(objects) + 1} >>\nstartxref\n{xref_offset}\n%%EOF\n".encode())
    return bytes(pdf)


def write_http_fixture(server_root: Path) -> None:
    """Write source files served by the local HTTP server."""
    server_root.mkdir()
    (server_root / "report.pdf").write_bytes(make_text_pdf_bytes("Fixture PDF Clean Text"))
    (server_root / "article.html").write_text(
        """
        <!doctype html>
        <html>
          <head><title>Local article</title></head>
          <body>
            <main id="article">Initial article text.</main>
            <script>
              document.getElementById("article").insertAdjacentHTML("beforeend", "<p>Rendered by JavaScript.</p>");
            </script>
          </body>
        </html>
        """,
        encoding="utf-8",
    )
    (server_root / "notes.md").write_text(
        "# Markdown fixture\n\nThis is served and cleaned.\n",
        encoding="utf-8",
    )


def write_expected_cleaned_outputs(expected_root: Path, base_url: str) -> dict[str, Path]:
    """Write expected cleaned Markdown files for downloaded local-server fixtures."""
    pdf_stem = sanitize_normalized_url_to_stem(f"{base_url}/report.pdf")
    html_stem = sanitize_normalized_url_to_stem(f"{base_url}/article.html")
    query_html_stem = sanitize_normalized_url_to_stem(f"{base_url}/article.html?utm_source=test")
    markdown_stem = sanitize_normalized_url_to_stem(f"{base_url}/notes.md")
    pdf_expected = expected_root / "pdf" / f"{pdf_stem}.md"
    html_expected = expected_root / "html" / f"{html_stem}.md"
    query_html_expected = expected_root / "html" / f"{query_html_stem}.md"
    markdown_expected = expected_root / "markdown" / f"{markdown_stem}.md"
    pdf_expected.parent.mkdir(parents=True)
    html_expected.parent.mkdir(parents=True)
    markdown_expected.parent.mkdir(parents=True)
    pdf_expected.write_text("Fixture PDF Clean Text", encoding="utf-8")
    html_expected.write_text("Initial article text.\n\nRendered by JavaScript.", encoding="utf-8")
    query_html_expected.write_text("Initial article text.\n\nRendered by JavaScript.", encoding="utf-8")
    markdown_expected.write_text("# Markdown fixture\n\nThis is served and cleaned.", encoding="utf-8")
    return {"pdf": pdf_expected, "html": html_expected, "query_html": query_html_expected, "markdown": markdown_expected}


def make_clean_pipeline(config_path: Path, data_dir: Path) -> UrlCleanContentPipeline:
    """Create the real clean-content pipeline with deterministic fixture formatting."""
    config = write_config(config_path, data_dir)
    formatting_agent = FormattingAgent(
        llm=make_llm_config(),
        prompt_template="{source_text}",
        encoder=tiktoken.get_encoding("o200k_base"),
        skip_threshold_pct=80,
        llm_client=FixtureFormattingClient(),
    )
    processor_factory = RawProcessorFactory(
        HtmlRawProcessor(HtmlTextExtractor(), formatting_agent),
        PdfRawProcessor(PypdfTextExtractor(), formatting_agent),
        PlainTextRawProcessor(formatting_agent),
    )
    return UrlCleanContentPipeline(RawContentScanner(config), processor_factory, CleaningErrorLog(config, fixed_today))


def test_url_pipeline_downloads_from_local_http_server_and_matches_expected_cleaned_outputs(tmp_path: Path) -> None:
    """Run local HTTP download and clean downloaded raw files against expected outputs."""
    server_root = tmp_path / "server"
    expected_root = tmp_path / "expected_cleaned"
    data_dir = tmp_path / "data"
    config_path = tmp_path / "config.yaml"
    write_http_fixture(server_root)
    config = write_config(config_path, data_dir)
    inbox_dir = config.get_url_inbox_dir()
    inbox_dir.mkdir(parents=True)

    with serve_directory(server_root) as base_url:
        expected_outputs = write_expected_cleaned_outputs(expected_root, base_url)
        inbox_file = inbox_dir / "links.txt"
        inbox_file.write_text(
            f"{base_url}/report.pdf\n{base_url}/article.html\n{base_url}/article.html?utm_source=test\n{base_url}/notes.md\n",
            encoding="utf-8",
        )

        download_summary = UrlDownloadPipeline(
            config,
            DownloaderFactory.default(config),
            InboxArchive(config, fixed_today),
            reachability_probe=None,
        ).run(read_queue(config))

        assert download_summary.successful_download_count == 4
        assert download_summary.failure_count == 0
        assert not inbox_file.exists()
        assert not (inbox_dir / "unprocessed" / "2026-05-31.txt").exists()

        clean_summary = make_clean_pipeline(config_path, data_dir).run(limit=None, raw_path=None, raw_paths=None, force=False)

        assert clean_summary.cleaned_count == 4
        assert clean_summary.failure_count == 0
        for expected_key, expected_path in expected_outputs.items():
            content_type = "html" if expected_key == "query_html" else expected_key
            actual_path = config.get_url_cleaned_dir() / content_type / expected_path.name
            assert actual_path.read_text(encoding="utf-8") == expected_path.read_text(encoding="utf-8")
