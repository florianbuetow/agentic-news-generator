# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- File progress counter for audio extraction and video integrity checking
- YouTube video download pipeline with configurable channels and format options
- Configuration system with YAML loading, validation, and template support
- AI-powered fake test detector for identifying placeholder tests (#2)
- Nuxt-based newspaper frontend for browsing generated content (#3)
- Topic segmentation and transcript processing pipeline (#4, #6)
- Multi-language transcription with automatic English translation (#7)
- FastText-based language detection and transcript quality analysis (#8)
- Embedding-based hierarchical topic detection with taxonomy support (#9)
- Shell script environment variable and argument violation detector
- Automated model benchmark runner with performance visualizations
- Hallucination analysis notebook and shellscript analyzer
- Progress tracking with ETA display for transcription and topic extraction
- Unified pipeline status display with storage size information
- Video integrity checking step in download/transcribe pipeline
- Script to find and remove empty data files
- All-ingestion pipeline target for running without topic detection
- Topic tree visualization, MiniRAG export, and LM Studio status scripts
- Comprehensive CI targets and development tooling in justfile
- Configurable minimum video duration filter for transcription

### Changed

- Rewrote video integrity checker from bash to Python with result caching
- Replaced bash transcription orchestrator with Python implementation
- Replaced simplified SRT timestamp format with plain text output
- Redesigned pipeline status display as a unified table layout
- Improved English language detection using majority vote
- Newspaper dev server port changed to 12000 with config-driven validation

### Removed

- Legacy bash transcription implementation

### Fixed

- Filtered out near-zero duration speech segments during audio extraction
- Escaped single quotes in ffmpeg concat file paths
- Extracted speech segments individually to avoid aselect memory errors
- Video integrity check now runs before audio extraction in all pipelines
- Real-time yt-dlp output monitoring for expired cookie detection
- Aborted yt-dlp download immediately on expired cookie or bot detection
- Zero hallucinations correctly treated as success in digest script
- Single combined download format to avoid orphaned video files
- Non-alphabetic word filtering in language detection
- Path construction in hallucination analysis notebook
- Improved error message when LM Studio has no models loaded

### Security

- Pinned python-multipart>=0.0.22 to fix GHSA-wp53-j4wj-2cfg
- Upgraded pip 25.3 to 26.0 for security fix
- Upgraded vulnerable dependencies and suppressed yt-dlp format warning

[Unreleased]: https://github.com/florianbuetow/agentic-news-generator/commits/main
