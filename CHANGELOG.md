# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows semantic versioning where practical for published releases.

## [Unreleased]

## [0.1.0a2] - 2026-03-28

### Added

- Standardized `uv`-first developer and CI workflows.
- Added OSS community-health files and GitHub issue / pull request templates.
- Added a Chroma-backed TurboRAG adapter and local smoke coverage against
  `chromadb 1.5.5`.
- Documented how TurboAgents fits beside Chroma `Context-1` as a compressed
  retrieval and rerank layer rather than a replacement for Chroma's search
  planner.

## [0.1.0a1] - 2026-03-27

### Fixed

- LanceDB persisted-table search now falls back cleanly when in-memory rerank
  state is unavailable after reopen.

## [0.1.0a0] - 2026-03-26

### Added

- Initial public alpha release of `turboagents`.
