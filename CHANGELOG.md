# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows semantic versioning where practical for published releases.

## [Unreleased]

## [0.1.0a3] - 2026-06-17

### Fixed

- `InMemoryTurboIndex` no longer raises `AttributeError` when instantiated
  directly: the dataclass dropped `slots=True` so `__post_init__` can attach the
  derived `Config`. RAG adapters were unaffected, but the base index is now
  usable on its own.

### Changed

- `Config.head_dim` now accepts any positive integer instead of only
  `{64, 128, 256}`. Vectors are zero-padded to the next power of two
  (`Config.transform_dim`) for the Walsh-Hadamard rotation and cropped back on
  dequantize, so common embedding sizes such as 384, 768 and 1536 work without
  the caller padding or truncating. Power-of-two dimensions are unchanged.

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
