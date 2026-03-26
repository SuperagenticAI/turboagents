# Contributing

Thanks for contributing to `turboagents`.

## Development Setup

This repository uses `uv` as the default Python workflow.

```bash
uv sync
```

Useful variants:

```bash
uv sync --extra rag
uv sync --extra mlx
uv sync --extra vllm
uv sync --all-extras
```

## Common Commands

Run the test suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest -q
```

Build the package:

```bash
uv build
```

Serve the docs locally:

```bash
uv run mkdocs serve
```

Build the docs:

```bash
uv run mkdocs build
```

## Pull Requests

- Keep changes focused and scoped.
- Add or update tests when behavior changes.
- Update docs and changelog entries when user-facing behavior changes.
- Do not include unrelated refactors in the same pull request.

## Release Process

- Update the version in `pyproject.toml`.
- Update `CHANGELOG.md`.
- Build locally with `uv build`.
- Publish explicitly with `uv publish` when you are ready.

Publishing is a manual action. This repository does not auto-publish to PyPI.
