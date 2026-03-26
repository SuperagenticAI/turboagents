# Deployment

## Local Build

Install docs dependencies:

```bash
uv sync --no-default-groups --group docs
```

Run the local docs server:

```bash
uv run mkdocs serve
```

Build the static site:

```bash
uv run mkdocs build
```

## GitHub Pages

This repository includes a GitHub Actions workflow at:

- `.github/workflows/docs.yml`

Repository:

- `https://github.com/SuperagenticAI/turboagents`

The workflow:

1. installs the docs dependencies
2. runs `uv run mkdocs build`
3. uploads the generated `site/` directory
4. deploys it to GitHub Pages

## Requirements

- GitHub Pages must be enabled for the repository
- the Pages source should be set to GitHub Actions
- the repository should allow Pages deployments from workflows
