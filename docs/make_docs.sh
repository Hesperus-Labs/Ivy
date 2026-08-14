#!/usr/bin/env bash
set -euo pipefail

# The published documentation lives in docs-site/ and is built with the same
# uv-managed environment used by GitHub Pages.
cd "$(dirname "$0")/.."
uv sync --python 3.12 --group docs
uv run --group docs mkdocs build --strict
