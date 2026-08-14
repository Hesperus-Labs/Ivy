#!/usr/bin/env bash
set -euo pipefail

# The old requirements/*.txt bootstrap was tied to the upstream Ivy matrix.
# uv now resolves the maintained Python 3.12/3.13 environment from the lock.
uv sync --python "${IVY_PYTHON:-3.13}" --extra "${IVY_EXTRA:-all-cpu}"
