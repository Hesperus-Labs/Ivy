# Contributing to Hesperus Ivy

The contributor documentation is published at
[hesperus-labs.github.io/Ivy](https://hesperus-labs.github.io/Ivy/), including
the [documentation workflow](https://hesperus-labs.github.io/Ivy/contributing/docs/)
and the [internal pipeline tour](https://hesperus-labs.github.io/Ivy/internals/pipeline/).

## Development setup

```bash
git clone https://github.com/Hesperus-Labs/Ivy.git
cd Ivy
uv sync --python 3.13
uv run --python 3.13 pytest tests
uv run --python 3.13 ruff check ivy/transpiler ivy/stateful/equinox.py ivy/cli.py tests
uv run --python 3.13 pyright
uv run --python 3.13 --group docs mkdocs build --strict
uv run --python 3.13 pre-commit run --all-files
```

Use `uv sync --extra all-cpu` for the complete CPU framework matrix. The
personalized NVIDIA environment is `uv sync --extra nvidia`; it resolves
PyTorch from the CUDA 13 index and JAX with its CUDA 13 plugin.

## Adding a primitive

Add the operation to `ivy/transpiler/registry.py`, implement its target-native
behavior in `ivy/transpiler/runtime.py`, add differential tests, and document
the semantics. A primitive is not complete until the strict Pages build and the
framework tests pass.

## Pull requests

Keep changes focused, include a regression test, and include a documentation
update for every public behavior change. Do not add install-time downloads,
private framework APIs, process-global state, or binary compiler artifacts.
Preserve upstream Ivy Apache-2.0 attribution in derived code.
