# Development and releases

## Environment

```bash
git clone https://github.com/Hesperus-Labs/Ivy.git
cd Ivy
uv sync --python 3.13 --extra all-cpu --group docs
```

The repository supports Python 3.12 and 3.13. Use explicit `--python` on `uv
run` so `.python-version` cannot accidentally change a CI matrix interpreter.

## Required checks

```bash
uv run --python 3.13 ruff check ivy/transpiler ivy/stateful/equinox.py ivy/cli.py tests
uv run --python 3.13 pyright
uv run --python 3.13 pytest tests --cov=ivy.transpiler --cov-report=term-missing
uv run --python 3.13 --group docs mkdocs build --strict
uv build
```

Run the examples for changed framework paths. Use the CPU extra for the full
portable matrix and the NVIDIA extra only in a separate environment.

## Change placement

| Change | Required locations |
| --- | --- |
| New primitive | registry, runtime, direct test, differential test, catalog/user note |
| New transpile option | public signature/docstring, tests, options reference, migration note |
| Equinox bridge behavior | bridge, module/state tests, Equinox guide/reference |
| Package/dependency range | pyproject, lockfile, install/status docs, CI matrix |
| Pages content | `docs-site/docs`, `mkdocs.yml` navigation, strict build |

Do not restore dependencies on private compiler binaries, caller-frame
mutation, Flax/Haiku internals, or eager imports of all target frameworks.

## Versioning

The package is `hesperus-ivy` while imports remain `ivy`. Alpha releases use
PEP 440 prerelease versions. Widen framework/Python ranges only after the
corresponding locked environment passes focused tests and examples.

The primitive `REGISTRY_REVISION` is separate from the package version. Update
it when the public lowering contract changes so caches/reports identify the
new manifest.

## Pull requests

Keep changes reviewable and include the root cause, user impact, validation,
and documentation. Generated artifacts (`site/`, local cache, wheel/sdist) are
not committed. Preserve upstream Apache-2.0 attribution in derived code.

## Release checklist

1. Run Python 3.12 and 3.13 CPU matrices.
2. Build Pages strictly and inspect navigation/search.
3. Build sdist/wheel and install the wheel in a clean environment.
4. Run CLI diagnostics and all checked-in examples.
5. Verify package metadata, version, URLs, and dependency profiles.
6. Tag the reviewed commit; Pages deploys version tags and `main`.
7. Publish release notes describing supported additions and explicit gaps.
