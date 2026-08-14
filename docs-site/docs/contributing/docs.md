# Documentation workflow

Documentation is part of the feature contract. A new public primitive or
transpiler behavior is complete only when its API page, compatibility entry,
example, and any relevant internals explanation are present.

## Local build

```bash
uv sync --python 3.12
uv run --group docs mkdocs build --strict
uv run --group docs mkdocs serve
```

The Pages workflow builds the site from `main` and from release tags. Pull
requests build the same artifact without deploying it, so navigation and
internal tutorials are checked before a release is published.

## Writing an internal tutorial

Explain the invariant before the implementation detail, show the smallest
executable snippet, and link the corresponding public API. Internal pages may
describe the AST, registry, cache, and Equinox state model, but must never ask a
user to import a private compiler binary or mutate a caller frame.
