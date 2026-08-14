# Documentation workflow

Documentation is part of the feature contract. A new public primitive or
transpiler behavior is complete only when its API page, compatibility entry,
example, and any relevant internals explanation are present.

## Local build

```bash
uv sync --python 3.13
uv run --python 3.13 --group docs mkdocs build --strict
uv run --python 3.13 --group docs mkdocs serve
```

The Pages workflow builds the site from `main` and from release tags. Pull
requests build the same artifact without deploying it, so navigation and
internal tutorials are checked before a release is published.

`mkdocs build --strict` treats warnings as failures. Every page must appear in
navigation, relative links must resolve, and mkdocstrings imports must work in
a docs-only environment.

## Writing an internal tutorial

Explain the invariant before the implementation detail, show the smallest
executable snippet, and link the corresponding public API. Internal pages may
describe the AST, registry, cache, and Equinox state model, but must never ask a
user to import a private compiler binary or mutate a caller frame.

## Documentation architecture

| Section | Audience | Required content |
| --- | --- | --- |
| Getting started | New user | install, first result, supported environment |
| Tutorials | Practitioner | goal-oriented runnable workflow |
| User guide/frameworks | Integrator | behavior, choices, limits, failure modes |
| Internals | Contributor | invariants, file ownership, extension process |
| API reference | API consumer | signatures, fields, errors, catalog |
| Migration | Existing user | old-to-new mapping and removed behavior |

Do not use an API dump as a substitute for explanation, and do not hand-copy a
signature that mkdocstrings can derive.

## Source of truth

Only `docs-site/docs/` feeds GitHub Pages; `mkdocs.yml` owns navigation/theme.
The old `docs/*.rst` collection is historical upstream material and must not
receive new Hesperus tutorials.

## Review checklist

- reachable through navigation and a relevant overview;
- examples use modern Python and `uv` commands;
- behavior matches tests and public APIs;
- framework claims link to current official docs;
- alpha limitations and portability boundaries are explicit;
- strict local build passes;
- live code/text is preferred over stale screenshots.
