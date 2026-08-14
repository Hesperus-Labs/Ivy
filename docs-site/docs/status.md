# Project status

Hesperus Ivy is an actively developed alpha release. This repository is the
source of truth for the Hesperus Labs fork; the old upstream Ivy build and
release workflows are not part of this project.

## Current release contract

| Area | Current state |
| --- | --- |
| Distribution | `hesperus-ivy` `2.0.0a1` from GitHub or a future release tag |
| Python | 3.12 and 3.13; the metadata is capped below 3.14 until the optional framework matrix is certified |
| Portable targets | NumPy, PyTorch, TensorFlow, JAX, and Equinox |
| Transpiler | Checked-in Python AST/runtime implementation; no private binary download |
| JAX state model | Equinox modules with explicit value state and explicit PRNG keys |
| Documentation | MkDocs Material, built strictly and deployed by GitHub Pages |

The maintained cross-framework contract is the primitive registry exposed by
`ivy.compatibility_report()`. It currently covers the tensor, creation,
reduction, neural-network, random, dtype, and state-boundary core. Every entry
has target behavior, a differential test, and links to the relevant official
framework documentation.

## What is intentionally outside the contract

Framework-internal symbols, private compiler hooks, distributed runtimes, data
pipelines, custom native kernels, and undocumented mutation semantics are not
silently converted. The transpiler returns a structured report or raises a
specific error so the boundary can be handled by application code.

The historical `ivy_tests/` corpus and the original Sphinx pages are retained
as upstream reference material. They are not the maintained CI or documentation
entry point. New behavior belongs in `tests/`, `docs-site/docs/`, and the
current `uv` workflows.

## Development priorities

1. Expand the registry with a lowering, official documentation links, and
   differential coverage as a unit.
2. Extend Equinox state conversions for more explicit module layouts without
   reintroducing hidden mutable state.
3. Certify additional Python/framework minor versions before widening the
   package metadata range.

See the [internal tutorial](tutorials/internals.md) for the implementation
workflow and [documentation workflow](contributing/docs.md) for the required
Pages checks.
