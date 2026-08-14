# Internals: add a lowering

Use this workflow for a new documented primitive:

1. Choose the canonical name and category in `ivy/transpiler/registry.py`.
2. Add the source aliases and the official framework documentation link.
3. Implement target behavior in `ivy/transpiler/runtime.py`, preserving dtype,
   axis, shape, and error semantics.
4. Add a source-defined function to `tests/transpiler/test_api.py` or a focused
   test module.
5. Add one user-facing tutorial or API note when the behavior is non-obvious.
6. Run `uv run --python 3.13 pytest tests` and `uv run --python 3.13 --group docs mkdocs build --strict`.

The registry is not a substitute for tests: a listed primitive is incomplete
until its target behavior and edge cases are covered. Keep conversions explicit
when a framework has no equivalent; do not silently route through NumPy for a
JAX or accelerator target.

## Design the canonical contract

Do not begin with one framework's signature and mechanically expose it to every
target. Identify the shared meaning and list intentional differences. Record
the canonical name, category, source aliases, target support, and direct
official documentation URLs.

Check at least:

- positional versus keyword-only arguments;
- `axis`/`dim` and `keepdims`/`keepdim` spelling;
- source weight layout and broadcasting;
- default dtype and integer promotion;
- empty/scalar behavior;
- random key/seed semantics;
- training versus inference behavior;
- gradient or stop-gradient expectations.

## Test in layers

Add a direct `runtime.call` test for target branches and a file-defined source
function for AST lowering. Parameterize all maintained targets when semantics
are shared. Assert native output types as well as values.

An unsupported target is preferable to a plausible but wrong approximation.
Raise `UnsupportedPrimitiveError` with an actionable boundary until a correct
and tested mapping exists.

## Review checklist

- registry revision updated if the public contract changes;
- official links resolve to current public APIs;
- generated report lists the canonical primitive;
- emitted source remains source-framework independent;
- CPU Python 3.12 and 3.13 suites pass;
- strict Pages build includes the new behavior;
- user guide documents non-obvious semantics and limitations.
