# Internals: add a lowering

Use this workflow for a new documented primitive:

1. Choose the canonical name and category in `ivy/transpiler/registry.py`.
2. Add the source aliases and the official framework documentation link.
3. Implement target behavior in `ivy/transpiler/runtime.py`, preserving dtype,
   axis, shape, and error semantics.
4. Add a source-defined function to `tests/transpiler/test_api.py` or a focused
   test module.
5. Add one user-facing tutorial or API note when the behavior is non-obvious.
6. Run `uv run pytest tests` and `uv run --group docs mkdocs build --strict`.

The registry is not a substitute for tests: a listed primitive is incomplete
until its target behavior and edge cases are covered. Keep conversions explicit
when a framework has no equivalent; do not silently route through NumPy for a
JAX or accelerator target.
