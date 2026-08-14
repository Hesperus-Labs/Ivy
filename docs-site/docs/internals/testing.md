# Internals: testing strategy

The maintained tests live under `tests/`. The retained upstream `ivy_tests/`
tree documents historical breadth but depends on obsolete generated databases,
workflow infrastructure, and version assumptions; it is not the Hesperus 2.0
release gate.

## Maintained layers

| Layer | Purpose |
| --- | --- |
| `tests/transpiler/test_api.py` | source inspection, reports, cache, emission, errors, graph aliases |
| `tests/transpiler/test_frameworks.py` | cross-framework parity, methods, random keys, modules, Equinox |
| `tests/transpiler/test_runtime_core.py` | direct target operation semantics |
| `tests/transpiler/test_cli.py` | JSON-serializable diagnostics and CLI contract |
| `tests/library/test_backends.py` | selected-backend arrays, devices, random behavior, legacy modules |

The full CPU environment runs the focused suite on Python 3.12 and 3.13.

## Why differential tests matter

A registry entry is not evidence of semantic parity. Each new primitive needs
a source-defined function, target-native inputs, expected values from a trusted
reference, native-type assertions, and edge cases for signature differences.

Use non-symmetric data and non-square matrices. Cover dtype, axis, shape,
keep-dimension, and random determinism where applicable.

## CI contract

The modern workflow runs:

```bash
ruff check ivy/transpiler ivy/stateful/equinox.py ivy/cli.py tests
pyright
pytest tests --cov=ivy.transpiler --cov-report=term-missing
python examples/transpile_torch_to_numpy.py
python examples/transpile_tensorflow_to_numpy.py
python examples/transpile_ivy_to_numpy.py
python examples/equinox_module.py
uv build
```

It also builds documentation strictly and installs the built wheel into a clean
Python 3.12 virtual environment.

## Historical CI failure note

The old upstream `pull_request_target` matrix used retired GitHub Actions and
failed during runner setup before checking out code. Those setup failures were
not product-test results. Hesperus replaces that workflow set with the focused
quality/Pages workflows maintained in this repository.

## Adding coverage

Put a test in the narrowest layer, then add a cross-framework differential case
when behavior spans targets. A bug fix should include a regression test that
fails for the observed reason, not merely a larger happy-path example.
