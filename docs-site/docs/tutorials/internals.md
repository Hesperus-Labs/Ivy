# Internal tutorial: follow one conversion end to end

This tutorial is for contributors who want to understand a conversion before
adding a primitive. It follows one function through inspection, lowering,
runtime dispatch, caching, and an Equinox wrapper.

## 1. Start with a framework-shaped function

Keep the example in a file so `inspect.getsource` can recover it:

```python
def model_step(x, weight, bias):
    import torch

    hidden = torch.matmul(x, weight) + bias
    return torch.relu(hidden)
```

`ivy.transpile` records `source="torch"` and `target="jax"`; it does not run
the source framework while converting.

## 2. Inspect the generated contract

```python
import ivy

converted, report = ivy.transpile(
    model_step,
    source="torch",
    target="jax",
    emit="build/generated",
    return_report=True,
)
print(report.converted_primitives)
print(report.emitted_path)
```

The generated function replaces framework calls with
`ivy.transpiler.runtime.call`. Imports of the source framework are removed;
the runtime imports only JAX when the generated callable is selected.

## 3. Follow a primitive

For `torch.matmul`, the AST lowerer records the canonical name `matmul` and
the runtime selects `jax.numpy.matmul`. For `torch.nn.functional.linear`, the
runtime applies the source framework's weight layout (`input @ weight.T`) so a
PyTorch checkpoint has the same forward behavior in JAX.

The registry is the audit trail:

```python
manifest = ivy.compatibility_report(source="torch", target="equinox")
for primitive in manifest["primitives"]:
    print(primitive["name"], primitive["framework_docs"]["jax"])
```

Every listed primitive needs a runtime implementation, differential test, and
official framework links before it is considered part of the stable contract.

## 4. Make state and randomness explicit

The JAX/Equinox target does not create a hidden mutable state store. Convert a
module with `ivy.to_equinox_module`, pass `state=` when the module is stateful,
and pass a JAX key to random operations:

```python
module, state = ivy.to_equinox_module(native_module, source="torch", state={})
output, state = module(x, state=state)
```

This value-oriented boundary composes with `eqx.filter_jit`, gradients, and
checkpointing.

## 5. Cache safely and diagnose failures

Generated source and its JSON metadata live in the user cache, keyed by source,
framework pair, and conversion options. The cache never unpickles a model or
writes into the repository. Use `ivy cache-info`, `ivy cache-clear`, and
`TranspileReport.to_json()` when reproducing a conversion issue.

When a call is outside the registry, the converter raises an explicit error.
Add the lowering in `ivy/transpiler/registry.py` and `runtime.py`, then add a
cross-framework test and rebuild the strict Pages site:

```bash
uv run pytest tests -q
uv run --group docs mkdocs build --strict
```

The [pipeline internals](../internals/pipeline.md) page contains the compact
architecture reference; this page is the executable tour for new contributors.
