# Choose a workflow

Hesperus Ivy has two related surfaces. Choose one deliberately:

1. **The transpiler** converts a file-defined callable into a callable whose
   supported operations execute on a selected target.
2. **The Ivy library API** dispatches `ivy.*` calls through a backend selected
   at runtime.

The transpiler is the preferred path for migration because the target is fixed,
the generated Python can be inspected, and the conversion produces a report.
Backend dispatch is useful when an application already uses Ivy's public array
API and wants to select a native backend at process startup.

## Decision table

| Situation | Use | Why |
| --- | --- | --- |
| Move a PyTorch forward pass to JAX | `ivy.transpile(..., target="jax")` | Produces a target-native callable and conversion report |
| Wrap a simple module as a JAX PyTree | `ivy.to_equinox_module(...)` | Makes parameters PyTree leaves and keeps state explicit |
| Compare a function across frameworks | Transpile once per target | Each result has the selected target's native array type |
| Keep existing `ivy.*` application code | `ivy.set_backend(...)` | Minimal changes to code already written against Ivy |
| Produce auditable build artifacts | `emit=` plus `return_report=True` | Stores generated Python and JSON metadata |
| Convert an interactive lambda | Extract it to a `.py` file first | The current converter requires inspectable source |
| Convert distributed/data-loader code | Write an application adapter | These systems are intentionally outside the portable core |

## Function conversion

Use functions for pure numerical kernels, losses, preprocessing math, and
stateless forward passes:

```python
import ivy


def score(x, weight):
    import torch

    return torch.sigmoid(torch.matmul(x, weight))


jax_score, report = ivy.transpile(
    score,
    source="torch",
    target="jax",
    return_report=True,
)
```

Input values are not automatically converted at the function boundary. Supply
arrays appropriate for the target—or values accepted by that target's public
array API. This avoids surprising device transfers inside the generated
function.

## Module conversion

Use `ivy.to_equinox_module` for a callable or a simple module whose forward
method is source-visible. Hesperus Ivy discovers array-valued parameters,
copies them into JAX arrays, and places them in an `eqx.Module` wrapper.

The bridge is intentionally conservative. Complex parameter containers,
custom descriptors, native extension calls, and mutation-heavy forward methods
need an explicit application adapter. See [modules and
parameters](modules.md) for the exact discovery rules.

## Backend-neutral Ivy code

```python
import ivy

ivy.set_backend("jax")
try:
    x = ivy.ones((2, 3))
    y = ivy.mean(ivy.relu(x), axis=-1)
finally:
    ivy.unset_backend()
```

Backend selection changes process-local Ivy dispatch. It does not rewrite
arbitrary PyTorch or TensorFlow code. Avoid changing the global backend inside
a library call; let the application own that lifecycle.

## Recommended migration sequence

1. Isolate one file-defined numerical function.
2. Convert it to NumPy and compare values and shapes.
3. Convert it to JAX or Equinox and compare again.
4. Inspect the report and emitted source.
5. Make random keys and state explicit at the application boundary.
6. Add JIT only after eager parity is established.
7. Pin the Hesperus Ivy commit and target framework versions for deployment.

This sequence keeps semantic differences visible. The [parity
tutorial](../tutorials/parity.md) supplies a reusable test pattern.
