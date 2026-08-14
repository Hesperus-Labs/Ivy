# Core library API

Ivy's core API is backend-neutral: the same call is dispatched to NumPy,
JAX, PyTorch, or TensorFlow after `ivy.set_backend(...)`. The backend smoke
tests exercise this contract on every supported installation.

## Backend selection

```python
ivy.set_backend("jax")       # or "numpy", "torch", "tensorflow"
backend = ivy.current_backend_str()
ivy.unset_backend()
```

`set_backend` installs the selected backend's implementations into the Ivy
namespace. `unset_backend` restores the previous backend (or Ivy's neutral
namespace). Backend selection is process-local; use a context manager around
library code that must not change an application's global choice.

## Array and numerical operations

The stable core includes the following calls. Each preserves the selected
backend's native array type and supports Ivy's common keyword conventions:

| Operation | Typical use |
| --- | --- |
| `ivy.asarray` / `ivy.array` | Create an array from Python, NumPy, or native values |
| `ivy.to_numpy` | Move a result to a NumPy inspection value |
| `ivy.matmul` | Matrix multiplication and batched matrix multiplication |
| `ivy.reshape` | Shape-only view/copy operation |
| `ivy.mean`, `ivy.sum` | Reductions with `axis` and `keepdims` |
| `ivy.relu` | ReLU activation |

The [compatibility matrix](../compatibility.md) is generated from the same
primitive registry used by the transpiler, so a listed operation has a
documented lowering and differential coverage.

## Stateful modules

For trainable state, prefer the Equinox bridge documented in
[Equinox modules and state](../tutorials/equinox.md). The legacy Ivy
`Linear`, `Sequential`, and normalization modules remain available for
upstream compatibility and dispatch through the selected backend.
