# Five-minute quickstart

This tutorial converts a small PyTorch-shaped function into a NumPy callable.
The same source contract works with the TensorFlow, JAX, and Equinox targets.

Save this as `quickstart.py`; the converter must recover a file-defined
function. An interactive lambda is not equivalent.

```python
import numpy as np
import ivy


def source_fn(x, weight, bias):
    import torch

    return torch.relu(torch.matmul(x, weight) + bias)


target_fn, report = ivy.transpile(
    source_fn,
    source="torch",
    target="numpy",
    return_report=True,
)

x = np.ones((2, 3), dtype=np.float32)
w = np.eye(3, dtype=np.float32)
b = np.zeros(3, dtype=np.float32)
print(target_fn(x, w, b))
print(report.to_json())
```

The result is a real `numpy.ndarray`; Ivy is not on the hot execution path.
The report records the source and target, converted primitive names, framework
versions, and whether generated source came from the cache.

The expected result is a 2×3 matrix of ones. The first report should identify
`source="torch"`, `target="numpy"`, `matmul` and `relu`, with `cache_hit=false`.

## Convert to JAX

```python
import jax.numpy as jnp

jax_fn = ivy.transpile(source_fn, source="torch", target="jax")
result = jax_fn(
    jnp.ones((2, 3), dtype=jnp.float32),
    jnp.eye(3, dtype=jnp.float32),
    jnp.zeros(3, dtype=jnp.float32),
)
print(type(result), result)
```

The result is a `jax.Array`; the source framework is not on the execution path.

## Emit readable source

```python
target_fn = ivy.transpile(
    source_fn,
    source="torch",
    target="numpy",
    emit="build/generated",
)
```

The directory receives a Python file and a JSON manifest. Emission is opt-in;
normal conversion keeps generated source in the user cache.

Review both files. The Python should contain runtime calls for `matmul` and
`relu`; the JSON should list the same primitives.

## Compile after parity

```python
compiled = ivy.transpile(
    source_fn,
    source="torch",
    target="jax",
    backend_compile=True,
)
```

This applies `jax.jit`. Test eager behavior first so a tracing failure is not
mistaken for a lowering failure.

## Next steps

- [Transpiling](transpiling.md) explains options and errors.
- [Parity testing](parity.md) builds a meaningful regression test.
- [PyTorch to Equinox](pytorch-to-equinox.md) migrates a module.
- [Source requirements](../guide/source-requirements.md) defines the boundary.
