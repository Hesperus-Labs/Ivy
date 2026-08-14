# Verify numerical parity

A successful conversion means the callable was generated, not that every edge
case is equivalent. Build a small differential test for each production
conversion.

## 1. Separate reference and target execution

```python
import numpy as np
import ivy


def torch_kernel(x, weight, bias):
    import torch

    return torch.relu(torch.matmul(x, weight) + bias)


target_kernel = ivy.transpile(
    torch_kernel,
    source="torch",
    target="jax",
)
```

Run the source with source-native tensors and the target with target-native
arrays. Convert only final results to NumPy for comparison.

## 2. Use discriminating inputs

```python
x = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], dtype=np.float32)
weight = np.array(
    [[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]],
    dtype=np.float32,
)
bias = np.array([0.1, -0.2], dtype=np.float32)
```

Avoid all-ones inputs and square symmetric matrices as the only case. They can
hide transpose, broadcasting, and axis bugs.

## 3. Compare the full contract

```python
import jax.numpy as jnp
import torch

expected = torch_kernel(
    torch.from_numpy(x),
    torch.from_numpy(weight),
    torch.from_numpy(bias),
).numpy()
target_result = target_kernel(jnp.asarray(x), jnp.asarray(weight), jnp.asarray(bias))
actual = np.asarray(target_result)

assert actual.shape == expected.shape
assert actual.dtype == expected.dtype
np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
```

Also assert native type and, when relevant, target device. Choose tolerances
based on dtype and numerical stability rather than one global constant.

## 4. Cover semantic edges

For each used primitive, select relevant cases:

- positive and negative axes;
- scalar, empty, singleton, and batched dimensions;
- broadcasting and non-contiguous source views;
- integer, boolean, float32, and supported lower-precision dtypes;
- clipping boundaries and division by zero;
- padding order and split remainder behavior;
- training/inference branches;
- deterministic reuse and key splitting for random operations.

## 5. Compare eager before compiled

Test these layers independently:

1. source eager against target eager;
2. target eager against target compiled;
3. module wrapper eager against function conversion;
4. checkpoint-restored module against the pre-save module.

When only compiled behavior differs, the likely cause is target tracing/static
argument behavior rather than AST lowering.

## 6. Test reports

```python
target_kernel, report = ivy.transpile(
    torch_kernel,
    source="torch",
    target="jax",
    return_report=True,
    cache=False,
)

assert report.source == "torch"
assert report.target == "jax"
assert {"matmul", "relu"} <= set(report.converted_primitives)
```

Report assertions detect accidental loss of a lowering even when simple inputs
happen to produce the right result through ordinary Python arithmetic.

## 7. Use the maintained test matrix

The repository's focused suite covers public backends, cross-framework
functions, random keys, module parameters, serialization, CLI JSON, cache
behavior, errors, and runtime primitives:

```bash
uv sync --python 3.13 --extra all-cpu
uv run --python 3.13 pytest tests -q
```

Application tests remain necessary because the registry cannot know your
shapes, layouts, tolerances, or state transitions.
