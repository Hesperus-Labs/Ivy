# Five-minute quickstart

This tutorial converts a small PyTorch-shaped function into a NumPy callable.
The same source contract works with the TensorFlow, JAX, and Equinox targets.

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
