# Migrate a PyTorch module to Equinox

This tutorial moves a small `torch.nn.Linear` module into an Equinox PyTree,
checks numerical parity, computes gradients, and saves native leaves.

## 1. Install the CPU matrix

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install --torch-backend=cpu \
  "hesperus-ivy[all-cpu] @ git+https://github.com/Hesperus-Labs/Ivy.git@main"
```

Pin a commit instead of `main` in an application lockfile.

## 2. Define a source-visible module

Put this code in `migration_demo.py`; file-defined source is required.

```python
import torch


class Classifier(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.nn.functional.gelu(self.linear(x))
```

For a first migration, keep forward free of hooks, custom autograd functions,
buffer mutation, and data-dependent Python control flow.

## 3. Create deterministic source weights

```python
import numpy as np
import torch

source_model = Classifier(3, 2)
with torch.no_grad():
    source_model.linear.weight.copy_(
        torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]])
    )
    source_model.linear.bias.copy_(torch.tensor([0.25, -0.25]))

inputs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
expected = source_model(torch.from_numpy(inputs)).detach().numpy()
```

Non-square weights make a mistaken transpose visible.

## 4. Convert to an Equinox module

```python
import ivy
import jax.numpy as jnp

model = ivy.to_equinox_module(source_model, source="torch")
actual = np.asarray(model(jnp.asarray(inputs)))
np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
```

The wrapper's converted parameters are JAX array leaves. Structural data and
the rewritten forward function are static Equinox fields.

## 5. Inspect conversion separately

The bridge returns the module, while a direct transpilation provides the most
detailed report/emission path:

```python
forward, report = ivy.transpile(
    source_model.forward,
    source="torch",
    target="equinox",
    emit="build/generated",
    return_report=True,
)
print(report.to_json())
```

Review `converted_primitives` and the emitted function before making the
converted module part of a larger training step.

## 6. Differentiate and compile

```python
import equinox as eqx
import jax


@eqx.filter_value_and_grad
def loss(model, x, target):
    prediction = model(x)
    return jax.numpy.mean((prediction - target) ** 2)


compiled_loss = eqx.filter_jit(loss)
value, gradients = compiled_loss(
    model,
    jnp.asarray(inputs),
    jnp.asarray(expected),
)
```

Filtered transformations handle mixed array/static PyTree leaves. Add Optax in
the normal Equinox/JAX style to update differentiable leaves.

## 7. Handle state explicitly

```python
model, state = ivy.to_equinox_module(
    source_model,
    source="torch",
    state={"step": 0},
)
output, next_state = model(jnp.asarray(inputs), state=state)
```

The generic bridge preserves the state value but does not infer arbitrary
PyTorch buffer mutation. Implement real updates explicitly.

## 8. Save and restore leaves

```python
ivy.save_equinox(model, "build/classifier.eqx")
restored = ivy.load_equinox("build/classifier.eqx", like=model)
```

The adjacent `.eqx.json` records the Hesperus format/version. The `like` model
provides the trusted PyTree structure required by Equinox.

## Migration checklist

- numerical parity on multiple shapes and non-square weights;
- dtype and native output type assertions;
- explicit keys for stochastic operations;
- explicit state transitions for running values;
- eager checks before `filter_jit`;
- emitted-source review and pinned dependency versions;
- checkpoint round trip using the same model structure.
