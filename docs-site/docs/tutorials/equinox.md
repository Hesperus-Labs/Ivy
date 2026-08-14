# Equinox modules, state, and random keys

Equinox modules are PyTrees. Parameters are explicit leaves, while static
configuration is marked with `eqx.field(static=True)`. Hesperus Ivy preserves
that split when it can inspect a module and never invents hidden global state.

```python
import equinox as eqx
import jax
import jax.numpy as jnp


class Scale(eqx.Module):
    weight: jax.Array

    def __call__(self, x):
        return x * self.weight


model = Scale(jnp.array(2.0))
value, gradient = eqx.filter_value_and_grad(
    lambda m: jnp.square(m(jnp.array(3.0)) - 6.0)
)(model)
```

Equinox modules are ordinary JAX PyTrees. Array parameters participate in
transformations without a separate variable-collection abstraction.

For a framework callable:

```python
import ivy

equinox_fn = ivy.transpile(source_fn, source="torch", target="equinox")
```

That target returns a JAX-executing callable. To guarantee an `eqx.Module`:

```python
model = ivy.to_equinox_module(native_module, source="torch")
assert isinstance(model, eqx.Module)
```

Random operations must receive `key=` explicitly. A stateful adapter returns
`(module, state)` and a stateful call returns `(output, new_state)`. Use
`eqx.nn.inference_mode(module, value=True)` for evaluation and
`eqx.tree_serialise_leaves` for checkpoints; Hesperus Ivy's
`ivy.save_equinox` also writes a small adjacent JSON format manifest.

## Stateful calls

```python
model, state = ivy.to_equinox_module(
    native_module,
    source="torch",
    state={"step": 0},
)
output, state = model(inputs, state=state)
```

The generic wrapper preserves state but does not infer arbitrary source
mutation. Implement real transitions explicitly.

## Filtered transformations

```python
@eqx.filter_jit
@eqx.filter_value_and_grad
def loss(model, x, y):
    return jnp.mean((model(x) - y) ** 2)
```

Filtered transformations trace array leaves and keep other leaves static.

## Save and restore

```python
ivy.save_equinox(model, "model.eqx")
restored = ivy.load_equinox("model.eqx", like=model)
```

Loading needs a matching PyTree structure. Keep architecture hyperparameters
in trusted code/configuration rather than a general object pickle.

See the [Equinox API reference](../reference/equinox.md) for the bridge
functions and the [internal state model](../internals/pipeline.md#explicit-state).

Continue with the complete [PyTorch-to-Equinox
migration](pytorch-to-equinox.md).
