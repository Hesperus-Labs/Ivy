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

For a framework callable:

```python
import ivy

equinox_fn = ivy.transpile(source_fn, source="torch", target="equinox")
```

Random operations must receive `key=` explicitly. A stateful adapter returns
`(module, state)` and a stateful call returns `(output, new_state)`. Use
`eqx.nn.inference_mode(module, value=True)` for evaluation and
`eqx.tree_serialise_leaves` for checkpoints; Hesperus Ivy's
`ivy.save_equinox` also writes a small adjacent JSON format manifest.

See the [Equinox API reference](../reference/equinox.md) for the bridge
functions and the [internal state model](../internals/pipeline.md#explicit-state).
