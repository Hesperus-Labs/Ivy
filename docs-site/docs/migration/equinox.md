# Migrate from Flax or Haiku to Equinox

Hesperus Ivy does not use Flax NNX or Haiku internally. The JAX target is an
Equinox `eqx.Module` and mutable state is an explicit value.

| Old concept | Equinox contract |
| --- | --- |
| Flax variables collection | `(module, state)` when state exists |
| implicit PRNG stream | required `key=` argument and deterministic splits |
| `train`/`eval` global mode | `eqx.nn.inference_mode(module, value=...)` |
| optimizer-owned parameters | PyTree leaves + `eqx.apply_updates`/Optax |
| framework checkpoint object | `eqx.tree_serialise_leaves` + JSON manifest |

Start with `ivy.to_equinox_module` for a callable or simple module. If a model
uses source-specific parameter layouts or mutation that cannot be represented
as a value update, the converter raises a structured report so the adapter can
be made explicit.

## Model and training migration

Define trainable arrays as `eqx.Module` fields. Use typed `jax.random.key`
values and split for independent random calls. Use
`eqx.filter_value_and_grad`, Optax, and `eqx.apply_updates` so the model,
optimizer, random key, and explicit layer state are values passed through the
training step.

Serialize leaves with a matching trusted model skeleton. Preserve architecture
hyperparameters in code/configuration; the Hesperus JSON sidecar records format
and version but does not pickle architecture.

Convert numerical forward kernels and straightforward parameters. Rewrite
collection mutation, lifted transforms, partition rules, and orchestration
directly in JAX/Equinox instead of emulating another framework inside a PyTree.

Follow the [worked module migration](../tutorials/pytorch-to-equinox.md) and
[randomness/state guide](../guide/randomness-and-state.md).
