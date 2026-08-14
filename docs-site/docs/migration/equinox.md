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
