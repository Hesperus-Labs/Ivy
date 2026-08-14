# Internals: runtime dispatch

Generated functions call `ivy.transpiler.runtime`, a lazy target dispatcher.
It is deliberately small enough to audit and uses only public framework APIs.

## Target canonicalization

The runtime maps `equinox` to JAX and `ivy` to NumPy execution. It imports one
target family on demand:

| Canonical runtime | Modules |
| --- | --- |
| NumPy | `numpy` |
| JAX | `jax.numpy`, `jax.nn`, `jax.random` |
| PyTorch | `torch`, `torch.nn.functional` |
| TensorFlow | `tensorflow`, `tf.nn` |

This lazy boundary keeps a base installation useful without every optional ML
framework.

## Signature normalization

Helpers normalize common differences before invoking target functions:

- `dim`, `axis`, and `axes` become one axis value;
- `keepdim` and `keepdims` become one boolean;
- shape arguments accept unpacked dimensions or one tuple/list;
- foreign arrays are converted to host values only at explicit bridge points;
- dtype names map to target public dtype objects.

Individual operation branches handle layout or semantic differences such as
functional linear, concatenation, clipping, embedding, one-hot encoding,
padding, activations, and random generation.

## Native result invariant

After dispatch, an operation returns an array/value from the selected target.
The runtime must not silently route a JAX/Torch/TensorFlow target operation
through NumPy just because NumPy has a convenient implementation. Host
conversion is reserved for module-parameter bridges and deterministic seed
derivation where it is explicitly documented.

## Random calls

JAX random calls use explicit keys. For non-JAX targets, `_seed_value` derives a
stable integer from key data without relying on JAX private key classes. This
preserves deterministic reuse, not cross-framework bit identity.

## Unsupported operations

The final branch raises `UnsupportedPrimitiveError`, including source operation
and target plus guidance to inspect coverage. This is a core safety property:
unsupported code never falls back to a source dependency behind the user's
back.

## Adding a runtime branch

Check official APIs for all maintained targets, normalize kwargs without
mutating values needed later, preserve dtype/shape semantics, and add direct
runtime plus source-to-target differential tests. Update the registry in the
same commit.
