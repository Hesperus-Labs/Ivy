# Troubleshooting

Start with the smallest file-defined reproducer. Diagnose conversion before
module wrapping, JIT, and accelerator placement.

```bash
python --version
ivy doctor
ivy cache-info
ivy coverage --source torch --target jax
```

Rerun with `return_report=True`, `cache=False`, and `emit=`. Compare eager
target output to a source/reference result before enabling compilation.

## Common failures

### Source is unavailable

Move a lambda or notebook/REPL function into a `.py` module and import it.

### Primitive is unsupported

Check the [catalog](../reference/primitives.md). Rewrite with covered
operations, keep the operation outside the boundary, or add a tested lowering.

### Output has the wrong native type

Confirm `target=` and construct target-native inputs. Device placement belongs
outside the generated callable.

### Dtype, axis, or shape differs

Make source dtypes explicit and test negative axes, empty dimensions, padding,
and split behavior separately. JAX 64-bit behavior depends on configuration.

### JAX tracer/concretization error

Run eagerly first. Inspect Python control flow, dynamic shapes, and static
arguments. For mixed Equinox PyTrees, use `eqx.filter_jit`.

### CUDA cannot initialize

Run `ivy doctor` outside training. Confirm driver compatibility and that CPU
and CUDA wheel indexes were not mixed in one uv resolution.

### Equinox checkpoint will not load

The `like` tree must have the same PyTree structure and compatible leaves as
the saved model. Rebuild the same architecture/inference configuration first.

## Minimal issue bundle

Include the Hesperus commit/version, registry revision, doctor JSON, explicit
source/target, synthetic reproducer, shapes/dtypes/devices, report JSON,
emitted source, and whether eager and compiled behavior differ. Never include
tokens, private URLs, customer data, or proprietary checkpoints.
