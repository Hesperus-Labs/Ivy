# Caching and target compilation

Source conversion and target compilation are separate phases. Hesperus Ivy
caches rewritten Python by default; target frameworks may then compile the
callable when `backend_compile=True`.

## Source cache

The cache key includes normalized generated source, source and target names,
the compilation option, primitive registry revision, and object name. Inspect
the platform-specific location without guessing:

```bash
ivy cache-info
```

Override it for CI or a hermetic build:

```bash
export HESPERUS_IVY_CACHE_DIR="$PWD/.cache/hesperus-ivy"
```

Each entry is a `.py` source file plus `.json` metadata. Writes use a per-key
file lock, `fsync`, and atomic replacement. The cache does not pickle callables
or models.

```bash
ivy cache-clear
```

Disable it for one conversion with `cache=False`.

## Target compilation

`backend_compile=True` applies a public target wrapper:

| Target | Wrapper |
| --- | --- |
| PyTorch | `torch.compile` |
| TensorFlow | `tf.function` |
| JAX / Equinox | `jax.jit` |
| NumPy / Ivy reference | no additional wrapper |

```python
compiled = ivy.transpile(
    fn,
    source="torch",
    target="jax",
    backend_compile=True,
)
```

Target compilers impose their own tracing rules. Python control flow over
tensor values, dynamic shapes, non-array arguments, side effects, and device
placement can affect tracing. Establish eager parity first.

For an Equinox module with mixed array and static leaves, prefer
`eqx.filter_jit(module)` after conversion. It treats array leaves dynamically
and other leaves statically, matching Equinox's [filtered transformation
model](https://docs.kidger.site/equinox/api/transformations/).

## Reproducible builds

The source cache is an optimization, not a release artifact. A deployment
should pin Hesperus Ivy, keep an application `uv.lock`, test with a cold cache,
archive emitted source/report when needed, and compile on deployment hardware
when the target compiler uses device-specific artifacts.

If behavior appears stale, compare `report.registry_revision`, clear the cache,
and rerun with `cache=False`. Record framework versions with benchmark results.
