# Production deployment

Hesperus Ivy `2.0.0a1` is an alpha. Evaluate it in production only with a
pinned commit, locked environment, and conversion-specific parity tests.

## Pin the source

```bash
uv pip install --torch-backend=cpu \
  "hesperus-ivy[all-cpu] @ git+https://github.com/Hesperus-Labs/Ivy.git@<commit>"
```

Replace `<commit>` with a reviewed 40-character SHA. Do not deploy a moving
`@main` reference.

## Choose the smallest runtime

| Workload | Extra |
| --- | --- |
| NumPy/reference conversion | base package |
| JAX/Equinox inference or training | `jax` |
| PyTorch target | `torch` |
| TensorFlow target | `tensorflow` |
| Cross-framework CPU CI | `all-cpu` |
| NVIDIA workstation | `nvidia` |

Only the selected target is imported by generated functions. The source
framework is unnecessary at runtime when emitted code no longer references it
and parameters are already converted.

## Artifact checklist

- pinned Hesperus commit/tag and application `uv.lock`;
- emitted Python/JSON when used;
- serialized `TranspileReport`;
- `ivy doctor` output from the deployment image;
- parity results for representative shapes and dtypes;
- post-compilation hardware benchmarks;
- leaf checkpoint plus trusted code constructing the `like` tree.

Set `HESPERUS_IVY_CACHE_DIR` to a writable location or disable caching in
read-only containers. Select a global Ivy backend once during service startup;
do not switch it per request. Transpiled callables fix their target and need no
global backend.

Generated Python is executable code. Review it and obtain it from the trusted
build. Diagnostics can include platform/device details, so inspect doctor JSON
before publishing it.

## Upgrade procedure

1. Resolve the new commit and lock dependencies.
2. Clear the source cache.
3. Run eager parity on maintained pairs.
4. Re-emit and review Python.
5. Run compiled parity and benchmarks on deployment hardware.
6. Record registry and framework versions.
7. Deploy with the normal application rollback path available.
