# Transpile functions and inspect behavior

## Source and target selection

Use explicit names in applications and notebooks. `source=None` is convenient
for framework-defined callables, while `target="equinox"` is the module-focused
alias for JAX.

```python
converted = ivy.transpile(fn, source="tensorflow", target="equinox")
converted = ivy.transpile(fn, source="jax", target="torch")
converted = ivy.transpile(fn, source="torch", target="tensorflow")
```

The legacy `to=` alias remains available for one migration cycle:

```python
converted = ivy.transpile(fn, source="torch", to="jax")
```

## Reports and failures

```python
converted, report = ivy.transpile(
    fn,
    source="torch",
    target="equinox",
    cache=True,
    return_report=True,
)
```

`TranspileError` subclasses identify the semantic boundary. A missing source
file raises `SourceUnavailableError`; a primitive without a lowering raises
`UnsupportedPrimitiveError` with its source operation and target. Hesperus Ivy
does not silently fall back to executing the original framework.

## Compilation and caching

Set `backend_compile=True` only after eager behavior is correct. It applies
`torch.compile`, `tf.function`, or `jax.jit` to the generated target callable.
The cache is keyed by generated source, source/target, configuration, and
framework versions. Inspect or clear it with:

```bash
ivy cache-info
ivy cache-clear
```
