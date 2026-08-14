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

Pass `source=` explicitly in applications. Inference works best for
framework-owned callables; a function defined in your package otherwise looks
Ivy-shaped.

## Callable requirements

Define the function in an importable Python file. Imports may be local or
global and common aliases are recognized. Decorators and annotations are
removed from generated source. Bound methods and simple modules with `forward`
are accepted; use `to_equinox_module` when the output must be an Equinox PyTree.

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

Errors occur at two stages. Object/source errors happen during `transpile`.
An unknown rewritten operation raises `UnsupportedPrimitiveError` when
execution reaches it, tying the diagnostic to the actual target path.

## Compilation and caching

Set `backend_compile=True` only after eager behavior is correct. It applies
`torch.compile`, `tf.function`, or `jax.jit` to the generated target callable.
The cache is keyed by generated source, source/target, compilation option,
registry revision, and object name. Inspect or clear it with:

```bash
ivy cache-info
ivy cache-clear
```

## Emit and audit

`emit="directory"` chooses a content-addressed filename;
`emit="file.py"` controls it. Both add an adjacent JSON manifest. Generated
source depends on `ivy.transpiler.runtime`, so pin Hesperus Ivy when deploying.

## Compose functions

```python
graph = ivy.transpile(
    preprocess,
    model,
    postprocess,
    source="ivy",
    target="jax",
)
output = graph(inputs)
print(graph.report.converted_primitives)
```

For mixed sources, transpile components separately and compose their target
callables explicitly.

## Option summary

| Option | Recommendation |
| --- | --- |
| `source`, `target` | Explicit in application code |
| `cache` | Enabled; disable to isolate stale behavior |
| `emit` | Enable for audited build artifacts |
| `return_report` | Enable in migration tools and CI |
| `backend_compile` | Enable after eager parity |
| `mode` | Use `auto`; conversion is currently source-based |

See [reports and emission](../guide/reports-and-emission.md) and the complete
[option reference](../reference/transpile-options.md).
