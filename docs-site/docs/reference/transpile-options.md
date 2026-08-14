# `transpile` options

```python
ivy.transpile(
    obj,
    *additional,
    source=None,
    target=None,
    to=None,
    args=None,
    kwargs=None,
    mode="auto",
    backend_compile=False,
    cache=True,
    emit=None,
    output_dir=None,
    return_report=False,
    **legacy,
)
```

## Parameters

| Parameter | Current behavior |
| --- | --- |
| `obj` | File-defined callable, bound method, or simple module with `forward` |
| `*additional` | Composes callables in order through compatibility `trace_graph` behavior |
| `source` | `torch`, `tensorflow`, `jax`, `ivy`, or `numpy`; explicit is recommended |
| `target` | `torch`, `tensorflow`, `jax`, `equinox`, `ivy`, or `numpy`; default `jax` |
| `to` | Deprecated compatibility alias for `target`; disagreement raises `ValueError` |
| `args`, `kwargs` | Retained compatibility inputs; source conversion does not execute them |
| `mode` | `auto`, `source`, or compatibility `trace`; all currently use inspectable source |
| `backend_compile` | Applies public target JIT/compiler wrapper after conversion |
| `cache` | Reads/writes the user source cache; default `True` |
| `emit` | Directory or `.py` file for generated source and JSON manifest |
| `output_dir` | Compatibility alias used when `emit` is absent |
| `return_report` | Returns `TranspileResult` instead of only the converted value |
| `**legacy` | Unknown options are ignored with `DeprecationWarning` during migration |

## Return types

- One callable, `return_report=False`: converted callable or module adapter.
- One callable, `return_report=True`: `TranspileResult`.
- Multiple callables: callable `Graph` composition.
- `target="equinox"` alone does not guarantee an `eqx.Module`; use
  `to_equinox_module` for that container.

## Modes

`source` requires recoverable Python source and surfaces
`SourceUnavailableError`. `auto` and `trace` currently share the same
deterministic AST path; `trace` is retained as a migration spelling rather than
a private graph tracer.

## Aliases

Source normalization accepts common `pytorch`, `tf`/Keras, and JAX NumPy
spellings. Target normalization accepts `tf`, `keras`, and `jnp`. Reports use
canonical names.

## Composition

```python
graph = ivy.transpile(first, second, source="ivy", target="numpy")
result = graph(x)  # second(first(x))
print(graph.report.to_json())
```

The combined report deduplicates converted primitives and reports a cache hit
only if all component conversions hit.
