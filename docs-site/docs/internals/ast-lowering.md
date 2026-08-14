# Internals: AST lowering

The source transformer lives in `ivy/transpiler/api.py`. It uses Python's
public `ast.NodeTransformer` to make a narrow set of changes while preserving
ordinary Python structure.

## Preparation

`_prepare_source` obtains and dedents source, parses it, finds the submitted
function, and removes decorators plus annotations. It initializes `_Lowerer`
with the source framework and callable globals, then unparses the transformed
function with `ast.unparse`.

Annotations are removed because a generated function may not import their
source-only names. Decorators are removed because they could execute source
framework wrappers or alter inspection behavior.

## Alias collection

The lowerer begins with standard aliases (`torch`, `tf`, `jnp`, `np`, `ivy`)
and inspects global values for module names. It also records import aliases
inside the function before deleting framework import nodes.

For example:

```python
from torch.nn import functional as F

return F.relu(x)
```

is resolved to a qualified PyTorch operation even though the generated source
does not retain the import.

## Call rewriting

Framework-qualified calls become:

```python
_ivy_runtime.call("torch.matmul", x, weight, target=_ivy_target)
```

The original positional and keyword arguments are preserved, then a fixed
target keyword is added. Canonical primitive names are collected for the
report.

Common tensor methods become `_ivy_runtime.method`. Method lowering also runs
when the receiver is itself a generated runtime call, so a chain such as
`torch.relu(x).flatten(0, 1)` remains portable.

## Dtype attributes

Framework dtype attributes are replaced by target-aware dtype lookup:

```python
torch.float32
```

becomes conceptually:

```python
_ivy_runtime.dtype("torch.float32", target=_ivy_target)
```

This avoids importing the source framework merely to evaluate a dtype constant.

## Intentionally preserved code

Arithmetic operators, indexing, comprehensions, control flow, and non-framework
global calls remain Python. This keeps the transformer understandable but means
emitted code must be audited. A future primitive should not be added through a
broad name-based rewrite that could capture unrelated application functions.

## Failure timing

An unknown framework call is still rewritten to the runtime, which raises
`UnsupportedPrimitiveError` when executed. This makes the error include the
qualified operation and selected target. Syntax/source discovery errors fail
during conversion.

## Contributor invariants

- Never mutate the caller's frame or globals.
- Never import every optional framework during conversion.
- Never silently execute the original framework as fallback.
- Preserve source locations when constructing AST nodes.
- Add alias, generated-source, report, and runtime parity tests together.
