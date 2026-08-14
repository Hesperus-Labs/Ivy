# Source requirements

The transpiler converts Python source, not bytecode and not an opaque runtime
graph. It uses `inspect.getsource`, parses the result with Python's `ast`
module, and compiles a rewritten function. That choice makes the output
auditable, but it also defines the input contract.

## Supported callable shape

A good source callable is:

- defined with `def` in a readable `.py` file;
- composed of ordinary Python expressions and registered framework calls;
- explicit about tensor inputs, parameters, state, and random keys;
- free of hidden mutation and source-framework globals required at execution;
- small enough that its conversion boundary is meaningful to test.

Imports may appear at module scope or inside the function. Framework imports
inside the function are removed after their aliases have been recorded.

```python
import torch as pt


def normalize_and_score(x, weight):
    value = pt.nn.functional.relu(pt.matmul(x, weight))
    return value / value.sum(dim=-1, keepdim=True)
```

## Inputs that are rejected

| Input | Result | Resolution |
| --- | --- | --- |
| REPL or notebook lambda | `SourceUnavailableError` | Move it to an importable `.py` function |
| Builtin/native function | `SourceUnavailableError` | Wrap it in a file-defined function using supported calls |
| Non-callable object | `UnsupportedObjectError` | Pass a function, bound method, or supported module |
| Unknown source name | `UnsupportedSourceError` | Use `torch`, `tensorflow`, `jax`, `ivy`, or `numpy` |
| Unknown target name | `UnsupportedTargetError` | Use a documented target |
| Unsupported framework call | `UnsupportedPrimitiveError` at execution | Add a lowering or isolate the operation |

Decorators and annotations are intentionally removed from generated source.
They commonly refer to source-only imports or wrap the original callable in a
way that is not portable. `inspect.unwrap` is used before inspection when the
wrapper correctly exposes `__wrapped__`.

## Python constructs

Ordinary assignments, arithmetic, indexing, local helper expressions, and
Python control flow remain Python in the generated function. Only recognized
framework calls, tensor methods, and dtype attributes are rewritten.

This has two consequences:

1. Python control flow executes with the target's normal eager/JIT rules. A
   data-dependent `if` over a JAX tracer still needs `jax.lax.cond` or a
   refactor.
2. An unrecognized global function remains a call to that global. Keep the
   converted boundary narrow and audit emitted source before deployment.

The converter does not promise arbitrary Python-to-graph compilation. Target
compilation is a separate optional step described in [caching and
compilation](cache-and-compilation.md).

## Source inference and aliases

When `source=` is omitted, Hesperus Ivy checks the callable's defining module.
This works for many framework-owned callables but cannot reliably infer the
intent of an application function defined in your package. Prefer an explicit
source in application and test code.

| User spelling | Canonical source/target |
| --- | --- |
| `pytorch` | `torch` |
| `tf`, `keras`, `tensorflow.keras` | `tensorflow` |
| `jnp`, `jax.numpy` | `jax` |
| `equinox` as a source | `jax` |
| `equinox` as a target | JAX runtime plus module-oriented reporting |

## Bound methods and simple modules

A bound method is inspected through its underlying function, then executed
against a proxy whose discoverable parameters and buffers have been converted
to target arrays. A module instance with a callable `forward` method is accepted
directly as a convenience.

This path covers straightforward feed-forward modules. Read [modules and
parameters](modules.md) before relying on nested modules, buffers, or custom
attribute behavior.

## A pre-conversion checklist

- The function is importable from a Python file.
- Every framework operation appears in the [primitive
  catalog](../reference/primitives.md).
- Inputs and outputs have a parity test including dtype and shape.
- State updates and random keys are values in the signature.
- Device placement happens outside the conversion boundary.
- The emitted Python contains no accidental source-framework call.
