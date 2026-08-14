# Primitive catalog

The current registry revision is `2026.08`. All entries declare PyTorch,
TensorFlow, JAX, Ivy, and NumPy sources plus PyTorch, TensorFlow, JAX, Equinox,
Ivy, and NumPy targets. Runtime branches—not the declaration alone—define
actual semantics and are covered by focused tests.

Query machine-readable details and official links:

```python
manifest = ivy.compatibility_report(source="torch", target="jax")
for operation in manifest["primitives"]:
    print(operation["name"], operation["category"])
    print(operation["framework_docs"]["jax"])
```

## Creation and dtype

| Primitive | Purpose |
| --- | --- |
| `zeros`, `ones`, `full`, `empty` | Create arrays with requested shape/dtype |
| `eye`, `arange`, `linspace` | Structured numerical creation |
| `zeros_like`, `ones_like` | Create from another value's shape/dtype |
| `cast` | Convert to a selected target dtype |

`empty` may map to a deterministic initialized value on targets without an
equivalent safe public operation. Do not depend on uninitialized memory.

## Tensor and linear algebra

| Primitive | Notes |
| --- | --- |
| `add`, `sub`, `mul`, `div` | Elementwise arithmetic and target broadcasting |
| `matmul`, `dot`, `tensordot`, `einsum` | Public target linear algebra |
| `linear` | Preserves source functional-linear weight layout |
| `reshape`, `transpose`, `flatten`, `expand_dims` | Shape/axis manipulation |
| `norm` | Target public linear-algebra norm |

## Reductions

`sum`, `mean`, `prod`, `max`, `min`, `argmax`, and `argmin` normalize common
axis/dimension and keep-dimension spellings. Edge behavior for empty inputs and
tie-breaking remains target-defined unless covered by a specific parity test.

## Activations and neural-network operations

| Primitive | Purpose |
| --- | --- |
| `relu`, `relu6`, `gelu`, `silu` | Common activations |
| `sigmoid`, `tanh`, `softmax`, `log_softmax` | Nonlinear/probability transforms |
| `dropout` | Training/inference random masking path |
| `one_hot`, `embedding` | Categorical lookup operations |

## Elementwise and selection

`exp`, `log`, `sqrt`, `abs`, `isfinite`, `where`, `maximum`, and `minimum`
dispatch to public target equivalents. Floating-point domain/NaN details follow
the target and device.

## Manipulation

`cat`, `stack`, `split`, `clip`, and `pad` normalize common naming and keyword
differences. Test application-specific padding order and split remainder cases.

## Autodiff and random

| Primitive | Contract |
| --- | --- |
| `stop_gradient` | Target detach/stop-gradient behavior |
| `key` | Construct/represent explicit random seed state |
| `randn` | Normally distributed random values |
| `uniform` | Uniform random values |
| `randint` | Integer random values in a half-open range |

Random calls preserve documented deterministic behavior, shapes, dtypes, and
ranges—not bit identity across frameworks.

## Tensor-method coverage

The lowerer additionally recognizes common methods including `view`, `permute`,
`float`, `double`, `half`, `to`, `astype`, `detach`, `contiguous`, `clone`,
`copy`, `size`, `numel`, `ndim`, `squeeze`, `unsqueeze`, `mm`, `clamp`,
`repeat`, `tile`, and `chunk`. These normalize into runtime method behavior and
may share a canonical primitive with a catalog entry.

Run `ivy coverage` for the authoritative JSON rather than scraping this page.
