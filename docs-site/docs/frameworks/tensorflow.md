# TensorFlow source and target

Use `source="tensorflow"` (or `tf`) for TensorFlow-shaped Python and
`target="tensorflow"` for native `tf.Tensor` results. Maintained semantics are
linked to the public [TensorFlow Python API](https://www.tensorflow.org/api_docs/python/tf).

## Function example

```python
def tensorflow_forward(x, weight, bias):
    import tensorflow as tf

    hidden = tf.linalg.matmul(x, weight) + bias
    return tf.nn.relu(hidden)


torch_forward = ivy.transpile(
    tensorflow_forward,
    source="tensorflow",
    target="torch",
)
```

Aliases imported as `tf` and qualified namespaces such as `tf.nn`,
`tf.linalg`, `tf.math`, and `tf.random` are normalized by the lowerer.

## Signature normalization

The runtime reconciles common framework spelling differences:

| TensorFlow concept | Portable form |
| --- | --- |
| `axis` | target axis/dimension argument |
| `keepdims` | target keep-dimension spelling |
| `tf.concat` | canonical `cat`/concatenate behavior |
| `tf.clip_by_value` | canonical clipping |
| `tf.cast` | selected target dtype conversion |
| `tf.reduce_*` | canonical reductions |
| `tf.range`, `tf.fill` | target creation operations |

Do not infer exact TensorFlow broadcasting, sparse/ragged, or graph-collection
behavior from the presence of a similarly named primitive. The catalog defines
the dense tensor contract currently tested.

## Random operations

A source `seed=` becomes deterministic target seed material. For a JAX target,
the runtime constructs an explicit JAX key. TensorFlow's stateful and stateless
RNG families do not have identical algorithms across targets, so validate
range, shape, dtype, and repeatability rather than exact cross-framework bits.

## Target compilation

`backend_compile=True` wraps a TensorFlow target with
[`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function).
Python side effects and shape-dependent retracing remain governed by
TensorFlow. Convert and test eagerly before enabling the wrapper.

## Keras and variables

Keras/TensorFlow module names normalize to the TensorFlow source family, but a
complex `tf.keras.Model` is not automatically reconstructed as an Equinox
architecture. Extract a file-defined numerical `call`/forward kernel or write a
small explicit adapter that copies variable values into the desired module
layout.

## Known boundaries

- `tf.data`, distributed strategies, SavedModel signatures, ragged/sparse
  tensors, resources, lookup tables, and custom ops are outside the core.
- Variable mutation is not inferred as Equinox state.
- TensorFlow-specific graph collections and autograph rewrites are not copied.
- Device scopes and mixed precision policies belong to application setup.
- The maintained dependency profile currently pins TensorFlow 2.21.x.

```bash
ivy coverage --source tensorflow --target equinox
```
