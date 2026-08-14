# Framework compatibility

The registry links each primitive to the maintained official documentation:
[PyTorch](https://docs.pytorch.org/docs/stable/),
[TensorFlow](https://www.tensorflow.org/api_docs/python/tf),
[JAX](https://docs.jax.dev/en/latest/),
[Equinox](https://docs.kidger.site/equinox/), and
[NumPy](https://numpy.org/doc/stable/). These links are emitted with every
compatibility report so an adapter's semantics can be checked at the source.

The matrix below is generated from the checked-in primitive registry. Query it
programmatically with `ivy.compatibility_report()` or from the CLI with
`ivy coverage`.

| Source | Target | Contract |
| --- | --- | --- |
| PyTorch | TensorFlow | documented stable core, native/composite lowerings |
| PyTorch | JAX/Equinox | Equinox-first path, explicit keys/state |
| TensorFlow | PyTorch | documented stable core |
| TensorFlow | JAX/Equinox | Equinox-first path, explicit keys/state |
| JAX/Equinox | PyTorch | documented stable core |
| JAX/Equinox | TensorFlow | documented stable core |
| Any maintained source | NumPy | reference target for pure tensor behavior |

Framework-internal symbols, distributed runtimes, data pipelines, and custom
native kernels are reported as outside the stable matrix. Version ranges are
kept narrow until scheduled compatibility CI certifies a framework minor.

## Certified environment

The quality workflow installs `all-cpu` on Python 3.12 and 3.13. Maintained
ranges are PyTorch 2.13.x, TensorFlow 2.21.x, JAX 0.11.x, Equinox 0.13.8.x,
Optax 0.2.8.x, and NumPy 2.x.

## What “supported” means

A supported primitive has a registry entry with official links, public target
runtime behavior, focused tests, and an explicit error instead of hidden
source-framework fallback. It does not promise bit-identical random/floating
results, identical storage aliasing, or every framework overload.

## Coverage by category

| Category | Operations |
| --- | --- |
| Creation/dtype | zeros, ones, full, empty, eye, ranges, like creation, cast |
| Arithmetic/tensor | arithmetic, reshape, transpose, flatten, expand dimensions |
| Linear algebra | matmul, dot, tensordot, einsum, linear, norm |
| Reductions | sum, mean, product, min/max, argmin/argmax |
| Activations | ReLU/ReLU6, GELU, SiLU, sigmoid, tanh, softmax variants |
| Elementwise/selection | exp, log, sqrt, abs, finite check, where, min/max |
| Manipulation | concatenate, stack, split, clipping, padding |
| Neural network | dropout, one-hot, embedding |
| Random/autodiff | normal, uniform, integer random, key, stop-gradient |

See the [primitive catalog](reference/primitives.md) for canonical names and
method coverage.

## Target-native results

| Target | Result family | Compile option |
| --- | --- | --- |
| NumPy / Ivy | `numpy.ndarray` | eager only |
| JAX / Equinox | `jax.Array` | `jax.jit` or filtered transforms |
| PyTorch | `torch.Tensor` | `torch.compile` |
| TensorFlow | `tf.Tensor` | `tf.function` |

Supply target-native inputs to control dtype/device placement.

The registry currently contains the stable tensor, creation, reduction,
neural-network, random, dtype, and state boundary primitives. Each entry also
contains direct official documentation links:

```python
import ivy

for item in ivy.compatibility_report()["primitives"]:
    print(item["name"], item["framework_docs"]["pytorch"])
```

```bash
ivy coverage --source tensorflow --target equinox > coverage.json
```

## Explicit gaps

- opaque/native callables without file-defined Python source;
- data pipelines, distributed runtimes, sharding wrappers, checkpoint formats;
- sparse, ragged, quantized, symbolic, and custom-device tensor families;
- custom kernels/autodiff extensions and arbitrary training loops;
- mutation or aliasing that cannot be represented as target values.

Keep these concerns outside the converted kernel or write an explicit adapter.
