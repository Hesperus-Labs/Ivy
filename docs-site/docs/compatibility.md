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

The registry currently contains the stable tensor, creation, reduction,
neural-network, random, dtype, and state boundary primitives. Each entry also
contains direct official documentation links:

```python
import ivy

for item in ivy.compatibility_report()["primitives"]:
    print(item["name"], item["framework_docs"]["pytorch"])
```
