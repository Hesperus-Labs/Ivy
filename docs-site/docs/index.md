# Hesperus Ivy

Hesperus Ivy is a portable ML API and source-to-source transpiler maintained
by Hesperus Labs and Aadesh Kumar. It lets a team keep a model’s public
behavior while moving implementation between PyTorch, TensorFlow, JAX/Equinox,
and NumPy.

This is the current Hesperus Ivy project, version `2.0.0a1` in active alpha
development. The [project status](status.md) page defines the supported
contract; the historical upstream Ivy Sphinx tree is not used to build these
pages.

The recommended workflow is Equinox-first: convert a framework function or
module to target-native JAX code, make state and random keys explicit, and then
compose it with Equinox’s filtered transformations.

## Choose a path

| You want to… | Start here |
| --- | --- |
| Install from Git in a clean environment | [Install](install.md) |
| Run your first conversion | [Five-minute quickstart](tutorials/quickstart.md) |
| Convert a real function and inspect generated code | [Transpile a function](tutorials/transpiling.md) |
| Use Equinox state, RNG keys, and gradients | [Equinox modules and state](tutorials/equinox.md) |
| Understand how the converter works internally | [Pipeline tour](internals/pipeline.md) |
| Add a supported primitive | [Add a lowering](internals/extending.md) |

## Stable contract

The stable source-to-source matrix covers the documented tensor,
neural-network, random, and module/state core. Native target autodiff,
control-flow, optimizer, and serialization APIs remain available around the
converted callable when that target supports them. The compatibility page is
generated from the primitive registry and states where a lowering is native,
composite, or intentionally unavailable.

Distributed runtimes, data pipelines, custom native kernels, and
framework-internal symbols are outside the stable contract. Hesperus Ivy emits
an explicit report for those boundaries instead of silently invoking the source
framework.

## Install in one command

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install --torch-backend=cu130 \
  "hesperus-ivy[nvidia] @ git+https://github.com/Hesperus-Labs/Ivy.git@main"
```

Use `--torch-backend=cpu` and the `all-cpu` extra for a CPU/reference setup.
Run `ivy doctor` to see exactly which optional frameworks and devices are
visible.
