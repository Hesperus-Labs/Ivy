# Hesperus Ivy

Hesperus Ivy is Aadesh Kumar’s Equinox-first fork of Ivy: a portable ML API
and a checked-in Python source-to-source transpiler for PyTorch, TensorFlow,
JAX/Equinox, and NumPy.

[![CI](https://github.com/Hesperus-Labs/Ivy/actions/workflows/quality.yml/badge.svg)](https://github.com/Hesperus-Labs/Ivy/actions/workflows/quality.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-2563eb)](https://hesperus-labs.github.io/Ivy/)

The documentation is the primary onboarding surface:
**[hesperus-labs.github.io/Ivy](https://hesperus-labs.github.io/Ivy/)**.
It contains runnable tutorials, the generated API reference, framework
compatibility tables, migration guides, and an internal implementation tour.

## Install from Git with uv

Hesperus Ivy supports Python 3.12 and 3.13. The base package is lightweight;
frameworks are opt-in extras.

```bash
# CPU/reference environment
uv venv --python 3.13
source .venv/bin/activate
uv pip install --torch-backend=cpu \
  "hesperus-ivy[all-cpu] @ git+https://github.com/Hesperus-Labs/Ivy.git@main"

# Aadesh's Linux/NVIDIA path (RTX 4060, CUDA 13 PyTorch wheels)
uv venv --python 3.13
source .venv/bin/activate
uv pip install --torch-backend=cu130 \
  "hesperus-ivy[nvidia] @ git+https://github.com/Hesperus-Labs/Ivy.git@main"
```

`@main` is the current GitHub branch. Pin that suffix to a release tag when
deploying a reproducible application environment.

For local development:

```bash
uv sync --python 3.12
uv run pytest tests
uv run --group docs mkdocs serve
```

Run `ivy doctor` after installation to inspect optional frameworks and GPU
visibility. The installer never downloads private compiler binaries and the
package can be built without network access after dependencies are resolved.

## First transpilation

```python
import numpy as np
import ivy


def torch_style(x, weight, bias):
    import torch

    return torch.relu(torch.matmul(x, weight) + bias)


numpy_fn, report = ivy.transpile(
    torch_style,
    source="torch",
    target="numpy",
    return_report=True,
)

value = numpy_fn(
    np.ones((2, 3), dtype=np.float32),
    np.eye(3, dtype=np.float32),
    np.zeros(3, dtype=np.float32),
)
print(value, report.converted_primitives)
```

The Equinox-first path uses the same API:

```python
import ivy

equinox_fn = ivy.transpile(torch_style, source="torch", target="equinox")
```

JAX random operations take an explicit `key=`. Stateful Equinox conversions
return `(module, state)` and never hide mutable state in a process-global.

## Public surface

- `ivy.transpile`: source-to-source conversion with native target callables,
  safe user-scoped caching, optional emitted source, and JSON reports.
- `ivy.trace_graph`: compatibility entry point returning a callable `Graph`.
- `ivy.to_equinox_module` / `ivy.from_equinox_module`: explicit Equinox module
  bridges and state handling.
- `ivy.compatibility_report`: machine-readable primitive coverage.
- `ivy doctor`: Python, framework, CUDA, and JAX-device diagnostics.

The maintained source-to-source matrix covers the documented tensor,
neural-network, random, and module/state core. Native target autodiff,
control-flow, optimizer, and serialization APIs remain available around the
converted callable when that target supports them. Data pipelines, distributed
runtimes, custom native kernels, and framework-internal symbols are explicitly
outside the stable transpiler contract.

## Development and documentation

The repository uses `pyproject.toml` and `uv.lock`. Build the complete Pages
site locally with:

```bash
uv sync --python 3.12
uv run --group docs mkdocs build --strict
```

Every tutorial is expected to run in CI. The **Internals** section explains the
source inspector, canonical lowering runtime, primitive registry, cache, and
Equinox state model so contributors can add a primitive without guessing at
hidden behavior.

Hesperus Ivy preserves the upstream Ivy Apache-2.0 license and attribution.
