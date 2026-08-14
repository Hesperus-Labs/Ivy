# Runnable examples

The repository keeps small examples under `examples/` so they can be copied
into a project or executed directly:

| Example | Requirements | What it demonstrates |
| --- | --- | --- |
| `transpile_torch_to_numpy.py` | base + NumPy | source inspection, native output, report |
| `transpile_ivy_to_numpy.py` | base + NumPy | Ivy compatibility alias |
| `equinox_module.py` | `jax` extra | explicit PyTree fields and gradients |
| `diagnose_environment.py` | base | JSON diagnostics for support requests |

Run a base example with:

```bash
uv run python examples/transpile_torch_to_numpy.py
```
