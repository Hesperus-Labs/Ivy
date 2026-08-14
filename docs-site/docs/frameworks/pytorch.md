# PyTorch source and target

Use `source="torch"` for PyTorch-shaped functions and modules, and
`target="torch"` when PyTorch tensors must remain the execution result.
Hesperus Ivy follows the public [PyTorch API
reference](https://docs.pytorch.org/docs/stable/) for maintained operations.

## Function example

```python
def torch_forward(x, weight, bias):
    import torch

    hidden = torch.matmul(x, weight) + bias
    return torch.nn.functional.gelu(hidden)


jax_forward = ivy.transpile(
    torch_forward,
    source="torch",
    target="jax",
)
```

Supported calls are recognized through `torch.*`, `torch.nn.functional.*`,
local module aliases, and common tensor methods. The catalog includes creation,
linear algebra, activations, reductions, manipulation, dtype conversion,
embedding, padding, dropout, and random operations.

## Tensor methods

The AST lowerer normalizes common methods including `reshape`, `view`,
`transpose`, `permute`, `flatten`, reductions, dtype conversions, `detach`,
`clone`, `squeeze`, `unsqueeze`, activations, matrix multiplication, clipping,
repetition, splitting, and chunking.

Method semantics are portable only where documented. `view` is treated as a
reshape request rather than a promise about storage contiguity. `detach`
becomes the target's stop-gradient behavior. `clone`/`copy` preserve value
semantics but do not promise identical storage aliasing.

## Modules

Simple `torch.nn.Module` objects are supported through inspectable `forward`
source and registered parameters/buffers. For a JAX PyTree result:

```python
eqx_model = ivy.to_equinox_module(torch_model, source="torch")
```

The bridge converts parameters to JAX arrays. Optimizer state, hooks, custom
autograd, parametrizations, and distributed wrappers do not migrate
automatically. See [modules and parameters](../guide/modules.md).

## Randomness

PyTorch often uses a process or generator state. For portability, prefer a
function signature containing an explicit seed or key. A PyTorch seed lowered
to JAX becomes a deterministic JAX key. The same seed does not promise the same
sample bits across frameworks.

## Target compilation

`backend_compile=True` wraps the generated PyTorch callable with
[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).
Compilation is optional and happens after source rewriting. Confirm eager
parity first, then benchmark representative input shapes.

## Known boundaries

- `torch.distributed`, DataLoader pipelines, TorchScript archives, and export
  graphs are not transpiler inputs.
- Custom C++/CUDA operators and `torch.autograd.Function` need adapters.
- Device placement and autocast belong outside the portable function.
- Mutation/aliasing is not guaranteed to match PyTorch storage semantics.
- The maintained dependency profile currently pins PyTorch 2.13.x.

Generate the live coverage manifest with:

```bash
ivy coverage --source torch --target jax
```
