# Modules and parameters

Function and module conversion share AST/runtime lowering, but modules also
need parameter discovery and a target-native container.

## Supported boundary

`ivy.transpile(module)` accepts an object with a callable `forward` method. A
bound `module.forward` is equivalent for the common case. The target proxy
discovers:

- PyTorch-style `_parameters` and `_buffers` mappings;
- child objects in `_modules`;
- public attributes with `shape` and `dtype`;
- nested public attributes with callable `forward` methods;
- simple static strings, numbers, containers, booleans, and `None`.

Array values are copied through a host NumPy representation and rebuilt as
target arrays. This semantic boundary may move data to host memory.

```python
converted = ivy.transpile(torch_module, source="torch", target="jax")
y = converted(jax.numpy.ones((4, in_features)))
```

The adapter is callable but not necessarily an `eqx.Module`. Use the bridge
when the result must be a PyTree:

```python
model = ivy.to_equinox_module(torch_module, source="torch")
model = equinox.filter_jit(model)
```

For parameterized modules, the Equinox wrapper stores converted parameters as
array leaves and keeps the source template, paths, and rewritten function
static.

## Weight layouts

Operations may encode source layouts. PyTorch functional linear treats weight
as `(out_features, in_features)`, so the runtime applies the correct transpose
on targets whose matrix multiplication expects the reverse orientation. Use
non-square weights in parity tests; square identities hide transpose bugs.

## Write an explicit adapter when needed

An application-owned Equinox module is required for parameter aliasing,
parametrizations, hooks, lazy materialization, forward-time buffer mutation,
custom autograd/native extensions, resource handles, dynamic child modules, or
distributed/sharded wrappers.

## Serialization

```python
ivy.save_equinox(model, "checkpoints/model.eqx")
restored = ivy.load_equinox("checkpoints/model.eqx", like=model_template)
```

Saving uses Equinox's [leaf
serializer](https://docs.kidger.site/equinox/api/serialisation/) and writes an
adjacent Hesperus manifest. Loading needs a trusted `like` tree with the same
structure and compatible leaves.
