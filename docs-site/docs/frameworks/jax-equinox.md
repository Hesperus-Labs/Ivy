# JAX and Equinox

JAX is the recommended function target; Equinox is the recommended module
layer. They share the same native runtime arrays, but `target="equinox"`
communicates module-oriented intent and includes Equinox in reports.

Current behavior follows the public [JAX documentation](https://docs.jax.dev/)
and [Equinox documentation](https://docs.kidger.site/equinox/).

## Function conversion

```python
jax_fn = ivy.transpile(
    source_fn,
    source="torch",
    target="jax",
)

x = jax.numpy.asarray(source_values)
y = jax_fn(x)
```

Target operations return `jax.Array` values. Hesperus Ivy uses public JAX
array types and functions; it does not depend on the historical private
`DeviceArray` class.

## Equinox module conversion

```python
model = ivy.to_equinox_module(native_model, source="torch")

@equinox.filter_jit
def predict(model, x):
    return model(x)
```

An `eqx.Module` is a JAX PyTree. Converted parameter arrays are dynamic leaves;
rewritten callables and structural metadata are static fields. This composes
with `eqx.filter_jit`, `eqx.filter_value_and_grad`, `jax.vmap`, and standard
PyTree utilities, subject to the operations inside the callable.

## State and inference

Pass `state=` to request an explicit `(module, state)` bridge and return
`(output, state)` from calls. The generic bridge does not fabricate source
mutation. Use Equinox's own state API or application code for real updates.

`inference=True` applies `eqx.nn.inference_mode` to the converted module value.
There is no global train/eval switch.

## Random keys

Use typed keys and split them explicitly:

```python
key = jax.random.key(42)
key, step_key = jax.random.split(key)
output = random_fn(step_key, shape)
```

The [JAX random API](https://docs.jax.dev/en/latest/jax.random.html) states that
keys are explicit and are not mutated by random calls. Reusing one key repeats
the same result.

## JIT and filtered transformations

Function-level `backend_compile=True` uses `jax.jit`. For modules, filtered
Equinox transformations are usually clearer because they partition mixed
PyTrees leaf by leaf. See the official [Equinox transformations
reference](https://docs.kidger.site/equinox/api/transformations/).

## Checkpoints

`ivy.save_equinox` and `ivy.load_equinox` delegate array leaves to Equinox's
native tree serialization. Loading requires a matching `like` tree; preserve
hyperparameters in trusted application metadata/code.

## Known boundaries

- Data-dependent Python control flow may fail under JIT.
- Exact floating-point values can vary across platforms/JAX releases.
- Sharding, distributed initialization, and device meshes are application
  concerns.
- Flax/Haiku variable collections are not part of the new internal model.
- The maintained profile currently pins JAX 0.11.x and Equinox 0.13.8.x.
