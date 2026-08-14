# Randomness and state

Portable numerical code must make random state and mutable model state visible.
Hesperus Ivy uses JAX/Equinox semantics as its design center because explicit
values compose with transformations and distributed execution.

## JAX keys are explicit

Current JAX uses typed keys created by `jax.random.key(seed)`. A random call
does not mutate its key; reuse reproduces the same sample. Split keys for
independent samples:

```python
import jax

key = jax.random.key(0)
key, dropout_key = jax.random.split(key)
key, noise_key = jax.random.split(key)
```

This follows the [official JAX random-number
model](https://docs.jax.dev/en/latest/random-numbers.html). Hesperus Ivy does
not introduce a global JAX PRNG stream.

## Cross-framework random lowering

| Source form | JAX target | Non-JAX target |
| --- | --- | --- |
| JAX call with a key | Passes the key to `jax.random` | Derives a deterministic integer seed |
| Torch/TF call with `seed=` | Constructs a deterministic JAX key | Uses public seeded target behavior |
| Call without usable key/seed | Uses the documented runtime path | May be stateful according to the target |

Conversion promises shape, dtype class, range, and deterministic reuse where
documented. It does not promise identical bits across frameworks, versions, or
accelerators.

## Explicit module state

```python
module, state = ivy.to_equinox_module(
    native_module,
    source="torch",
    state={"step": 0},
)
output, state = module(x, state=state)
```

The generic wrapper currently returns supplied state unchanged. It establishes
the value contract but does not infer arbitrary source mutation. Implement
running statistics, caches, counters, and streaming buffers explicitly.

Use `inference=True` to apply `eqx.nn.inference_mode(..., value=True)` to a new
bridge, or call that public Equinox function directly for an existing tree.

## Rules for portable random code

- Accept a key or seed at the application boundary.
- Split keys close to consuming operations.
- Return next key/state when an API owns sequencing.
- Test determinism and distribution properties separately.
- Do not compare exact cross-framework samples as a portability guarantee.
- Record framework versions for multi-machine reproducibility.
