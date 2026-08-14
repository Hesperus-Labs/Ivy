# NumPy and the Ivy reference target

NumPy is the simplest target for semantic inspection and parity tests. The
`ivy` transpiler target currently uses the same NumPy runtime path while
preserving the historical API spelling in reports and migration code.

## Use NumPy for first-pass parity

```python
numpy_fn, report = ivy.transpile(
    source_fn,
    source="torch",
    target="numpy",
    return_report=True,
)
```

This makes output inspection straightforward and avoids target compiler
effects. It does not prove accelerator, autodiff, or JIT compatibility; repeat
the parity test on the intended deployment target.

## Ivy library backend dispatch

The public Ivy API remains available for backend-neutral application code:

```python
ivy.set_backend("numpy")
try:
    x = ivy.asarray([[1.0, -2.0]])
    y = ivy.relu(x)
finally:
    ivy.unset_backend()
```

The transpiler and the backend dispatcher solve different problems. A
transpiled callable fixes a target at conversion time. `set_backend` changes
how public `ivy.*` functions dispatch within the process.

## `unify` compatibility alias

`ivy.unify(fn, source=...)` is retained as a compatibility route to the Ivy
reference target. New migration code should call `ivy.transpile(...,
target="ivy")` explicitly so reports and intent are easy to read.

## Behavioral notes

- NumPy has no target JIT wrapper in Hesperus Ivy.
- Autodiff is not supplied by the NumPy target.
- Device placement is CPU/host.
- Source random keys are mapped to deterministic NumPy seed material where
  supported, not bit-identical JAX samples.
- Array aliasing and in-place semantics are not a cross-framework guarantee.

The maintained NumPy range is 2.x. The registry links each operation to its
current [NumPy reference](https://numpy.org/doc/stable/reference/).
