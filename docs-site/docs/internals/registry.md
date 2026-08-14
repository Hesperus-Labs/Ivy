# Internals: the primitive registry

`ivy.transpiler.registry.PRIMITIVES` is the source of truth for the public
conversion coverage table. Each `PrimitiveSpec` records:

- canonical operation name and category;
- maintained source and target frameworks;
- the official documentation URL used for semantics, plus direct links for
  PyTorch, TensorFlow, JAX, Equinox, and NumPy;
- the registry revision used in cache keys and reports.

```python
import ivy

manifest = ivy.compatibility_report(source="torch", target="equinox")
print(manifest["total"])
for item in manifest["primitives"]:
    print(item["name"], item["framework_docs"]["jax"])
```

The generated Pages compatibility table is deliberately inspectable JSON so a
release can be audited without reading implementation details.
