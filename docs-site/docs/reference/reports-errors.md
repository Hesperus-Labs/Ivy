# Reports and errors

## `TranspileResult`

A frozen pair containing `.value` and `.report`. It supports tuple unpacking.

## `TranspileReport`

A frozen, JSON-serializable record with `to_dict()` and `to_json(indent=2)`.
See [reports and generated code](../guide/reports-and-emission.md) for field
semantics and an audit workflow.

## Exception hierarchy

All conversion-specific exceptions derive from `TranspileError`:

| Error | Meaning |
| --- | --- |
| `UnsupportedSourceError` | Source name cannot be normalized to a maintained family |
| `UnsupportedTargetError` | Target name is outside the supported target set |
| `SourceUnavailableError` | A readable function definition cannot be recovered |
| `UnsupportedObjectError` | Submitted value is not a supported callable/module |
| `UnsupportedPrimitiveError` | A rewritten operation has no runtime lowering for the target |

`ValueError` is used for conflicting `target=`/`to=` values and invalid mode
spellings. Missing optional Equinox installation raises `ModuleNotFoundError`
with the required extra. A non-Equinox object passed to
`from_equinox_module` raises `TypeError`.

## Handle only actionable boundaries

```python
from ivy.transpiler import SourceUnavailableError, UnsupportedPrimitiveError

try:
    converted = ivy.transpile(fn, source="torch", target="jax")
except SourceUnavailableError:
    raise RuntimeError("Move this callable into a Python module")

try:
    output = converted(x)
except UnsupportedPrimitiveError as exc:
    print(exc)
```

Do not catch `TranspileError` and silently execute the source function in
production. That reintroduces the dependency/device behavior the conversion
was intended to remove.

## Deprecation warnings

`to=`, `output_dir=`, trace spelling, graph aliases, and unknown legacy options
exist for migration. Run tests with deprecation warnings visible and update new
code to `target=`, `emit=`, and the direct function/module APIs.
