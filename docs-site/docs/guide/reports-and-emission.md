# Reports and generated code

Every conversion can return a `TranspileReport`. Treat the report as part of
the build artifact: it explains what was converted, which framework pair was
selected, what operations were rewritten, and which dependency versions were
visible.

```python
import ivy

result = ivy.transpile(
    source_fn,
    source="torch",
    target="equinox",
    return_report=True,
)
converted = result.value
report = result.report
print(report.to_json())
```

`TranspileResult` can also be unpacked as `(value, report)`.

## Report fields

| Field | Meaning |
| --- | --- |
| `object_name` | Qualified name of the submitted callable |
| `source`, `target` | Canonical framework names after alias normalization |
| `mode` | Requested `auto`, `source`, or compatibility `trace` spelling |
| `cache_hit` | Whether generated source was loaded from the user cache |
| `source_available` | Whether source inspection succeeded |
| `emitted_path` | Generated `.py` path when `emit=` was requested |
| `converted_primitives` | Canonical operations found by AST lowering |
| `warnings` | Structured conversion warnings |
| `framework_versions` | Installed versions relevant to the selected pair |
| `registry_revision` | Primitive-contract revision included in cache identity |

The report is JSON-serializable and contains no tensor values, model weights,
credentials, or source-framework objects.

## Emit readable Python

Pass a directory to choose a content-addressed filename, or a `.py` path to
control it:

```python
converted, report = ivy.transpile(
    source_fn,
    source="tensorflow",
    target="jax",
    emit="build/ivy-generated",
    return_report=True,
)
```

Emission writes a Python file and adjacent JSON manifest. The Python imports
the public transpiler runtime, fixes the target, and defines the rewritten
function. The JSON records source, target, cache key, and primitives.

!!! warning "Generated source is tied to the runtime contract"
    Generated Python imports `ivy.transpiler.runtime`. Pin the Hesperus Ivy
    commit or release that produced it.

## Audit generated code

1. Framework operations became `_ivy_runtime.call` or
   `_ivy_runtime.method` calls.
2. No unexpected global helper from the source application remains.
3. Dtype constants and axis conventions match the parity test.
4. Random operations have the intended key/seed behavior.
5. Inputs and results remain native to the selected target.

## Attach a support bundle

```bash
ivy doctor > ivy-doctor.json
ivy coverage --source torch --target jax > ivy-coverage.json
```

Also attach `report.to_json()`, the emitted source/manifest pair, and a small
input generator. Do not attach a proprietary checkpoint when a synthetic
array can reproduce the problem.
