# Migrate from upstream Ivy

The import name remains `ivy`, but the distribution is now
`hesperus-ivy`. Install from the Hesperus Labs Git repository and update
`ivy.transpile(..., to=...)` calls to the explicit `target=` spelling.

| Upstream pattern | Hesperus Ivy |
| --- | --- |
| `to="jax"` | `target="equinox"` for modules, `target="jax"` for functions |
| binary compiler download | checked-in Python transpiler |
| Flax/Haiku module bridge | Equinox module + explicit state |
| hidden framework imports | lazy target-only imports |
| implicit global cache | user-scoped, content-addressed cache |

`to=`, `output_dir=`, and the legacy graph names remain compatibility aliases
for the migration cycle. Their behavior is captured in the report and emits a
deprecation warning where semantics changed.
