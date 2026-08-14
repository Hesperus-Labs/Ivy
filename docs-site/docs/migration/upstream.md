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

## Migration procedure

1. Create a Python 3.12/3.13 uv environment and pin a Hesperus commit.
2. Replace legacy builds with `pyproject.toml`/`uv.lock` commands.
3. Replace `to=` with `target=` and `output_dir=` with `emit=`.
4. Isolate file-defined callable boundaries; remove private graph/compiler use.
5. Convert modules to Equinox and make keys/state explicit.
6. Use focused `tests/` plus application parity tests.
7. Build `docs-site/` strictly; historical RST is not published.

## Behavior not carried forward

Hesperus does not promise the complete historical experimental API, private
compiler packages, generated test databases, legacy automation, Flax/Haiku
adapters, or old docs deployment. Retained files are reference material until
explicitly migrated into the maintained contract.

Compatibility aliases enable staged migration but should not become new
dependencies. Run with deprecation warnings visible and validate actual
application shapes, dtypes, state, and randomness.
