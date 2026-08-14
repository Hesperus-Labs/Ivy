# Documentation locations

The published Hesperus Ivy documentation is the MkDocs site in
[`docs-site/`](../docs-site/) and is deployed to
<https://hesperus-labs.github.io/Ivy/> by
[`.github/workflows/docs-pages.yml`](../.github/workflows/docs-pages.yml).

All current user-facing pages, API references, migration guides, tutorials,
and internals documentation belong in [`docs-site/docs/`](../docs-site/docs/).
Build them locally with:

```bash
uv sync --python 3.13 --group docs
uv run --python 3.13 --group docs mkdocs build --strict
```

The remaining RST files under this directory are a historical upstream Ivy
archive. They are intentionally not included in the Pages build and should not
be edited for new Hesperus behavior. If an archived page is still useful,
rewrite it as a current Markdown page under `docs-site/docs/` and add it to
`mkdocs.yml`.
