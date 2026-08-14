# Internals: cache format

`ivy.transpiler.cache.SourceCache` stores generated Python and JSON metadata in
a user-scoped directory. It is an optimization and audit aid, not an object
serialization system.

## Directory selection

Resolution order:

1. `HESPERUS_IVY_CACHE_DIR` when set;
2. `platformdirs.user_cache_dir("hesperus-ivy", "Hesperus Labs")`;
3. a `~/.cache/hesperus-ivy` fallback for minimal source checkouts.

The directory is created when requested. Applications needing read-only roots
should point the variable at a writable runtime directory or use `cache=False`.

## Key construction

`cache_key` hashes length-prefixed representations with SHA-256. Length
prefixing avoids ambiguity between adjacent parts. Callable conversion includes
generated source, framework pair, compile option, registry revision, and object
identity text.

## Entry layout

For key `<sha256>`:

```text
<sha256>.py
<sha256>.json
<sha256>.lock
```

The JSON must decode to an object; corrupt or incomplete entries are treated as
misses. The cache does not execute content merely to decide whether it is valid.

## Atomic writes

A `filelock.FileLock` protects each key when available. Content is written to a
temporary file in the same directory, flushed, synced, then moved into place
with `os.replace`. Same-filesystem replacement prevents readers from observing
partial content.

## Clearing

`clear_cache` removes only `.py`, `.json`, and `.lock` entries within this
resolved cache root and returns the count removed. It does not recursively
delete arbitrary directories.

## Security model

The cache contains executable generated Python. It assumes the user cache
directory is trusted like the installed application code. It never unpickles
objects, never downloads compiler binaries, and never writes into a caller's
source file. Do not share a writable cache across mutually untrusted users.
