"""Small, safe, user-scoped cache for generated transpiler source."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from filelock import FileLock
except ImportError:  # pragma: no cover - only used in minimal source checkouts
    FileLock = None  # type: ignore[assignment,misc]

try:
    from platformdirs import user_cache_dir
except ImportError:  # pragma: no cover
    user_cache_dir = None  # type: ignore[assignment]


def cache_root() -> Path:
    """Return the cache directory, honoring ``HESPERUS_IVY_CACHE_DIR``."""

    configured = os.environ.get("HESPERUS_IVY_CACHE_DIR")
    if configured:
        root = Path(configured).expanduser()
    elif user_cache_dir is not None:
        root = Path(user_cache_dir("hesperus-ivy", "Hesperus Labs"))
    else:
        root = Path.home() / ".cache" / "hesperus-ivy"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_key(*parts: Any) -> str:
    """Create a stable SHA-256 key from conversion inputs."""

    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            data = part
        else:
            data = repr(part).encode("utf-8", "backslashreplace")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


class SourceCache:
    """Persist generated Python and metadata with atomic replacement."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or cache_root()

    def _paths(self, key: str) -> tuple[Path, Path, Path]:
        return (
            self.root / f"{key}.py",
            self.root / f"{key}.json",
            self.root / f"{key}.lock",
        )

    def load(self, key: str) -> tuple[str, Mapping[str, Any]] | None:
        """Load a valid source/metadata pair, or return ``None``."""

        source_path, metadata_path, _ = self._paths(key)
        if not source_path.is_file() or not metadata_path.is_file():
            return None
        try:
            source = source_path.read_text(encoding="utf-8")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(metadata, dict):
            return None
        return source, metadata

    def save(self, key: str, source: str, metadata: Mapping[str, Any]) -> None:
        """Atomically write a generated source/metadata pair."""

        source_path, metadata_path, lock_path = self._paths(key)
        lock = FileLock(str(lock_path)) if FileLock is not None else None
        context = lock if lock is not None else _NullContext()
        with context:
            for destination, content in (
                (source_path, source),
                (metadata_path, json.dumps(dict(metadata), indent=2, sort_keys=True)),
            ):
                fd, temporary = tempfile.mkstemp(
                    prefix=f".{destination.name}.", dir=self.root
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, destination)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)

    def clear(self) -> int:
        """Remove generated cache entries and return the number removed."""

        count = 0
        for path in self.root.glob("*.*"):
            if path.suffix in {".py", ".json", ".lock"}:
                try:
                    path.unlink()
                except OSError:
                    continue
                count += 1
        return count


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False
