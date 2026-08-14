"""Structured metadata returned by a transpilation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TranspileReport:
    """A reproducible record of how an object was converted.

    Reports are intentionally JSON serialisable so they can be attached to a
    build artifact or embedded in a generated source directory.
    """

    object_name: str
    source: str
    target: str
    mode: str
    cache_hit: bool = False
    source_available: bool = True
    emitted_path: str | None = None
    converted_primitives: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    framework_versions: Mapping[str, str] = field(default_factory=dict)
    registry_revision: str = "2026.08"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""

        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Return a formatted JSON representation."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class TranspileResult:
    """Value/report pair returned by ``return_report=True``."""

    value: Any
    report: TranspileReport

    def __iter__(self):
        yield self.value
        yield self.report
