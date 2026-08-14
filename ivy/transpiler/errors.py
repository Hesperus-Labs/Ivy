"""Exceptions raised by the Hesperus Ivy transpiler.

The transpiler deliberately reports semantic gaps instead of silently executing
the source framework.  This makes a conversion failure actionable and keeps a
successful conversion honest about which operations were lowered.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class TranspileError(RuntimeError):
    """Base class for all conversion failures."""

    def __init__(self, message: str, *, report: Any | None = None) -> None:
        super().__init__(message)
        self.report = report


class UnsupportedSourceError(TranspileError):
    """Raised when a source framework is not part of the maintained matrix."""


class UnsupportedTargetError(TranspileError):
    """Raised when a requested target is not part of the maintained matrix."""


class SourceUnavailableError(TranspileError):
    """Raised when source-to-source conversion cannot inspect an object."""


class UnsupportedObjectError(TranspileError):
    """Raised for object shapes that need an explicit adapter."""


class UnsupportedPrimitiveError(TranspileError):
    """Raised when a primitive has no registered lowering."""

    def __init__(
        self,
        primitive: str,
        *,
        target: str,
        report: Any | None = None,
        hints: Mapping[str, str] | None = None,
    ) -> None:
        hint = ""
        if hints:
            hint = " Suggested alternatives: " + ", ".join(
                f"{name} ({value})" for name, value in hints.items()
            )
        super().__init__(
            f"No lowering is registered for {primitive!r} when targeting "
            f"{target!r}.{hint}",
            report=report,
        )
        self.primitive = primitive
        self.target = target
