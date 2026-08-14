"""Compatibility facade for the Hesperus Ivy transpiler.

The historical fork loaded a private ``_compiler.so`` downloaded by
``setup.py``.  The public facade now delegates to the checked-in Python
implementation and keeps the old names available to existing applications.
"""

from ivy.transpiler.api import (
    cache_info,
    clear_cache,
    compatibility_report,
    trace_graph,
    transpile,
    unify,
)
from ivy.transpiler.registry import PRIMITIVES, REGISTRY_REVISION, PrimitiveSpec

__all__ = [
    "cache_info",
    "clear_cache",
    "compatibility_report",
    "trace_graph",
    "transpile",
    "unify",
    "PRIMITIVES",
    "REGISTRY_REVISION",
    "PrimitiveSpec",
]
