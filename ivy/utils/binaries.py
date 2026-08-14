"""Legacy binary hooks retained as safe no-ops.

Older Ivy releases downloaded private ``.so`` files from a GitHub repository at
install time.  Hesperus Ivy ships its transpiler as Python source, so importing
the package must never inspect the network or mutate the checkout.
"""

from __future__ import annotations

import warnings


def check_for_binaries() -> None:
    """Return without checking for the retired private compiler binaries."""

    return None


def cleanup_and_fetch_binaries(clean: bool = True) -> None:
    """Explain the replacement for the removed binary download workflow."""

    del clean
    warnings.warn(
        "Hesperus Ivy no longer downloads compiler binaries; the transpiler is "
        "included as Python source. No action is required.",
        DeprecationWarning,
        stacklevel=2,
    )
