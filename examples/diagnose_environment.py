"""Print a support-friendly environment report."""

from __future__ import annotations

import json

from ivy.cli import doctor

if __name__ == "__main__":
    print(json.dumps(doctor(), indent=2, sort_keys=True))
