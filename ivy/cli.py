"""Command-line diagnostics for Hesperus Ivy."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from typing import Any


def _installed(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def doctor() -> dict[str, Any]:
    """Collect environment and optional-backend information."""

    result: dict[str, Any] = {
        "package": "hesperus-ivy",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "frameworks": {
            name: _installed(name)
            for name in ("numpy", "torch", "tensorflow", "jax", "equinox", "optax")
        },
        "cuda": {"torch_available": False, "jax_devices": []},
    }
    try:
        import torch

        result["cuda"]["torch_available"] = bool(torch.cuda.is_available())
        result["cuda"]["torch_device_count"] = int(torch.cuda.device_count())
        if result["cuda"]["torch_device_count"]:
            result["cuda"]["torch_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # optional backend diagnostics must never crash
        result["cuda"]["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import jax

        result["cuda"]["jax_devices"] = [str(device) for device in jax.devices()]
    except Exception as exc:
        result["cuda"]["jax_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import pynvml

        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        result["cuda"]["nvidia_driver"] = (
            driver.decode() if isinstance(driver, bytes) else str(driver)
        )
        result["cuda"]["nvidia_devices"] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            result["cuda"]["nvidia_devices"].append(
                {
                    "index": index,
                    "name": name.decode() if isinstance(name, bytes) else str(name),
                    "memory_total": int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
                }
            )
        pynvml.nvmlShutdown()
    except Exception as exc:  # optional NVIDIA diagnostics must never crash
        result["cuda"]["nvidia_error"] = f"{type(exc).__name__}: {exc}"
    return result


def doctor_main() -> int:
    """Print the diagnostics document and return a shell status."""

    print(json.dumps(doctor(), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the ``ivy`` command-line interface."""

    parser = argparse.ArgumentParser(prog="ivy", description="Hesperus Ivy tools")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="show framework and accelerator diagnostics")
    subparsers.add_parser("cache-info", help="show the transpilation cache location")
    subparsers.add_parser("cache-clear", help="clear generated transpilation source")
    coverage_parser = subparsers.add_parser("coverage", help="show primitive coverage")
    coverage_parser.add_argument("--source")
    coverage_parser.add_argument("--target")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor_main()
    from .transpiler import cache_info, clear_cache, compatibility_report

    if args.command == "cache-info":
        print(json.dumps(cache_info(), indent=2, sort_keys=True))
    elif args.command == "cache-clear":
        print(f"removed {clear_cache()} cache files")
    elif args.command == "coverage":
        print(json.dumps(compatibility_report(source=args.source, target=args.target), indent=2))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
