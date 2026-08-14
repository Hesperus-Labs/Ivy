"""Equinox-native module conversion helpers.

These helpers intentionally keep state explicit.  They are lightweight bridge
functions around the public Equinox APIs and do not import Flax or Haiku.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ivy._version import __version__


def _module_parameters(module: Any) -> tuple[tuple[tuple[str, ...], ...], tuple[Any, ...]]:
    if hasattr(module, "named_parameters"):
        items = tuple(module.named_parameters())
        return tuple(tuple(name.split(".")) for name, _ in items), tuple(
            value for _, value in items
        )
    values = tuple(
        ((name,), value)
        for name, value in vars(module).items()
        if not name.startswith("_") and hasattr(value, "shape") and hasattr(value, "dtype")
    )
    return tuple(path for path, _ in values), tuple(value for _, value in values)


def _set_module_path(module: Any, path: tuple[str, ...], value: Any) -> None:
    current = module
    for name in path[:-1]:
        current = getattr(current, name)
    setattr(current, path[-1], value)


def _require_equinox():
    try:
        import equinox as eqx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Equinox is required for this operation. Install the JAX extra with "
            "`uv pip install 'hesperus-ivy[jax]'`."
        ) from exc
    return eqx


def from_equinox_module(module: Any, *, state: Any | None = None):
    """Validate an Equinox module and return it with optional explicit state.

    A stateless conversion returns the original ``eqx.Module``.  If a state
    object is supplied, the result is ``(module, state)`` so callers cannot
    accidentally hide mutable state in a global variable.
    """

    eqx = _require_equinox()
    if not isinstance(module, eqx.Module):
        raise TypeError(f"Expected an Equinox Module, received {type(module)!r}")
    return (module, state) if state is not None else module


def to_equinox_module(
    native_module: Any | Callable[..., Any],
    *,
    source: str | None = None,
    state: Any | None = None,
    inference: bool = False,
):
    """Convert a callable/module into a small native Equinox wrapper.

    Parameterized framework modules are transpiled through their ``forward`` or
    ``__call__`` source.  The wrapper itself is an ``eqx.Module`` and therefore
    composes with ``eqx.filter_jit`` and ``eqx.filter_value_and_grad``.  Complex
    framework-specific parameter layouts should use a generated adapter and
    will report an explicit conversion error rather than silently sharing
    source-framework tensors.
    """

    eqx = _require_equinox()
    import ivy
    from ivy.transpiler.api import _as_target_array, _ModuleAdapter, _TargetModuleProxy

    if isinstance(native_module, eqx.Module):
        converted = native_module
    else:
        callable_obj = (
            native_module.forward
            if hasattr(native_module, "forward")
            else native_module
        )
        converted_callable = ivy.transpile(
            callable_obj,
            source=source,
            target="equinox",
        )
        if isinstance(converted_callable, _ModuleAdapter):
            parameter_paths, parameter_values = _module_parameters(native_module)
        else:
            parameter_paths, parameter_values = (), ()

        if parameter_values:

            class Converted(eqx.Module):
                fn: Callable[..., Any] = eqx.field(static=True)
                template: Any = eqx.field(static=True)
                paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
                parameters: tuple[Any, ...]
                stateful: bool = eqx.field(static=True)

                def __call__(self, *args: Any, **kwargs: Any) -> Any:
                    current_state = kwargs.pop("state", None)
                    proxy = _TargetModuleProxy(self.template, "equinox")
                    for path, value in zip(self.paths, self.parameters, strict=True):
                        _set_module_path(proxy, path, value)
                    output = self.fn(proxy, *args, **kwargs)
                    if self.stateful:
                        if current_state is None:
                            raise TypeError(
                                "state= is required for a stateful Equinox module"
                            )
                        return output, current_state
                    return output

            converted = Converted(
                converted_callable.forward,
                native_module,
                parameter_paths,
                tuple(_as_target_array(value, "equinox") for value in parameter_values),
                state is not None,
            )
        else:

            class Converted(eqx.Module):
                fn: Callable[..., Any] = eqx.field(static=True)
                stateful: bool = eqx.field(static=True)

                def __call__(self, *args: Any, **kwargs: Any) -> Any:
                    current_state = kwargs.pop("state", None)
                    output = self.fn(*args, **kwargs)
                    if self.stateful:
                        if current_state is None:
                            raise TypeError(
                                "state= is required for a stateful Equinox module"
                            )
                        return output, current_state
                    return output

            # A bound method with no discoverable array attributes still needs
            # to be an Equinox module.  Keep the adapter as a static callable
            # rather than returning a framework-specific object.
            converted = Converted(converted_callable, state is not None)
    if inference:
        converted = eqx.nn.inference_mode(converted, value=True)
    return (converted, state) if state is not None else converted


def save_equinox(module: Any, path: str | Path) -> None:
    """Serialize Equinox array leaves using its native tree serializer."""

    eqx = _require_equinox()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(destination), module)
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(
            {
                "format": "equinox.tree_serialise_leaves",
                "package": "hesperus-ivy",
                "version": __version__,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_equinox(path: str | Path, like: Any) -> Any:
    """Restore Equinox leaves into a structure created by the caller."""

    eqx = _require_equinox()
    return eqx.tree_deserialise_leaves(str(path), like)
