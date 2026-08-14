"""Native module conversion helpers.

Hesperus Ivy keeps the historical Keras/Paddle/PyTorch adapters for users of
the stateful Ivy API.  JAX module conversion is intentionally Equinox-native;
the retired Flax/Haiku entry points remain as migration errors so an old
application fails loudly with the new explicit-state API.
"""

from __future__ import annotations

import inspect
from typing import Any

import ivy
from ivy.utils.backend import current_backend


def to_ivy_module(
    native_module=None,
    native_module_class=None,
    args=None,
    kwargs=None,
    device=None,
    devices=None,
    inplace_update=False,
):
    """Convert a maintained native module through the active Ivy backend."""

    return current_backend().to_ivy_module(
        native_module,
        native_module_class,
        args,
        kwargs,
        device,
        devices,
        inplace_update,
    )


class ModuleConverters:
    """Compatibility namespace for native module conversion functions."""

    @staticmethod
    def from_haiku_module(*_args: Any, **_kwargs: Any):
        raise NotImplementedError(
            "The legacy Haiku adapter was retired in Hesperus Ivy 2.0. "
            "Use ivy.to_equinox_module with explicit state and key arguments."
        )

    @staticmethod
    def from_flax_module(*_args: Any, **_kwargs: Any):
        raise NotImplementedError(
            "The legacy Flax adapter was retired in Hesperus Ivy 2.0. "
            "Use ivy.to_equinox_module with an explicit (module, state) contract."
        )

    @staticmethod
    def from_keras_module(
        native_module=None,
        constructor_args: list[Any] | None = None,
        constructor_kwargs: dict[str, Any] | None = None,
        instance_args: list[Any] | None = None,
        instance_kwargs: dict[str, Any] | None = None,
        device=None,
        devices=None,
    ):
        """Convert a Keras module instance to an Ivy module instance."""

        c_args = ivy.default(constructor_args, [])
        c_kwargs = ivy.default(constructor_kwargs, {})
        i_args = ivy.default(instance_args, [])
        i_kwargs = ivy.default(instance_kwargs, {})
        if inspect.isclass(native_module):
            if not i_args and not i_kwargs:
                raise ivy.utils.exceptions.IvyException(
                    "instance_args or instance_kwargs are required for a native class"
                )
            native_module = native_module(*c_args, **c_kwargs)
            native_module.build((i_args[0].shape[-1],))
        from ivy.stateful.module import _KerasIvyModule

        return _KerasIvyModule(
            *i_args,
            native_module=native_module,
            device=device,
            devices=devices,
            **i_kwargs,
        )

    @staticmethod
    def from_paddle_module(
        native_module=None,
        constructor_args: list[Any] | None = None,
        constructor_kwargs: dict[str, Any] | None = None,
        instance_args: list[Any] | None = None,
        instance_kwargs: dict[str, Any] | None = None,
        device=None,
        devices=None,
    ):
        """Convert a Paddle layer instance to an Ivy module instance."""

        c_args = ivy.default(constructor_args, [])
        c_kwargs = ivy.default(constructor_kwargs, {})
        i_args = ivy.default(instance_args, [])
        i_kwargs = ivy.default(instance_kwargs, {})
        if inspect.isclass(native_module):
            native_module = native_module(*c_args, **c_kwargs)
        from ivy.stateful.module import _PaddleIvyModule

        return _PaddleIvyModule(
            *i_args,
            native_module=native_module,
            device=device,
            devices=devices,
            **i_kwargs,
        )

    @staticmethod
    def from_torch_module(
        native_module=None,
        constructor_args: list[Any] | None = None,
        constructor_kwargs: dict[str, Any] | None = None,
        instance_args: list[Any] | None = None,
        instance_kwargs: dict[str, Any] | None = None,
        device=None,
        devices=None,
        inplace_update=False,
    ):
        """Convert a PyTorch module instance to an Ivy module instance."""

        try:
            import torch  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Install the PyTorch extra before converting a PyTorch module."
            ) from exc
        c_args = ivy.default(constructor_args, [])
        c_kwargs = ivy.default(constructor_kwargs, {})
        i_args = ivy.default(instance_args, [])
        i_kwargs = ivy.default(instance_kwargs, {})
        if inspect.isclass(native_module):
            native_module = native_module(*c_args, **c_kwargs)
        from ivy.stateful.module import _TorchIvyModule

        return _TorchIvyModule(
            *i_args,
            native_module=native_module,
            device=device,
            devices=devices,
            inplace_update=inplace_update,
            **i_kwargs,
        )
