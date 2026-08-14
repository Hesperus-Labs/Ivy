"""Target-native operation runtime used by generated functions.

The source-to-source layer rewrites framework-qualified calls to this small
dispatcher.  Optional frameworks are imported lazily, and every branch keeps
the result in the selected target's array type.  The dispatcher deliberately
handles the stable tensor/neural-network core; framework-internal operations
raise an actionable error instead of silently converting through NumPy.
"""

from __future__ import annotations

from typing import Any

from .errors import UnsupportedPrimitiveError


def _canonical_target(target: str) -> str:
    target = target.lower()
    aliases = {
        "tf": "tensorflow",
        "keras": "tensorflow",
        "pytorch": "torch",
        "jnp": "jax",
        "equinox": "jax",
        "np": "numpy",
        "ivy": "numpy",
    }
    target = aliases.get(target, target)
    if target not in {"jax", "torch", "tensorflow", "numpy"}:
        raise ValueError(f"Unsupported target runtime: {target!r}")
    return target


def _modules(target: str):
    target = _canonical_target(target)
    if target == "numpy":
        import numpy as xp

        return xp, None, None
    if target == "jax":
        import jax
        import jax.numpy as xp

        return xp, jax.nn, jax.random
    if target == "torch":
        import torch

        return torch, torch.nn.functional, None
    if target == "tensorflow":
        import tensorflow as tf

        return tf, tf.nn, None
    raise AssertionError(target)


def _pop_axis(kwargs: dict[str, Any]) -> Any:
    axis = kwargs.pop("axis", None)
    if axis is None:
        axis = kwargs.pop("dim", None)
    if axis is None:
        axis = kwargs.pop("axes", None)
    return axis


def _pop_keepdims(kwargs: dict[str, Any]) -> bool:
    value = kwargs.pop("keepdims", None)
    if value is None:
        value = kwargs.pop("keepdim", False)
    return bool(value)


def _shape_from_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Normalize ``(2, 3)``, ``((2, 3),)`` and list shapes."""

    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return tuple(args[0])
    return tuple(args)


def _host_value(value: Any) -> Any:
    """Return a host value when a foreign framework exposes ``.numpy()``."""

    try:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        if hasattr(value, "numpy") and not isinstance(value, (str, bytes)):
            return value.numpy()
    except (AttributeError, TypeError, ValueError):
        pass
    return value


def _seed_value(key: Any) -> int:
    """Derive a stable integer seed from a JAX-style key or Python integer."""

    try:
        import numpy as np

        array = np.asarray(_host_value(key), dtype=np.uint64).reshape(-1)
        if array.size:
            # Mix both uint32 halves without depending on a framework private
            # key representation.  This is deterministic, not a promise of
            # bit-for-bit distribution equivalence across frameworks.
            seed = int(array[0])
            for item in array[1:]:
                seed = ((seed * 0x9E3779B1) ^ int(item)) & 0xFFFFFFFF
            return seed
    except (TypeError, ValueError, OverflowError):
        pass
    return int(key) if isinstance(key, (int, bool)) else 0


def _jax_key(kwargs: dict[str, Any], random_module: Any):
    key = kwargs.pop("key", None)
    if key is None:
        key = kwargs.pop("rng", None)
    if key is None:
        raise TypeError(
            "JAX/Equinox random operations require an explicit key= argument"
        )
    return key, random_module


def dtype(value: Any, *, target: str) -> Any:
    """Resolve a source dtype spelling to the selected target dtype.

    Generated functions commonly contain ``torch.float32`` or ``tf.int64``
    constants.  Imports are removed during lowering, so these constants are
    rewritten to this helper rather than leaking a source framework import.
    """

    target = _canonical_target(target)
    name = str(value).rsplit(".", 1)[-1].lower()
    aliases = {
        "float": "float32",
        "double": "float64",
        "half": "float16",
        "long": "int64",
        "int": "int32",
        "bool": "bool",
        "complex": "complex64",
    }
    name = aliases.get(name, name)
    if target == "torch":
        import torch

        return getattr(torch, name)
    if target == "tensorflow":
        import tensorflow as tf

        return getattr(tf, name)
    if target == "jax":
        import jax.numpy as jnp

        return getattr(jnp, name)
    import numpy as np

    return getattr(np, name)


def _random_call(
    name: str,
    op: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    target: str,
    xp: Any,
    random_module: Any,
) -> Any:
    """Lower common random APIs, requiring explicit keys for JAX targets."""

    source_is_jax_random = op.startswith("jax.random")
    key = args[0] if source_is_jax_random and args else None
    values = args[1:] if key is not None else args
    if key is None:
        key = kwargs.pop("key", kwargs.pop("rng", None))
    if key is None and "seed" in kwargs:
        key = kwargs.pop("seed")

    if name in {"key", "prngkey"}:
        seed = args[0] if args else kwargs.pop("seed", 0)
        if target == "jax":
            return random_module.key(seed) if name == "key" else random_module.PRNGKey(seed)
        return _seed_value(seed)

    if name == "split" and source_is_jax_random:
        if target == "jax":
            if key is None:
                raise TypeError("jax.random.split requires an explicit key")
            return random_module.split(key, *(values or (2,)), **kwargs)
        count = int(values[0]) if values else int(kwargs.pop("num", 2))
        return tuple(_seed_value(key) + index + 1 for index in range(count))

    if target == "jax":
        if key is None:
            raise TypeError(
                "JAX/Equinox random operations require an explicit key= argument"
            )
        if isinstance(key, (int, bool)):
            key = random_module.PRNGKey(key)
        if name in {"randint", "integers"}:
            shape = values[0] if values else kwargs.pop("shape", ())
            low = values[1] if len(values) > 1 else kwargs.pop("minval", 0)
            high = values[2] if len(values) > 2 else kwargs.pop("maxval", None)
            if high is None:
                high, low = low, 0
            return random_module.randint(key, shape, low, high, **kwargs)
        shape_args = values[:1] if len(values) == 1 else values
        shape = _shape_from_args(shape_args) if shape_args else tuple(kwargs.pop("shape", ()))
        mean = kwargs.pop("mean", 0)
        std = kwargs.pop("std", 1)
        if name in {"randn", "normal", "standard_normal"}:
            result = random_module.normal(key, shape=shape, **kwargs)
        elif name in {"rand", "uniform"}:
            result = random_module.uniform(key, shape=shape, **kwargs)
        else:
            raise UnsupportedPrimitiveError(op, target=target)
        return result * std + mean

    seed = _seed_value(key) if key is not None else None
    if target == "torch":
        if name in {"randint", "integers"}:
            if source_is_jax_random:
                shape = values[0] if values else kwargs.pop("shape", ())
                low = values[1] if len(values) > 1 else kwargs.pop("minval", 0)
                high = values[2] if len(values) > 2 else kwargs.pop("maxval", None)
            else:
                low = values[0] if values else kwargs.pop("low", 0)
                high = values[1] if len(values) > 1 else kwargs.pop("high", None)
                shape = values[2] if len(values) > 2 else kwargs.pop("size", kwargs.pop("shape", ()))
            if high is None:
                high, low = low, 0
            return xp.randint(low, high, _shape_from_args((shape,)), **kwargs)
        if name in {"randn", "normal", "standard_normal"}:
            shape_args = values[:1] if len(values) == 1 else values
            shape = _shape_from_args(shape_args) if shape_args else tuple(kwargs.pop("shape", ()))
            mean = kwargs.pop("mean", 0)
            std = kwargs.pop("std", 1)
            generator = None
            if seed is not None:
                generator = xp.Generator(device="cpu").manual_seed(seed)
            result = xp.randn(*shape, generator=generator, **kwargs)
            return result * std + mean
        if name in {"rand", "uniform"}:
            shape_args = values[:1] if len(values) == 1 else values
            shape = _shape_from_args(shape_args) if shape_args else tuple(kwargs.pop("shape", ()))
            generator = None
            if seed is not None:
                generator = xp.Generator(device="cpu").manual_seed(seed)
            return xp.rand(*shape, generator=generator, **kwargs)
    elif target == "tensorflow":
        if name in {"randint", "integers"}:
            if source_is_jax_random:
                shape = values[0] if values else kwargs.pop("shape", ())
                low = values[1] if len(values) > 1 else kwargs.pop("minval", 0)
                high = values[2] if len(values) > 2 else kwargs.pop("maxval", None)
            else:
                low = values[0] if values else kwargs.pop("low", 0)
                high = values[1] if len(values) > 1 else kwargs.pop("high", None)
                shape = values[2] if len(values) > 2 else kwargs.pop("size", kwargs.pop("shape", ()))
            if high is None:
                high, low = low, 0
            if seed is not None:
                stateless_seed = [seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF]
                return xp.random.stateless_uniform(
                    shape,
                    stateless_seed,
                    minval=low,
                    maxval=high,
                    dtype=kwargs.pop("dtype", "int32"),
                )
            return xp.random.uniform(
                shape=shape,
                minval=low,
                maxval=high,
                dtype=kwargs.pop("dtype", "int32"),
            )
        shape_args = values[:1] if len(values) == 1 else values
        shape = _shape_from_args(shape_args) if shape_args else tuple(kwargs.pop("shape", ()))
        if seed is not None:
            stateless_seed = [seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF]
            if name in {"randn", "normal", "standard_normal"}:
                mean = kwargs.pop("mean", 0)
                std = kwargs.pop("std", 1)
                result = xp.random.stateless_normal(shape, stateless_seed, **kwargs)
                return result * std + mean
            if name in {"rand", "uniform"}:
                return xp.random.stateless_uniform(shape, stateless_seed, **kwargs)
        if name in {"randn", "normal", "standard_normal"}:
            mean = kwargs.pop("mean", 0)
            std = kwargs.pop("std", 1)
            result = xp.random.normal(shape=shape, **kwargs)
            return result * std + mean
        if name in {"rand", "uniform"}:
            return xp.random.uniform(shape=shape, **kwargs)
    else:
        import numpy as np

        generator = np.random.default_rng(seed)
        if name in {"randint", "integers"}:
            if source_is_jax_random:
                shape = values[0] if values else kwargs.pop("shape", ())
                low = values[1] if len(values) > 1 else kwargs.pop("minval", 0)
                high = values[2] if len(values) > 2 else kwargs.pop("maxval", None)
            else:
                low = values[0] if values else kwargs.pop("low", 0)
                high = values[1] if len(values) > 1 else kwargs.pop("high", None)
                shape = values[2] if len(values) > 2 else kwargs.pop("size", kwargs.pop("shape", ()))
            if high is None:
                high, low = low, 0
            return generator.integers(low, high, size=shape, **kwargs)
        shape_args = values[:1] if len(values) == 1 else values
        shape = _shape_from_args(shape_args) if shape_args else tuple(kwargs.pop("shape", ()))
        if name in {"randn", "normal", "standard_normal"}:
            return generator.normal(
                loc=kwargs.pop("mean", 0),
                scale=kwargs.pop("std", 1),
                size=shape,
                **kwargs,
            )
        if name in {"rand", "uniform"}:
            return generator.uniform(size=shape, **kwargs)
    raise UnsupportedPrimitiveError(op, target=target)


def _activation(name: str, value: Any, *, target: str, nn: Any, xp: Any, **kwargs: Any) -> Any:
    if target == "numpy":
        if name == "relu":
            return xp.maximum(value, 0)
        if name == "sigmoid":
            return 1 / (1 + xp.exp(-value))
        if name in {"silu", "swish"}:
            return value * (1 / (1 + xp.exp(-value)))
        if name in {"gelu", "mish", "softplus"}:
            if name == "softplus":
                return xp.log1p(xp.exp(value))
            if name == "mish":
                return value * xp.tanh(xp.log1p(xp.exp(value)))
            return 0.5 * value * (1 + xp.tanh((2 / xp.pi) ** 0.5 * (value + 0.044715 * value**3)))
        if name == "leaky_relu":
            return xp.where(value >= 0, value, kwargs.pop("negative_slope", 0.01) * value)
        if name == "hardtanh":
            return xp.clip(value, kwargs.pop("min_val", -1), kwargs.pop("max_val", 1))
        if name == "relu6":
            return xp.clip(value, 0, 6)
        if name in {"sigmoid", "tanh", "exp", "log", "sqrt"}:
            return getattr(xp, name)(value, **kwargs)
    fn = getattr(nn, name, None)
    if fn is None and name == "swish":
        fn = getattr(nn, "silu", None)
    if fn is None:
        raise UnsupportedPrimitiveError(name, target=target)
    return fn(value, **kwargs)


def call(op: str, *args: Any, target: str, **kwargs: Any) -> Any:
    """Execute a registered operation using only target-native primitives."""

    target = _canonical_target(target)
    xp, nn, random_module = _modules(target)
    original = op
    op = op.lower()
    name = op.rsplit(".", 1)[-1]
    name = {
        "reduce_sum": "sum",
        "reduce_mean": "mean",
        "reduce_prod": "prod",
        "reduce_max": "max",
        "reduce_min": "min",
        "reduce_all": "all",
        "reduce_any": "any",
        "reduce_variance": "var",
        "reduce_std": "std",
    }.get(name, name)

    if name in {
        "randn",
        "normal",
        "standard_normal",
        "rand",
        "uniform",
        "randint",
        "integers",
        "key",
        "prngkey",
    } and ("random" in op or name in {"randn", "rand", "randint"}):
        return _random_call(name, op, args, kwargs, target=target, xp=xp, random_module=random_module)

    if name in {"convert_to_tensor", "constant", "as_tensor", "tensor", "array", "asarray"}:
        if target == "torch":
            kwargs.pop("device", None)
            return xp.as_tensor(*args, **kwargs)
        if target == "tensorflow":
            kwargs.pop("device", None)
            return xp.convert_to_tensor(*args, **kwargs)
        kwargs.pop("device", None)
        kwargs.pop("requires_grad", None)
        return xp.asarray(*args, **kwargs)

    if name == "device":
        # Device placement is a target concern.  Preserve it for PyTorch and
        # treat it as metadata elsewhere rather than attempting a host-device
        # transfer during a source conversion.
        return xp.device(*args, **kwargs) if target == "torch" else (args[0] if args else "cpu")

    if name == "cast":
        value, dtype_value = args[0], args[1] if len(args) > 1 else kwargs.pop("dtype", None)
        if target == "tensorflow":
            return xp.cast(value, dtype_value, **kwargs)
        if target == "torch":
            return xp.as_tensor(value, dtype=dtype_value, **kwargs)
        return xp.asarray(value, dtype=dtype_value, **kwargs)

    if name in {"zeros", "ones", "full", "empty", "eye", "arange", "linspace"}:
        kwargs.pop("device", None)
        kwargs.pop("requires_grad", None)
        return getattr(xp, name)(*args, **kwargs)
    if name in {"zeros_like", "ones_like", "full_like", "empty_like"}:
        kwargs.pop("device", None)
        kwargs.pop("requires_grad", None)
        return getattr(xp, name)(*args, **kwargs)

    if name in {"add", "subtract", "sub", "multiply", "mul", "divide", "div", "true_divide", "floor_divide", "remainder", "mod"}:
        fn = {
            "add": "add", "subtract": "subtract", "sub": "subtract",
            "multiply": "multiply", "mul": "multiply", "divide": "divide",
            "div": "divide", "true_divide": "true_divide", "floor_divide": "floor_divide",
            "remainder": "remainder", "mod": "remainder",
        }[name]
        if target == "tensorflow" and fn == "true_divide":
            return xp.math.divide_no_nan(*args, **kwargs)
        return getattr(xp, fn)(*args, **kwargs)

    if name == "divide_no_nan":
        if target == "tensorflow":
            return xp.math.divide_no_nan(*args, **kwargs)
        denominator = args[1]
        return xp.where(denominator == 0, 0, xp.divide(args[0], denominator, **kwargs))

    if name in {"maximum", "minimum", "fmax", "fmin", "logical_and", "logical_or", "logical_not", "logical_xor"}:
        if target == "tensorflow" and name in {"fmax", "fmin"}:
            name = "maximum" if name == "fmax" else "minimum"
        if target == "torch" and name in {"maximum", "minimum", "fmax", "fmin"}:
            left, right = args[0], args[1]
            if not isinstance(right, xp.Tensor):
                right = xp.as_tensor(right, dtype=left.dtype, device=left.device)
            return getattr(xp, name)(left, right, **kwargs)
        return getattr(xp, name)(*args, **kwargs)

    if name in {"matmul", "mm", "bmm", "dot", "tensordot", "einsum", "outer", "inner", "vdot"}:
        if name == "einsum":
            return xp.einsum(*args, **kwargs)
        if name == "tensordot":
            axes = kwargs.pop("axes", args[2] if len(args) > 2 else 2)
            return xp.tensordot(args[0], args[1], axes=axes, **kwargs)
        if name in {"dot", "outer", "inner", "vdot"}:
            if target == "tensorflow":
                axes = 0 if name == "outer" else 1
                return xp.tensordot(args[0], args[1], axes=axes, **kwargs)
            return getattr(xp, name)(*args, **kwargs)
        if target == "torch":
            return getattr(xp, name)(*args, **kwargs)
        return xp.matmul(*args, **kwargs)

    if name in {"linear", "dense"}:
        if target == "torch":
            return nn.linear(*args, **kwargs)
        value, weight = args[0], args[1]
        bias = args[2] if len(args) > 2 else kwargs.pop("bias", None)
        if target == "tensorflow":
            result = xp.linalg.matmul(value, weight, transpose_b=True)
        else:
            result = xp.matmul(value, xp.swapaxes(weight, -1, -2))
        return result if bias is None else result + bias

    if name in {"reshape", "view"}:
        shape = args[1:] if len(args) > 2 else args[1]
        return xp.reshape(args[0], shape, **kwargs)

    if name in {"expand_dims", "unsqueeze"}:
        axis = args[1] if len(args) > 1 else kwargs.pop("axis", 0)
        if target == "torch":
            return xp.unsqueeze(args[0], dim=axis, **kwargs)
        return xp.expand_dims(args[0], axis=axis, **kwargs)

    if name in {"transpose", "permute"}:
        value = args[0]
        axes = args[1:]
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if target == "torch":
            if name == "transpose" and len(axes) >= 2:
                return xp.transpose(value, axes[0], axes[1], **kwargs)
            return value.permute(*axes) if axes else value.transpose(-2, -1)
        if name == "transpose" and op.startswith("torch.") and len(axes) == 2:
            # ``torch.transpose(x, dim0, dim1)`` swaps two dimensions; this
            # differs from NumPy/JAX's full-permutation ``transpose``.
            if target == "tensorflow":
                rank = value.shape.rank
                if rank is None:
                    raise UnsupportedPrimitiveError(
                        "torch.transpose(dynamic-rank)", target=target
                    )
                permutation = list(range(rank))
                permutation[axes[0]], permutation[axes[1]] = (
                    permutation[axes[1]],
                    permutation[axes[0]],
                )
                return xp.transpose(value, perm=permutation, **kwargs)
            return xp.swapaxes(value, axes[0], axes[1], **kwargs)
        if target == "tensorflow":
            return xp.transpose(value, perm=axes or None, **kwargs)
        return xp.transpose(value, axes=axes or None, **kwargs)

    if name in {"flatten"}:
        start = kwargs.pop("start_dim", kwargs.pop("axis", args[1] if len(args) > 1 else 0))
        end = kwargs.pop("end_dim", args[2] if len(args) > 2 else -1)
        if target == "torch":
            return xp.flatten(args[0], start_dim=start, end_dim=end, **kwargs)
        shape = args[0].shape
        end = len(shape) + end if end < 0 else end
        merged = 1
        for dimension in shape[start : end + 1]:
            merged *= dimension
        return xp.reshape(args[0], shape[:start] + (merged,) + shape[end + 1 :], **kwargs)

    if name in {"sum", "mean", "prod", "max", "min", "all", "any", "var", "std", "logsumexp"}:
        axis = _pop_axis(kwargs)
        keepdims = _pop_keepdims(kwargs)
        if name == "logsumexp":
            if target == "torch":
                return xp.logsumexp(args[0], dim=axis, keepdim=keepdims, **kwargs)
            if target == "tensorflow":
                return xp.reduce_logsumexp(args[0], axis=axis, keepdims=keepdims, **kwargs)
            values = args[0]
            maximum = xp.max(values, axis=axis, keepdims=True)
            result = maximum + xp.log(xp.sum(xp.exp(values - maximum), axis=axis, keepdims=True))
            if not keepdims:
                result = xp.squeeze(result) if axis is None else xp.squeeze(result, axis=axis)
            return result
        if target == "torch":
            if axis is None:
                return getattr(xp, name)(args[0], **kwargs)
            if name in {"max", "min"}:
                result = getattr(xp, name)(args[0], dim=axis, keepdim=keepdims, **kwargs)
                return result.values
            return getattr(xp, name)(args[0], dim=axis, keepdim=keepdims, **kwargs)
        if target == "tensorflow":
            tf_name = {"sum": "reduce_sum", "mean": "reduce_mean", "prod": "reduce_prod", "max": "reduce_max", "min": "reduce_min", "all": "reduce_all", "any": "reduce_any", "var": "math.reduce_variance", "std": "math.reduce_std"}.get(name, name)
            fn = xp
            for part in tf_name.split("."):
                fn = getattr(fn, part)
            return fn(args[0], axis=axis, keepdims=keepdims, **kwargs)
        return getattr(xp, name)(args[0], axis=axis, keepdims=keepdims, **kwargs)

    if name in {"argmax", "argmin"}:
        axis = _pop_axis(kwargs)
        if target == "torch":
            return getattr(xp, name)(args[0], dim=axis, **kwargs)
        if target == "tensorflow":
            return getattr(xp, name)(args[0], axis=axis, **kwargs)
        return getattr(xp, name)(args[0], axis=axis, **kwargs)

    if name in {"relu", "gelu", "silu", "swish", "sigmoid", "tanh", "leaky_relu", "hardtanh", "relu6", "mish", "softplus"}:
        return _activation(name, args[0], target=target, nn=nn, xp=xp, **kwargs)

    if name in {"softmax", "log_softmax"}:
        axis = _pop_axis(kwargs)
        axis = -1 if axis is None else axis
        if target == "torch":
            return getattr(nn, name)(args[0], dim=axis, **kwargs)
        if target == "tensorflow":
            if name == "softmax":
                return nn.softmax(args[0], axis=axis, **kwargs)
            return xp.nn.log_softmax(args[0], axis=axis, **kwargs)
        if target == "jax":
            return getattr(nn, name)(args[0], axis=axis, **kwargs)
        values = args[0] - xp.max(args[0], axis=axis, keepdims=True)
        log_values = values - xp.log(xp.sum(xp.exp(values), axis=axis, keepdims=True))
        return log_values if name == "log_softmax" else xp.exp(log_values)

    if name in {"exp", "log", "sqrt", "sin", "cos", "tan", "abs", "negative", "neg", "floor", "ceil", "round", "sign"}:
        normalized = "negative" if name == "neg" else name
        return getattr(xp, normalized)(*args, **kwargs)

    if name in {"cat", "concat", "concatenate"}:
        axis = _pop_axis(kwargs)
        axis = 0 if axis is None else axis
        if target == "torch":
            return xp.cat(args[0], dim=axis, **kwargs)
        if target == "tensorflow":
            return xp.concat(args[0], axis=axis, **kwargs)
        return xp.concatenate(args[0], axis=axis, **kwargs)

    if name == "stack":
        axis = _pop_axis(kwargs)
        axis = 0 if axis is None else axis
        if target == "torch":
            return xp.stack(args[0], dim=axis, **kwargs)
        return xp.stack(args[0], axis=axis, **kwargs)

    if name in {"repeat", "tile"}:
        repeats = args[1:] if len(args) > 1 else (kwargs.pop("repeats", 1),)
        repeats = repeats[0] if len(repeats) == 1 else repeats
        if target == "torch":
            return args[0].repeat(*repeats) if isinstance(repeats, tuple) else args[0].repeat(repeats)
        return getattr(xp, name)(args[0], repeats, **kwargs)

    if name in {"split", "chunk", "unbind"}:
        axis = _pop_axis(kwargs)
        axis = 0 if axis is None else axis
        sections = args[1] if len(args) > 1 else kwargs.pop("sections", kwargs.pop("chunks", 1))
        if target == "torch":
            return getattr(xp, name)(args[0], sections, dim=axis, **kwargs) if name != "unbind" else xp.unbind(args[0], dim=axis)
        if name == "chunk":
            return xp.array_split(args[0], sections, axis=axis)
        return getattr(xp, name)(args[0], sections, axis=axis, **kwargs)

    if name in {"clip", "clamp"}:
        minimum = kwargs.pop("min", kwargs.pop("a_min", None))
        maximum = kwargs.pop("max", kwargs.pop("a_max", None))
        if len(args) > 1:
            minimum = args[1]
        if len(args) > 2:
            maximum = args[2]
        if target == "torch":
            return xp.clamp(args[0], min=minimum, max=maximum, **kwargs)
        if target == "tensorflow":
            return xp.clip_by_value(args[0], minimum, maximum, **kwargs)
        return xp.clip(args[0], minimum, maximum, **kwargs)

    if name == "where":
        return xp.where(*args, **kwargs)

    if name in {"dropout", "feature_alpha_dropout"}:
        training = kwargs.pop("training", True)
        if not training:
            return args[0]
        probability = kwargs.pop("p", kwargs.pop("rate", 0.5))
        if target == "torch":
            return nn.dropout(args[0], p=probability, training=True, **kwargs)
        if target == "tensorflow":
            return xp.nn.dropout(args[0], rate=probability, **kwargs)
        key, random_module = _jax_key(kwargs, random_module)
        keep = random_module.bernoulli(key, 1 - probability, args[0].shape)
        return args[0] * keep / (1 - probability)

    if name in {"norm", "normalize"}:
        if target == "torch":
            return nn.normalize(args[0], **kwargs) if name == "normalize" else xp.linalg.norm(args[0], **kwargs)
        if target == "tensorflow":
            return xp.linalg.norm(args[0], **kwargs)
        if name == "normalize":
            axis = kwargs.pop("axis", -1)
            eps = kwargs.pop("eps", 1e-12)
            return args[0] / xp.maximum(xp.linalg.norm(args[0], axis=axis, keepdims=True), eps)
        return xp.linalg.norm(args[0], **kwargs)

    if name in {"one_hot", "embedding"}:
        if name == "one_hot":
            depth = args[1] if len(args) > 1 else kwargs.pop("num_classes", kwargs.pop("depth", None))
            if target == "torch":
                return nn.one_hot(args[0].long(), num_classes=depth, **kwargs)
            if target == "tensorflow":
                return xp.one_hot(args[0], depth, **kwargs)
            if target == "numpy":
                dtype_value = kwargs.pop("dtype", None)
                return xp.eye(int(depth), dtype=dtype_value)[xp.asarray(args[0], dtype=int)]
            return nn.one_hot(args[0], depth, **kwargs)
        indices, weight = args[0], args[1]
        if target == "torch":
            return nn.embedding(indices, weight, **kwargs)
        if target == "tensorflow":
            return xp.gather(weight, indices, axis=0, **kwargs)
        return xp.take(weight, indices, axis=0, **kwargs)

    if name in {"isfinite", "isnan", "isinf"}:
        if target == "tensorflow":
            tf_name = {"isfinite": "is_finite", "isnan": "is_nan", "isinf": "is_inf"}[name]
            return getattr(xp.math, tf_name)(*args, **kwargs)
        return getattr(xp, name)(*args, **kwargs)
    if name in {"numel", "size", "ndim", "item"}:
        value = args[0]
        if name == "numel":
            return int(value.numel()) if target == "torch" else int(xp.size(value))
        if name == "size":
            return value.size(*args[1:]) if target == "torch" else value.shape
        if name == "ndim":
            return value.ndim
        return value.item()
    if name in {"stop_gradient", "detach"}:
        if target == "tensorflow":
            return xp.stop_gradient(args[0])
        if target == "jax":
            import jax

            return jax.lax.stop_gradient(args[0])
        if target == "torch":
            return args[0].detach()
        return args[0]

    if name in {"pad"}:
        value = args[0]
        padding = args[1] if len(args) > 1 else kwargs.pop("pad", None)
        if (
            op.startswith("torch.")
            and isinstance(padding, (tuple, list))
            and padding
            and not isinstance(padding[0], (tuple, list))
            and len(padding) % 2 == 0
        ):
            # PyTorch lists pairs from the last dimension toward the first;
            # NumPy/JAX/TensorFlow use one pair per axis in axis order.
            pairs = [tuple(padding[index : index + 2]) for index in range(0, len(padding), 2)]
            if target != "torch":
                padding = tuple(reversed(pairs))
        if target == "torch":
            if isinstance(padding, (tuple, list)) and padding and isinstance(
                padding[0], (tuple, list)
            ):
                # Ivy/NumPy spell padding from the first axis onward, while
                # ``torch.nn.functional.pad`` spells it from the last axis.
                padding = tuple(item for pair in reversed(padding) for item in pair)
            return xp.nn.functional.pad(value, padding, **kwargs)
        return xp.pad(value, padding, **kwargs)

    raise UnsupportedPrimitiveError(original, target=target)


def method(name: str, value: Any, *args: Any, target: str, **kwargs: Any) -> Any:
    """Lower common tensor methods whose signatures differ across frameworks."""

    target = _canonical_target(target)
    name = name.lower()
    if name in {"reshape", "view"}:
        shape = args if len(args) != 1 or not isinstance(args[0], (tuple, list)) else args[0]
        return call("reshape", value, shape, target=target, **kwargs)
    if name == "transpose":
        if target == "torch" and len(args) >= 2:
            return value.transpose(args[0], args[1])
        return call("transpose", value, *args, target=target, **kwargs)
    if name == "permute":
        if target == "torch":
            axes = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
            return value.permute(*axes)
        return call("transpose", value, *args, target=target, **kwargs)
    if name == "flatten":
        start = kwargs.pop("start_dim", args[0] if args else 0)
        end = kwargs.pop("end_dim", args[1] if len(args) > 1 else -1)
        return call("flatten", value, start, end, target=target, **kwargs)
    if name in {"sum", "mean", "max", "min", "prod", "all", "any", "var", "std"}:
        return call(name, value, *args, target=target, **kwargs)
    if name in {"float", "double", "half", "to", "astype", "type"}:
        if target == "torch":
            return getattr(value, name)(*args, **kwargs)
        if name == "to" and (
            kwargs.pop("device", None) is not None
            or (args and isinstance(args[0], str) and args[0].lower().startswith(("cpu", "cuda", "tpu")))
        ):
            return value
        dtype_value = kwargs.pop("dtype", args[0] if args else None)
        if dtype_value is None or isinstance(dtype_value, str):
            dtype_value = dtype(dtype_value or name, target=target)
        if target == "tensorflow":
            import tensorflow as tf

            return tf.cast(value, dtype_value)
        return value.astype(dtype_value)
    if name in {"detach", "contiguous", "clone", "copy"}:
        if target == "torch":
            return getattr(value, name)()
        if name == "copy" and hasattr(value, "copy"):
            return value.copy()
        return value
    if name in {"size", "numel", "ndim"}:
        if name == "ndim":
            return value.ndim
        if name == "numel":
            return int(value.numel()) if target == "torch" else int(value.size)
        if target == "torch":
            return value.size(*args)
        return value.shape if not args else value.shape[args[0]]
    if name == "squeeze":
        axis = args[0] if args else kwargs.pop("axis", None)
        if target == "torch":
            return value.squeeze(axis) if axis is not None else value.squeeze()
        return value.squeeze(axis)
    if name == "unsqueeze":
        axis = args[0] if args else kwargs.pop("axis", 0)
        if target == "torch":
            return value.unsqueeze(axis)
        return value[..., None] if axis == -1 else call("expand_dims", value, axis, target=target)
    if name in {"repeat", "tile", "split", "chunk"}:
        return call(name, value, *args, target=target, **kwargs)
    if name in {"matmul", "mm", "dot", "add", "sub", "mul", "div", "maximum", "minimum", "where", "clamp", "clip"}:
        return call(name, value, *args, target=target, **kwargs)
    if name in {"relu", "relu6", "gelu", "silu", "sigmoid", "tanh", "softmax", "log_softmax", "exp", "log", "sqrt"}:
        return call(name, value, target=target, **kwargs)
    if hasattr(value, name):
        return getattr(value, name)(*args, **kwargs)
    raise UnsupportedPrimitiveError(f"tensor.{name}", target=target)
