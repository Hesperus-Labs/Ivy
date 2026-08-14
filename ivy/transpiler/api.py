"""Public, pure-Python Hesperus Ivy transpiler.

This module intentionally has no eager imports of PyTorch, TensorFlow, JAX or
Equinox.  A conversion imports only its selected target, which makes the base
package useful in small CPU environments and in documentation builds.
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.metadata
import inspect
import json
import textwrap
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import FunctionType
from typing import Any

from . import runtime
from .cache import SourceCache, cache_key
from .errors import (
    SourceUnavailableError,
    TranspileError,
    UnsupportedObjectError,
    UnsupportedSourceError,
    UnsupportedTargetError,
)
from .registry import REGISTRY_REVISION, coverage, primitive_name
from .report import TranspileReport, TranspileResult

SUPPORTED_SOURCES = ("torch", "tensorflow", "jax", "ivy", "numpy")
SUPPORTED_TARGETS = ("torch", "tensorflow", "jax", "equinox", "ivy", "numpy")
_FRAMEWORK_ROOTS = {"torch", "tensorflow", "tf", "jax", "jaxlib", "numpy", "np", "ivy"}
_METHODS = {
    "reshape",
    "view",
    "transpose",
    "permute",
    "flatten",
    "sum",
    "mean",
    "max",
    "min",
    "float",
    "double",
    "half",
    "to",
    "astype",
    "type",
    "detach",
    "contiguous",
    "clone",
    "copy",
    "size",
    "numel",
    "ndim",
    "squeeze",
    "unsqueeze",
    "relu",
    "relu6",
    "gelu",
    "silu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
    "matmul",
    "mm",
    "clamp",
    "clip",
    "repeat",
    "tile",
    "split",
    "chunk",
}
_DTYPE_NAMES = {
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
    "complex64",
    "complex128",
    "float",
    "double",
    "half",
    "long",
    "int",
}


def _canonical_source(source: str | None) -> str:
    if source is None:
        return "ivy"
    source = source.lower()
    if source in {"tf", "tensorflow.keras", "keras"}:
        source = "tensorflow"
    if source == "pytorch":
        source = "torch"
    if source in {"jnp", "jax.numpy", "equinox"}:
        source = "jax"
    if source not in SUPPORTED_SOURCES:
        raise UnsupportedSourceError(
            f"Unsupported source framework {source!r}. "
            f"Supported sources: {', '.join(SUPPORTED_SOURCES)}"
        )
    return source


def _canonical_target(target: str | None, to: str | None) -> str:
    if target is not None and to is not None and target != to:
        raise ValueError("target= and legacy to= specify different targets")
    selected = target if target is not None else to
    selected = (selected or "jax").lower()
    if selected in {"tf", "keras"}:
        selected = "tensorflow"
    if selected == "jnp":
        selected = "jax"
    if selected not in SUPPORTED_TARGETS:
        raise UnsupportedTargetError(
            f"Unsupported target framework {selected!r}. "
            f"Supported targets: {', '.join(SUPPORTED_TARGETS)}"
        )
    return selected


def _framework_from_object(obj: Any) -> str:
    module_name = getattr(obj, "__module__", "") or ""
    if module_name.startswith("torch"):
        return "torch"
    if module_name.startswith(("tensorflow", "keras")):
        return "tensorflow"
    if module_name.startswith(("jax", "equinox")):
        return "jax"
    if module_name.startswith("ivy"):
        return "ivy"
    return "ivy"


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _qualified(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _is_framework_path(path: str, aliases: Mapping[str, str]) -> bool:
    root = path.split(".", 1)[0]
    resolved = aliases.get(root, root)
    return resolved.split(".", 1)[0] in _FRAMEWORK_ROOTS


def _resolve_alias(path: str, aliases: Mapping[str, str]) -> str:
    root, _, remainder = path.partition(".")
    resolved = aliases.get(root, root)
    return f"{resolved}.{remainder}" if remainder else resolved


class _Lowerer(ast.NodeTransformer):
    """Rewrite framework calls and incompatible tensor methods."""

    def __init__(self, *, source: str, globals_dict: Mapping[str, Any]) -> None:
        self.source = source
        self.aliases: dict[str, str] = {
            "torch": "torch",
            "tf": "tensorflow",
            "tensorflow": "tensorflow",
            "jax": "jax",
            "jnp": "jax.numpy",
            "np": "numpy",
            "numpy": "numpy",
            "ivy": "ivy",
        }
        self.primitives: list[str] = []
        for name, value in globals_dict.items():
            module_name = getattr(value, "__name__", "") or ""
            if module_name.startswith("torch"):
                self.aliases[name] = module_name
            elif module_name.startswith(("tensorflow", "keras")):
                self.aliases[name] = "tensorflow" + module_name[len("tensorflow") :]
            elif module_name.startswith("jax") or module_name.startswith("numpy"):
                self.aliases[name] = module_name

    def _runtime_call(self, primitive: str, node: ast.Call) -> ast.Call:
        name = primitive_name(primitive)
        if name not in self.primitives:
            self.primitives.append(name)
        keywords = list(node.keywords)
        keywords.append(ast.keyword(arg="target", value=ast.Name(id="_ivy_target", ctx=ast.Load())))
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_ivy_runtime", ctx=ast.Load()),
                    attr="call",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(primitive), *node.args],
                keywords=keywords,
            ),
            node,
        )

    def _runtime_method(self, method_name: str, node: ast.Call) -> ast.Call:
        if method_name not in self.primitives:
            self.primitives.append(method_name)
        assert isinstance(node.func, ast.Attribute)
        keywords = list(node.keywords)
        keywords.append(ast.keyword(arg="target", value=ast.Name(id="_ivy_target", ctx=ast.Load())))
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_ivy_runtime", ctx=ast.Load()),
                    attr="method",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(method_name), node.func.value, *node.args],
                keywords=keywords,
            ),
            node,
        )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        path = _qualified(node.func)
        if path:
            resolved = _resolve_alias(path, self.aliases)
            if _is_framework_path(path, self.aliases):
                return self._runtime_call(resolved, node)
            if isinstance(node.func, ast.Name):
                alias = self.aliases.get(node.func.id)
                if alias and _is_framework_path(alias, self.aliases):
                    return self._runtime_call(alias, node)
        # The receiver can itself be a generated runtime call, in which case
        # it has no qualified AST path.  Method lowering must still apply.
        if isinstance(node.func, ast.Attribute) and node.func.attr in _METHODS:
            return self._runtime_method(node.func.attr, node)
        return node

    def _dtype_call(self, path: str, node: ast.AST) -> ast.Call:
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_ivy_runtime", ctx=ast.Load()),
                    attr="dtype",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(path)],
                keywords=[
                    ast.keyword(
                        arg="target", value=ast.Name(id="_ivy_target", ctx=ast.Load())
                    )
                ],
            ),
            node,
        )

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)
        path = _qualified(node)
        if path and _is_framework_path(path, self.aliases) and path.rsplit(".", 1)[-1].lower() in _DTYPE_NAMES:
            return self._dtype_call(_resolve_alias(path, self.aliases), node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        alias = self.aliases.get(node.id)
        if alias and alias.rsplit(".", 1)[-1].lower() in _DTYPE_NAMES:
            return self._dtype_call(alias, node)
        return node

    def visit_Import(self, node: ast.Import) -> ast.AST | None:
        framework_import = False
        for alias in node.names:
            if alias.name.split(".", 1)[0] in _FRAMEWORK_ROOTS:
                framework_import = True
                local_name = alias.asname or alias.name.split(".", 1)[0]
                self.aliases[local_name] = alias.name
        if framework_import:
            return None
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        module = node.module or ""
        if module.split(".", 1)[0] in _FRAMEWORK_ROOTS:
            for alias in node.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                self.aliases[local_name] = f"{module}.{alias.name}"
            return None
        return node


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise SourceUnavailableError(f"Could not find function {name!r} in inspected source")


def _prepare_source(obj: FunctionType | Callable[..., Any], source: str) -> tuple[str, tuple[str, ...]]:
    try:
        raw_source = textwrap.dedent(inspect.getsource(obj))
    except (OSError, TypeError) as exc:
        raise SourceUnavailableError(
            f"Source for {getattr(obj, '__qualname__', obj)!r} is unavailable. "
            "Pass a file-defined function or use an explicit trace adapter."
        ) from exc
    tree = ast.parse(raw_source)
    function = _find_function(tree, getattr(obj, "__name__", "<callable>"))
    function.decorator_list = []
    function.returns = None
    for argument in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]:
        argument.annotation = None
    if function.args.vararg:
        function.args.vararg.annotation = None
    if function.args.kwarg:
        function.args.kwarg.annotation = None
    globals_dict = getattr(obj, "__globals__", {})
    lowerer = _Lowerer(source=source, globals_dict=globals_dict)
    function = lowerer.visit(function)
    ast.fix_missing_locations(function)
    generated = ast.unparse(function)
    return generated, tuple(sorted(set(lowerer.primitives)))


def _compile_generated(
    source_code: str,
    *,
    obj: Callable[..., Any],
    target: str,
) -> Callable[..., Any]:
    namespace: dict[str, Any] = dict(getattr(obj, "__globals__", {}))
    namespace.update({"_ivy_runtime": runtime, "_ivy_target": target})
    namespace.pop("__builtins__", None)
    exec(compile(source_code, "<ivy-transpiled>", "exec"), namespace, namespace)
    generated = namespace.get(getattr(obj, "__name__", "<callable>"))
    if not callable(generated):
        raise TranspileError("Generated source did not define a callable")
    functools.update_wrapper(generated, obj)
    generated.__ivy_transpiled_target__ = target
    generated.__ivy_transpiled_source__ = source_code
    return generated


def _compile_backend(fn: Callable[..., Any], target: str, enabled: bool) -> Callable[..., Any]:
    if not enabled:
        return fn
    if target == "torch":
        import torch

        return torch.compile(fn)
    if target == "tensorflow":
        import tensorflow as tf

        return tf.function(fn)
    if target in {"jax", "equinox"}:
        import jax

        return jax.jit(fn)
    return fn


@dataclass
class _ModuleAdapter:
    """Callable adapter for simple framework modules with inspectable forward."""

    source_module: Any
    forward: Callable[..., Any]
    target: str
    target_module: Any = None

    def __post_init__(self) -> None:
        self.target_module = _TargetModuleProxy(self.source_module, self.target)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(self.target_module, *args, **kwargs)


def _as_target_array(value: Any, target: str) -> Any:
    """Convert common framework tensor/parameter objects to target arrays."""

    # Convert through a host NumPy view before constructing a target tensor.
    # This is important for PyTorch -> TensorFlow and JAX -> PyTorch bridges:
    # each framework deliberately rejects some of the other frameworks' array
    # protocols instead of silently copying them.
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "numpy") and not isinstance(value, (str, bytes)):
            value = value.numpy()
    except (AttributeError, TypeError, ValueError):
        pass
    if target in {"jax", "equinox"}:
        import jax.numpy as jnp

        return jnp.asarray(value)
    if target in {"numpy", "ivy"}:
        import numpy as np

        return np.asarray(value)
    if target == "tensorflow":
        import tensorflow as tf

        return tf.convert_to_tensor(value)
    if target == "torch":
        import torch

        return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return value


def _is_array_like(value: Any) -> bool:
    return hasattr(value, "shape") and (
        hasattr(value, "dtype") or hasattr(value, "detach")
    )


class _TargetModuleProxy:
    """Proxy a simple source module while exposing target-native parameters."""

    def __init__(self, source_module: Any, target: str) -> None:
        object.__setattr__(self, "_source_module", source_module)
        object.__setattr__(self, "_target", target)
        for registry_name in ("_parameters", "_buffers"):
            registry = getattr(source_module, registry_name, {})
            for name, value in registry.items():
                if value is not None:
                    object.__setattr__(self, name, _as_target_array(value, target))
        for name, value in getattr(source_module, "_modules", {}).items():
            if value is not None:
                object.__setattr__(self, name, _TargetModuleProxy(value, target))
        for name, value in vars(source_module).items():
            if name.startswith("_"):
                continue
            if _is_array_like(value):
                object.__setattr__(self, name, _as_target_array(value, target))
            elif hasattr(value, "forward") and callable(value):
                object.__setattr__(self, name, _TargetModuleProxy(value, target))
            elif isinstance(value, (str, int, float, bool, tuple, list, dict, type(None))):
                object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        source = object.__getattribute__(self, "_source_module")
        return getattr(source, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        source = object.__getattribute__(self, "_source_module")
        target = object.__getattribute__(self, "_target")
        if hasattr(source, "forward"):
            converted = transpile(
                source.forward,
                source=_framework_from_object(source),
                target=target,
            )
            return converted(*args, **kwargs)
        return source(*args, **kwargs)


def _transpile_callable(
    obj: Callable[..., Any],
    *,
    source: str,
    target: str,
    mode: str,
    cache: bool,
    emit: str | Path | None,
    backend_compile: bool,
) -> tuple[Callable[..., Any], TranspileReport]:
    if not callable(obj):
        raise UnsupportedObjectError(f"Expected a callable, received {type(obj)!r}")
    mode = "auto" if mode is None else mode
    if mode not in {"auto", "source", "trace"}:
        raise ValueError("mode must be 'auto', 'source', or 'trace'")
    # ``trace`` is retained as a compatibility spelling.  For source-visible
    # callables the AST path is the deterministic graph representation; a
    # source-unavailable callable still receives the same explicit diagnostic.
    callable_obj = obj
    is_bound_method = inspect.ismethod(obj)
    module_instance: Any | None = None
    # Accept a native module instance directly as a convenience.  Its forward
    # method is the inspectable source and `_ModuleAdapter` will expose its
    # parameters in the selected target.
    if not is_bound_method and not inspect.isfunction(obj) and hasattr(obj, "forward"):
        forward = obj.forward
        if callable(forward):
            module_instance = obj
            callable_obj = forward.__func__ if inspect.ismethod(forward) else forward
            is_bound_method = True
    if is_bound_method:
        callable_obj = getattr(callable_obj, "__func__", callable_obj)
    callable_obj = inspect.unwrap(callable_obj)
    try:
        source_code, primitives = _prepare_source(callable_obj, source)
        source_available = True
    except SourceUnavailableError:
        if mode == "source":
            raise
        raise
    key = cache_key(
        source_code,
        source,
        target,
        backend_compile,
        REGISTRY_REVISION,
        getattr(obj, "__qualname__", repr(obj)),
    )
    source_cache = SourceCache()
    cache_hit = False
    metadata: Mapping[str, Any] = {}
    if cache:
        cached = source_cache.load(key)
        if cached:
            source_code, metadata = cached
            primitives = tuple(metadata.get("converted_primitives", primitives))
            cache_hit = True
    if cache and not cache_hit:
        source_cache.save(
            key,
            source_code,
            {
                "source": source,
                "target": target,
                "converted_primitives": primitives,
            },
        )
    generated = _compile_generated(source_code, obj=callable_obj, target=target)
    if is_bound_method:
        owner = module_instance if module_instance is not None else obj.__self__
        generated = _ModuleAdapter(owner, generated, target)
    generated = _compile_backend(generated, target, backend_compile)
    emitted_path: str | None = None
    if emit is not None:
        requested = Path(emit)
        if requested.suffix != ".py":
            requested.mkdir(parents=True, exist_ok=True)
            requested = requested / f"{getattr(obj, '__name__', 'transpiled')}_{key[:12]}.py"
        else:
            requested.parent.mkdir(parents=True, exist_ok=True)
        requested.write_text(
            "# Generated by Hesperus Ivy.\n"
            f"# source={source}; target={target}; cache_key={key}\n\n"
            "from ivy.transpiler import runtime as _ivy_runtime\n"
            f"_ivy_target = {target!r}\n\n{source_code}\n",
            encoding="utf-8",
        )
        requested.with_suffix(".json").write_text(
            json.dumps(
                {
                    "source": source,
                    "target": target,
                    "cache_key": key,
                    "converted_primitives": primitives,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        emitted_path = str(requested)
    report = TranspileReport(
        object_name=getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj))),
        source=source,
        target=target,
        mode=mode,
        cache_hit=cache_hit,
        source_available=source_available,
        emitted_path=emitted_path,
        converted_primitives=tuple(primitives),
        framework_versions={
            name: _version(name)
            for name in {
                "numpy",
                "torch" if source == "torch" or target == "torch" else "",
                "tensorflow" if source == "tensorflow" or target == "tensorflow" else "",
                "jax" if source == "jax" or target in {"jax", "equinox"} else "",
                "equinox" if target == "equinox" else "",
            }
            if name
        },
    )
    return generated, report


def transpile(
    obj: Callable[..., Any],
    *additional: Callable[..., Any],
    source: str | None = None,
    target: str | None = None,
    to: str | None = None,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    mode: str = "auto",
    backend_compile: bool = False,
    cache: bool = True,
    emit: str | Path | None = None,
    output_dir: str | Path | None = None,
    return_report: bool = False,
    **legacy: Any,
) -> Any:
    """Convert a Python callable to a native maintained framework target.

    ``to`` and ``output_dir`` are accepted as compatibility aliases for Ivy
    releases before Hesperus Ivy 2.0.  ``args`` and ``kwargs`` are retained for
    API compatibility and are used by future graph adapters; source-to-source
    conversion itself does not execute the source function during conversion.
    """

    if additional:
        # The upstream API accepted ``transpile(fn_a, fn_b, ...)`` and built a
        # composed graph.  Preserve that migration path while keeping the
        # single-callable return value lightweight for the common case.
        if target is not None and to is not None and target != to:
            raise ValueError("target= and legacy to= specify different targets")
        return trace_graph(
            obj,
            *additional,
            source=source,
            target=target if target is not None else to,
            args=args,
            kwargs=kwargs,
            mode=mode,
            backend_compile=backend_compile,
            cache=cache,
            emit=emit,
            output_dir=output_dir,
            **legacy,
        )
    del args, kwargs
    if legacy:
        unknown = ", ".join(sorted(legacy))
        warnings.warn(
            f"Ignoring legacy transpile options: {unknown}",
            DeprecationWarning,
            stacklevel=2,
        )
    source = _canonical_source(source or _framework_from_object(obj))
    if to is not None and target is None:
        warnings.warn(
            "to= is retained as a compatibility alias; use target= instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    target = _canonical_target(target, to)
    if output_dir is not None and emit is None:
        emit = output_dir
    generated, report = _transpile_callable(
        obj,
        source=source,
        target=target,
        mode=mode,
        cache=cache,
        emit=emit,
        backend_compile=backend_compile,
    )
    if return_report:
        return TranspileResult(generated, report)
    return generated


class Graph:
    """Small callable graph object retained for the legacy ``trace_graph`` API."""

    def __init__(self, fn: Callable[..., Any], report: TranspileReport) -> None:
        self.fn = fn
        self.report = report

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def trace_graph(*objs: Callable[..., Any], args=None, kwargs=None, to=None, **options: Any):
    """Trace-compatible entry point backed by the source transpiler.

    Multiple callables are composed in order, matching the historical Ivy
    graph entry point while keeping each lowering independently inspectable.
    """

    if not objs:
        raise ValueError("trace_graph requires at least one callable")
    source = options.pop("source", None)
    target = to or options.pop("target", None)
    results = [
        transpile(
            obj,
            source=source,
            target=target,
            args=args,
            kwargs=kwargs,
            return_report=True,
            **options,
        )
        for obj in objs
    ]
    if len(results) == 1:
        return Graph(results[0].value, results[0].report)

    def composed(*call_args: Any, **call_kwargs: Any) -> Any:
        value = results[0].value(*call_args, **call_kwargs)
        for item in results[1:]:
            value = item.value(value)
        return value

    first = results[0].report
    report = replace(
        first,
        object_name="Graph(" + ", ".join(item.report.object_name for item in results) + ")",
        converted_primitives=tuple(
            dict.fromkeys(
                primitive
                for item in results
                for primitive in item.report.converted_primitives
            )
        ),
        cache_hit=all(item.report.cache_hit for item in results),
    )
    return Graph(composed, report)


def unify(
    obj: Callable[..., Any],
    *additional: Callable[..., Any],
    source: str | None = None,
    **kwargs: Any,
):
    """Compatibility alias for converting to the Ivy/NumPy reference target."""

    kwargs.pop("to", None)
    return transpile(obj, *additional, source=source, target="ivy", **kwargs)


def clear_cache() -> int:
    """Clear the user-scoped generated source cache."""

    return SourceCache().clear()


def cache_info() -> dict[str, str]:
    """Return the cache location for diagnostics and support requests."""

    return {"path": str(SourceCache().root)}


def compatibility_report(*, source: str | None = None, target: str | None = None) -> dict:
    """Return the current primitive coverage manifest."""

    return coverage(source=source, target=target)
