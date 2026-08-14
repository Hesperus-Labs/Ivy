"""Public primitive registry used for coverage and conversion reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

REGISTRY_REVISION = "2026.08"


@dataclass(frozen=True)
class PrimitiveSpec:
    """Portable description of one public operation."""

    name: str
    category: str
    sources: tuple[str, ...]
    targets: tuple[str, ...]
    docs: str
    framework_docs: Mapping[str, str] = field(default_factory=dict)


_COMMON = (
    ("zeros", "creation", "https://numpy.org/doc/stable/reference/generated/numpy.zeros.html"),
    ("ones", "creation", "https://numpy.org/doc/stable/reference/generated/numpy.ones.html"),
    ("full", "creation", "https://numpy.org/doc/stable/reference/generated/numpy.full.html"),
    ("empty", "creation", "https://numpy.org/doc/stable/reference/generated/numpy.empty.html"),
    ("eye", "creation", "https://numpy.org/doc/stable/reference/generated/numpy.eye.html"),
    ("arange", "creation", "https://numpy.org/doc/stable/reference/generated/numpy.arange.html"),
    ("linspace", "creation", "https://numpy.org/doc/stable/reference/generated/numpy.linspace.html"),
    ("zeros_like", "creation", "https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html"),
    ("ones_like", "creation", "https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html"),
    ("cast", "dtype", "https://numpy.org/doc/stable/reference/generated/numpy.asarray.html"),
    ("add", "tensor", "https://numpy.org/doc/stable/reference/generated/numpy.add.html"),
    ("sub", "tensor", "https://numpy.org/doc/stable/reference/generated/numpy.subtract.html"),
    ("mul", "tensor", "https://numpy.org/doc/stable/reference/generated/numpy.multiply.html"),
    ("div", "tensor", "https://numpy.org/doc/stable/reference/generated/numpy.divide.html"),
    ("matmul", "tensor", "https://numpy.org/doc/stable/reference/generated/numpy.matmul.html"),
    ("dot", "tensor", "https://numpy.org/doc/stable/reference/generated/numpy.dot.html"),
    ("tensordot", "tensor", "https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html"),
    ("einsum", "tensor", "https://numpy.org/doc/stable/reference/generated/numpy.einsum.html"),
    ("linear", "neural_network", "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html"),
    ("reshape", "tensor", "https://numpy.org/doc/stable/reference/generated/numpy.reshape.html"),
    ("transpose", "tensor", "https://numpy.org/doc/stable/reference/generated/numpy.transpose.html"),
    ("flatten", "tensor", "https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html"),
    ("expand_dims", "tensor", "https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html"),
    ("sum", "reduction", "https://numpy.org/doc/stable/reference/generated/numpy.sum.html"),
    ("mean", "reduction", "https://numpy.org/doc/stable/reference/generated/numpy.mean.html"),
    ("prod", "reduction", "https://numpy.org/doc/stable/reference/generated/numpy.prod.html"),
    ("max", "reduction", "https://numpy.org/doc/stable/reference/generated/numpy.max.html"),
    ("min", "reduction", "https://numpy.org/doc/stable/reference/generated/numpy.min.html"),
    ("argmax", "reduction", "https://numpy.org/doc/stable/reference/generated/numpy.argmax.html"),
    ("argmin", "reduction", "https://numpy.org/doc/stable/reference/generated/numpy.argmin.html"),
    ("relu", "neural_network", "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html"),
    ("relu6", "neural_network", "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu6.html"),
    ("gelu", "neural_network", "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html"),
    ("silu", "neural_network", "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.silu.html"),
    ("sigmoid", "neural_network", "https://docs.pytorch.org/docs/stable/generated/torch.sigmoid.html"),
    ("tanh", "neural_network", "https://numpy.org/doc/stable/reference/generated/numpy.tanh.html"),
    ("softmax", "neural_network", "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html"),
    ("log_softmax", "neural_network", "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html"),
    ("exp", "elementwise", "https://numpy.org/doc/stable/reference/generated/numpy.exp.html"),
    ("log", "elementwise", "https://numpy.org/doc/stable/reference/generated/numpy.log.html"),
    ("sqrt", "elementwise", "https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html"),
    ("abs", "elementwise", "https://numpy.org/doc/stable/reference/generated/numpy.absolute.html"),
    ("where", "selection", "https://numpy.org/doc/stable/reference/generated/numpy.where.html"),
    ("maximum", "selection", "https://numpy.org/doc/stable/reference/generated/numpy.maximum.html"),
    ("minimum", "selection", "https://numpy.org/doc/stable/reference/generated/numpy.minimum.html"),
    ("cat", "manipulation", "https://pytorch.org/docs/stable/generated/torch.cat.html"),
    ("stack", "manipulation", "https://numpy.org/doc/stable/reference/generated/numpy.stack.html"),
    ("split", "manipulation", "https://numpy.org/doc/stable/reference/generated/numpy.split.html"),
    ("clip", "manipulation", "https://numpy.org/doc/stable/reference/generated/numpy.clip.html"),
    ("dropout", "neural_network", "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html"),
    ("norm", "linear_algebra", "https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html"),
    ("one_hot", "neural_network", "https://www.tensorflow.org/api_docs/python/tf/one_hot"),
    ("embedding", "neural_network", "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html"),
    ("isfinite", "elementwise", "https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html"),
    ("stop_gradient", "autodiff", "https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html"),
    ("randn", "random", "https://docs.jax.dev/en/latest/_autosummary/jax.random.normal.html"),
    ("uniform", "random", "https://docs.jax.dev/en/latest/_autosummary/jax.random.uniform.html"),
    ("randint", "random", "https://docs.jax.dev/en/latest/_autosummary/jax.random.randint.html"),
    ("key", "random", "https://docs.jax.dev/en/latest/random-numbers.html"),
    ("pad", "manipulation", "https://numpy.org/doc/stable/reference/generated/numpy.pad.html"),
)


def _framework_docs(name: str, canonical: str) -> dict[str, str]:
    """Return current official links for each maintained framework."""

    torch_path = {
        "cast": "torch.Tensor.to",
        "linear": "torch.nn.functional.linear",
        "one_hot": "torch.nn.functional.one_hot",
        "stop_gradient": "torch.Tensor.detach",
        "randn": "torch.randn",
        "uniform": "torch.rand",
        "randint": "torch.randint",
        "key": "torch.Generator",
        "norm": "torch.linalg.norm",
        "cat": "torch.cat",
        "stack": "torch.stack",
        "split": "torch.split",
        "clip": "torch.clamp",
        "relu": "torch.nn.functional.relu",
        "relu6": "torch.nn.functional.relu6",
        "gelu": "torch.nn.functional.gelu",
        "silu": "torch.nn.functional.silu",
        "softmax": "torch.nn.functional.softmax",
        "log_softmax": "torch.nn.functional.log_softmax",
        "dropout": "torch.nn.functional.dropout",
        "embedding": "torch.nn.functional.embedding",
    }.get(name, f"torch.{name}")
    tf_path = {
        "full": "tf.fill",
        "empty": "tf.zeros",
        "arange": "tf.range",
        "cast": "tf.cast",
        "sub": "tf.subtract",
        "mul": "tf.multiply",
        "div": "tf.divide",
        "dot": "tf.tensordot",
        "sum": "tf.reduce_sum",
        "mean": "tf.reduce_mean",
        "prod": "tf.reduce_prod",
        "max": "tf.reduce_max",
        "min": "tf.reduce_min",
        "linear": "tf.linalg.matmul",
        "norm": "tf.linalg.norm",
        "one_hot": "tf.one_hot",
        "embedding": "tf.gather",
        "stop_gradient": "tf.stop_gradient",
        "randn": "tf.random.normal",
        "uniform": "tf.random.uniform",
        "randint": "tf.random.uniform",
        "key": "tf.random.Generator",
        "cat": "tf.concat",
        "clip": "tf.clip_by_value",
        "dropout": "tf.nn.dropout",
        "sigmoid": "tf.math.sigmoid",
        "relu": "tf.nn.relu",
        "relu6": "tf.nn.relu6",
        "gelu": "tf.nn.gelu",
        "silu": "tf.nn.silu",
        "softmax": "tf.nn.softmax",
        "log_softmax": "tf.nn.log_softmax",
    }.get(name, f"tf.{name}")
    jax_path = {
        "cast": "jax.numpy.astype",
        "linear": "jax.numpy.matmul",
        "sub": "jax.numpy.subtract",
        "mul": "jax.numpy.multiply",
        "div": "jax.numpy.divide",
        "flatten": "jax.Array.flatten",
        "relu": "jax.nn.relu",
        "relu6": "jax.nn.relu6",
        "gelu": "jax.nn.gelu",
        "silu": "jax.nn.silu",
        "one_hot": "jax.nn.one_hot",
        "embedding": "jax.numpy.take",
        "norm": "jax.numpy.linalg.norm",
        "stop_gradient": "jax.lax.stop_gradient",
        "randn": "jax.random.normal",
        "uniform": "jax.random.uniform",
        "randint": "jax.random.randint",
        "key": "jax.random.key",
        "cat": "jax.numpy.concatenate",
        "clip": "jax.numpy.clip",
        "sigmoid": "jax.nn.sigmoid",
        "softmax": "jax.nn.softmax",
        "log_softmax": "jax.nn.log_softmax",
    }.get(name, f"jax.numpy.{name}")
    jax_docs = (
        "https://docs.kidger.site/equinox/api/nn/"
        if name == "dropout"
        else f"https://docs.jax.dev/en/latest/_autosummary/{jax_path}.html"
    )
    return {
        "pytorch": f"https://docs.pytorch.org/docs/stable/generated/{torch_path}.html",
        "tensorflow": f"https://www.tensorflow.org/api_docs/python/{tf_path}",
        "jax": jax_docs,
        "equinox": "https://docs.kidger.site/equinox/",
        "numpy": canonical,
    }


PRIMITIVES: dict[str, PrimitiveSpec] = {
    name: PrimitiveSpec(
        name=name,
        category=category,
        sources=("torch", "tensorflow", "jax", "ivy", "numpy"),
        targets=("torch", "tensorflow", "jax", "equinox", "ivy", "numpy"),
        docs=docs,
        framework_docs=_framework_docs(name, docs),
    )
    for name, category, docs in _COMMON
}


def primitive_name(name: str) -> str:
    """Normalize a qualified framework name to its registry key."""

    return name.rsplit(".", 1)[-1].lower()


def coverage(*, source: str | None = None, target: str | None = None) -> dict:
    """Return a serialisable registry coverage report."""

    selected: Iterable[PrimitiveSpec] = PRIMITIVES.values()
    if source:
        selected = [item for item in selected if source in item.sources]
    if target:
        selected = [item for item in selected if target in item.targets]
    items = list(selected)
    return {
        "registry_revision": REGISTRY_REVISION,
        "source": source,
        "target": target,
        "total": len(items),
        "primitives": [
            {
                "name": item.name,
                "category": item.category,
                "sources": item.sources,
                "targets": item.targets,
                "docs": item.docs,
                "framework_docs": dict(item.framework_docs),
            }
            for item in items
        ],
    }
