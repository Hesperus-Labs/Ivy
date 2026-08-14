from __future__ import annotations

import json

import numpy as np
import pytest

import ivy
from ivy.transpiler import (
    SourceUnavailableError,
    UnsupportedPrimitiveError,
    cache_info,
    clear_cache,
    compatibility_report,
)


def torch_style(x, weight, bias):
    import torch

    hidden = torch.matmul(x, weight) + bias
    return torch.relu(hidden).reshape((-1, hidden.shape[-1]))


def ivy_style(x):
    import ivy

    return ivy.mean(ivy.relu(x), axis=-1, keepdims=True)


def torch_functional_style(x):
    from torch.nn import functional as F

    return F.relu(x)


def torch_method_style(x):
    return x.relu().flatten(0, 1)


def test_transpile_to_numpy_returns_native_array(tmp_path, monkeypatch):
    monkeypatch.setenv("HESPERUS_IVY_CACHE_DIR", str(tmp_path / "cache"))
    fn, report = ivy.transpile(
        torch_style,
        source="torch",
        target="numpy",
        emit=tmp_path / "generated",
        return_report=True,
    )
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    weight = np.eye(3, dtype=np.float32)
    bias = np.ones(3, dtype=np.float32)
    result = fn(x, weight, bias)
    np.testing.assert_allclose(result, np.maximum(x + 1, 0))
    assert report.source == "torch"
    assert report.target == "numpy"
    assert "matmul" in report.converted_primitives
    assert list((tmp_path / "generated").glob("*.json"))
    assert list((tmp_path / "generated").glob("*.py"))


def test_ivy_alias_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HESPERUS_IVY_CACHE_DIR", str(tmp_path / "cache"))
    first = ivy.transpile(ivy_style, source="ivy", target="numpy", return_report=True)
    second = ivy.transpile(ivy_style, source="ivy", target="numpy", return_report=True)
    value = second.value(np.array([[-1.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(value, np.array([[1.0]], dtype=np.float32))
    assert first.report.cache_hit is False
    assert second.report.cache_hit is True
    assert cache_info()["path"] == str(tmp_path / "cache")


def test_reports_are_json_serialisable():
    result = ivy.transpile(ivy_style, source="ivy", target="numpy", return_report=True)
    json.dumps(result.report.to_dict())
    assert compatibility_report(source="torch", target="numpy")["total"] > 0


def test_clear_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HESPERUS_IVY_CACHE_DIR", str(tmp_path / "cache"))
    ivy.transpile(ivy_style, source="ivy", target="numpy")
    assert clear_cache() > 0


def test_source_unavailable_is_explicit():
    with pytest.raises(SourceUnavailableError):
        ivy.transpile(lambda value: value + 1, source="ivy", target="numpy")


def test_local_framework_aliases_are_lowered():
    fn = ivy.transpile(torch_functional_style, source="torch", target="numpy")
    np.testing.assert_allclose(fn(np.array([-1.0, 2.0])), np.array([0.0, 2.0]))


def test_tensor_methods_are_lowered_after_generated_calls():
    fn = ivy.transpile(torch_method_style, source="torch", target="numpy")
    np.testing.assert_allclose(fn(np.array([[-1.0, 2.0]], dtype=np.float32)), [0.0, 2.0])


def torch_unknown_style(x):
    import torch

    return torch.unique(x)


def test_unsupported_primitive_fails_with_actionable_error():
    fn = ivy.transpile(torch_unknown_style, source="torch", target="numpy")
    with pytest.raises(UnsupportedPrimitiveError, match="torch.unique"):
        fn(np.array([1, 1, 2]))


def graph_first(x):
    return x + 1


def graph_second(x):
    return x * 2


def test_trace_graph_composes_multiple_source_callables():
    graph = ivy.trace_graph(graph_first, graph_second, source="ivy", to="numpy")
    np.testing.assert_allclose(graph(np.array([1.0])), [4.0])
    assert graph.report.object_name.startswith("Graph(")


def test_transpile_multiple_callables_keeps_upstream_composition():
    graph = ivy.transpile(graph_first, graph_second, source="ivy", target="numpy")
    np.testing.assert_allclose(graph(np.array([1.0])), [4.0])


def test_trace_mode_keeps_legacy_spelling():
    converted, report = ivy.transpile(
        graph_first, source="ivy", target="numpy", mode="trace", return_report=True
    )
    np.testing.assert_allclose(converted(np.array([1.0])), [2.0])
    assert report.mode == "trace"
