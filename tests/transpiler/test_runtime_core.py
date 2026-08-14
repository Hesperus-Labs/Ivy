"""Differential checks for the less-obvious cross-framework lowerings."""

from __future__ import annotations

import numpy as np
import pytest

from ivy.transpiler import runtime


def _values(target: str):
    if target == "numpy":
        return np.asarray([[1.0, 2.0], [3.0, 4.0]]), np.asarray([[0, 1], [1, 0]])
    if target == "jax":
        jnp = pytest.importorskip("jax.numpy")
        return jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), jnp.asarray([[0, 1], [1, 0]])
    if target == "torch":
        torch = pytest.importorskip("torch")
        return torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[0, 1], [1, 0]])
    tf = pytest.importorskip("tensorflow")
    return tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([[0, 1], [1, 0]])


@pytest.mark.parametrize("target", ["numpy", "jax", "torch", "tensorflow"])
def test_activation_one_hot_embedding_and_torch_padding(target):
    values, indices = _values(target)
    sigmoid = runtime.call("torch.sigmoid", values, target=target)
    one_hot = runtime.call("torch.nn.functional.one_hot", indices, 3, target=target)
    embedding = runtime.call(
        "torch.nn.functional.embedding", indices, values, target=target
    )
    padded = runtime.call(
        "torch.nn.functional.pad", values, (1, 1, 1, 1), target=target
    )

    np.testing.assert_allclose(np.asarray(sigmoid), 1 / (1 + np.exp(-np.asarray(values))))
    assert np.asarray(one_hot).shape == (2, 2, 3)
    assert np.asarray(embedding).shape == (2, 2, 2)
    np.testing.assert_allclose(
        np.asarray(padded),
        np.pad(np.asarray(values), ((1, 1), (1, 1))),
    )
