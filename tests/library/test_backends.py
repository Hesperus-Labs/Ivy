"""Small contract tests for the public Ivy backend surface.

The historical Ivy test matrix is intentionally enormous and requires a
separate generated-dtype database.  These tests exercise the stable public
contract that users rely on in a fresh uv installation, across every backend
declared by Hesperus Ivy.
"""

from __future__ import annotations

import numpy as np
import pytest

import ivy


def _backend_value(name: str):
    if name == "numpy":
        return np.asarray
    if name == "jax":
        jnp = pytest.importorskip("jax.numpy")
        return jnp.asarray
    if name == "torch":
        torch = pytest.importorskip("torch")
        return torch.as_tensor
    if name == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        return tf.convert_to_tensor
    raise AssertionError(name)


@pytest.fixture(autouse=True)
def reset_backend():
    ivy.unset_backend()
    yield
    ivy.unset_backend()


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch", "tensorflow"])
def test_core_array_contract_across_backends(backend):
    """Core creation, elementwise, linear algebra, and reduction APIs agree."""

    as_native = _backend_value(backend)
    ivy.set_backend(backend)

    values = as_native([[-1.0, 2.0], [3.0, 4.0]])
    x = ivy.array(values)
    result = ivy.matmul(ivy.relu(x), ivy.ones((2, 2)))

    np.testing.assert_allclose(
        np.asarray(ivy.to_numpy(result)),
        np.asarray([[2.0, 2.0], [7.0, 7.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(np.asarray(ivy.to_numpy(ivy.mean(x))), 2.0)
    assert np.asarray(ivy.to_numpy(ivy.reshape(x, (4,)))).shape == (4,)
    assert ivy.current_backend_str() == backend


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch", "tensorflow"])
def test_public_device_and_random_contract(backend):
    _backend_value(backend)
    ivy.set_backend(backend)

    device = ivy.default_device()
    assert str(device).startswith("cpu")
    samples = ivy.random_uniform(shape=(2, 3), seed=0)
    repeated = ivy.random_uniform(shape=(2, 3), seed=0)
    assert np.asarray(ivy.to_numpy(samples)).shape == (2, 3)
    np.testing.assert_allclose(ivy.to_numpy(samples), ivy.to_numpy(repeated))


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch", "tensorflow"])
def test_legacy_stateful_linear_uses_selected_backend(backend):
    _backend_value(backend)
    ivy.set_backend(backend)
    layer = ivy.Linear(2, 3)
    output = layer(ivy.ones((4, 2)))
    assert np.asarray(ivy.to_numpy(output)).shape == (4, 3)


def test_jax_public_array_type_and_explicit_key():
    jax = pytest.importorskip("jax")
    ivy.set_backend("jax")

    native = jax.random.normal(jax.random.key(5), (2, 2))
    assert isinstance(native, jax.Array)
    assert str(ivy.dev(native)).startswith("cpu")
    assert np.asarray(ivy.to_numpy(native)).shape == (2, 2)


def test_equinox_serialization_round_trip(tmp_path):
    eqx = pytest.importorskip("equinox")
    jax = pytest.importorskip("jax")

    class Bias(eqx.Module):
        value: jax.Array

        def __call__(self, x):
            return x + self.value

    module = Bias(jax.numpy.asarray(3.0))
    converted, state = ivy.from_equinox_module(module, state={"step": 4})
    assert state == {"step": 4}
    assert np.asarray(converted(jax.numpy.asarray(2.0))).item() == 5.0

    path = tmp_path / "bias.eqx"
    ivy.save_equinox(module, path)
    restored = ivy.load_equinox(path, Bias(jax.numpy.asarray(0.0)))
    assert np.asarray(restored(jax.numpy.asarray(2.0))).item() == 5.0
    assert path.with_suffix(".eqx.json").exists()
