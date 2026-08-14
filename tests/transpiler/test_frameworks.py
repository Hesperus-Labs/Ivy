from __future__ import annotations

import numpy as np
import pytest

import ivy


def torch_style(x, weight, bias):
    import torch

    return torch.relu(torch.matmul(x, weight) + bias)


class TorchLikeModule:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        import torch

        return torch.relu(torch.matmul(x, self.weight) + self.bias)


class StatelessTorchModule:
    def forward(self, x):
        import torch

        return torch.relu(x)


def tensorflow_style(x, weight, bias):
    import tensorflow as tf

    return tf.nn.relu(tf.linalg.matmul(x, weight) + bias)


def tensorflow_cast_style(x):
    import tensorflow as tf

    return tf.cast(x, tf.float32)


def tensorflow_random_style(shape):
    import tensorflow as tf

    return tf.random.normal(shape, seed=11)


def jax_style(x, weight, bias):
    import jax.numpy as jnp

    return jnp.maximum(jnp.matmul(x, weight) + bias, 0)


def jax_random_style(key, shape):
    import jax

    return jax.random.normal(key, shape)


def jax_randint_style(key, shape):
    import jax

    return jax.random.randint(key, shape, 0, 5)


def torch_manipulation_style(x):
    import torch

    return torch.stack([x, x], dim=0).clamp(min=0, max=1)


def torch_extended_style(x):
    import torch

    value = torch.transpose(x, 0, 1)
    value = torch.nn.functional.relu6(value)
    return value.flatten(0, 1).to(dtype=torch.float32)


def test_tensorflow_source_to_numpy():
    tf = pytest.importorskip("tensorflow")
    converted = ivy.transpile(tensorflow_style, source="tensorflow", target="numpy")
    inputs = [
        np.ones((2, 3), dtype=np.float32),
        np.eye(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
    ]
    expected = tensorflow_style(
        *(tf.convert_to_tensor(value) for value in inputs)
    ).numpy()
    np.testing.assert_allclose(converted(*inputs), expected)


def test_tensorflow_cast_to_all_array_targets():
    source = tensorflow_cast_style
    values = {
        "numpy": np.ones(3, dtype=np.float64),
        "jax": pytest.importorskip("jax.numpy").ones(3, dtype=np.float32),
        "torch": pytest.importorskip("torch").ones(3, dtype=pytest.importorskip("torch").float64),
        "tensorflow": pytest.importorskip("tensorflow").ones(3, dtype="float64"),
    }
    for target, value in values.items():
        result = ivy.transpile(source, source="tensorflow", target=target)(value)
        assert str(np.asarray(result).dtype) in {"float32", "float64"}


@pytest.mark.parametrize("target", ["numpy", "jax", "torch", "tensorflow"])
def test_stack_clip_lowerings(target):
    if target == "numpy":
        value = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
    elif target == "jax":
        value = pytest.importorskip("jax.numpy").array([-1.0, 0.5, 2.0])
    elif target == "torch":
        value = pytest.importorskip("torch").tensor([-1.0, 0.5, 2.0])
    else:
        value = pytest.importorskip("tensorflow").constant([-1.0, 0.5, 2.0])
    result = ivy.transpile(torch_manipulation_style, source="torch", target=target)(value)
    np.testing.assert_allclose(np.asarray(result), [[0.0, 0.5, 1.0]] * 2)


def test_tensorflow_seed_becomes_explicit_jax_key():
    pytest.importorskip("tensorflow")
    jax = pytest.importorskip("jax")
    converted = ivy.transpile(tensorflow_random_style, source="tensorflow", target="jax")
    first = converted((2, 2))
    second = converted((2, 2))
    assert np.asarray(first).shape == (2, 2)
    # The source seed is represented as a deterministic JAX key.
    np.testing.assert_allclose(np.asarray(first), np.asarray(second))
    assert jax.devices()


@pytest.mark.parametrize("target", ["numpy", "jax", "torch", "tensorflow"])
def test_jax_randint_lowering(target):
    jax = pytest.importorskip("jax")
    converted = ivy.transpile(jax_randint_style, source="jax", target=target)
    result = converted(jax.random.key(3), (2, 3))
    values = np.asarray(result)
    assert values.shape == (2, 3)
    assert np.all((values >= 0) & (values < 5))


def test_jax_source_to_numpy():
    jnp = pytest.importorskip("jax.numpy")
    converted = ivy.transpile(jax_style, source="jax", target="numpy")
    inputs = [
        np.ones((2, 3), dtype=np.float32),
        np.eye(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
    ]
    expected = np.asarray(jax_style(*(jnp.asarray(value) for value in inputs)))
    np.testing.assert_allclose(converted(*inputs), expected)


def test_jax_random_requires_and_uses_explicit_key():
    jax = pytest.importorskip("jax")
    converted = ivy.transpile(jax_random_style, source="jax", target="jax")
    key = jax.random.key(42)
    result = converted(key, (2, 2))
    expected = jax.random.normal(key, (2, 2))
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("target", ["numpy", "torch", "tensorflow"])
def test_jax_random_key_is_deterministic_on_non_jax_targets(target):
    jax = pytest.importorskip("jax")
    key = jax.random.key(7)
    converted = ivy.transpile(jax_random_style, source="jax", target=target)
    first = converted(key, (2, 2))
    second = converted(key, (2, 2))
    np.testing.assert_allclose(np.asarray(first), np.asarray(second))


def _inputs():
    return (
        np.ones((2, 3), dtype=np.float32),
        np.eye(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
    )


def test_torch_to_jax_and_tensorflow():
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    pytest.importorskip("tensorflow")
    x, weight, bias = _inputs()
    jax_fn = ivy.transpile(torch_style, source="torch", target="jax")
    tf_fn = ivy.transpile(torch_style, source="torch", target="tensorflow")
    np.testing.assert_allclose(np.asarray(jax_fn(x, weight, bias)), np.full((2, 3), 2))
    np.testing.assert_allclose(tf_fn(x, weight, bias).numpy(), np.full((2, 3), 2))


def test_tensorflow_to_torch_and_jax():
    torch = pytest.importorskip("torch")
    pytest.importorskip("jax")
    pytest.importorskip("tensorflow")
    x, weight, bias = _inputs()
    torch_fn = ivy.transpile(tensorflow_style, source="tensorflow", target="torch")
    jax_fn = ivy.transpile(tensorflow_style, source="tensorflow", target="jax")
    np.testing.assert_allclose(
        torch_fn(*(torch.tensor(value) for value in (x, weight, bias))).numpy(),
        np.full((2, 3), 2),
    )
    np.testing.assert_allclose(np.asarray(jax_fn(x, weight, bias)), np.full((2, 3), 2))


def test_jax_to_torch_and_tensorflow():
    torch = pytest.importorskip("torch")
    pytest.importorskip("jax")
    pytest.importorskip("tensorflow")
    x, weight, bias = _inputs()
    torch_fn = ivy.transpile(jax_style, source="jax", target="torch")
    tf_fn = ivy.transpile(jax_style, source="jax", target="tensorflow")
    np.testing.assert_allclose(
        torch_fn(*(torch.tensor(value) for value in (x, weight, bias))).numpy(),
        np.full((2, 3), 2),
    )
    np.testing.assert_allclose(tf_fn(x, weight, bias).numpy(), np.full((2, 3), 2))


def test_equinox_bridge_keeps_state_explicit():
    eqx = pytest.importorskip("equinox")
    stateful, state = ivy.to_equinox_module(
        jax_style,
        source="jax",
        state={"step": 0},
    )
    assert isinstance(stateful, eqx.Module)
    output, next_state = stateful(*_inputs(), state=state)
    np.testing.assert_allclose(np.asarray(output), np.full((2, 3), 2))
    assert next_state == state


def test_torch_module_to_equinox_keeps_parameters_and_state_explicit():
    eqx = pytest.importorskip("equinox")
    jnp = pytest.importorskip("jax.numpy")
    torch = pytest.importorskip("torch")

    model = torch.nn.Linear(3, 3)
    with torch.no_grad():
        model.weight.copy_(torch.eye(3))
        model.bias.fill_(1)
    converted, state = ivy.to_equinox_module(model, source="torch", state={"step": 0})
    assert isinstance(converted, eqx.Module)
    output, next_state = converted(jnp.ones((2, 3)), state=state)
    np.testing.assert_allclose(np.asarray(output), np.full((2, 3), 2))
    assert next_state == state


def test_bound_module_parameters_are_converted_to_target_arrays():
    jnp = pytest.importorskip("jax.numpy")
    torch = pytest.importorskip("torch")
    model = TorchLikeModule(torch.eye(3), torch.ones(3))
    converted = ivy.transpile(model.forward, source="torch", target="jax")
    output = converted(jnp.ones((2, 3)))
    np.testing.assert_allclose(np.asarray(output), np.full((2, 3), 2))


def test_torch_linear_module_to_jax():
    jnp = pytest.importorskip("jax.numpy")
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 3)
    with torch.no_grad():
        model.weight.copy_(torch.eye(3))
        model.bias.fill_(1)
    converted = ivy.transpile(model.forward, source="torch", target="jax")
    output = converted(jnp.ones((2, 3)))
    np.testing.assert_allclose(np.asarray(output), np.full((2, 3), 2))


@pytest.mark.parametrize("target", ["numpy", "jax", "tensorflow", "torch"])
def test_torch_linear_module_parameters_follow_target(target):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 3)
    with torch.no_grad():
        model.weight.copy_(torch.eye(3))
        model.bias.fill_(1)
    if target == "jax":
        pytest.importorskip("jax")
        value = pytest.importorskip("jax.numpy").ones((2, 3))
    elif target == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        value = tf.ones((2, 3))
    elif target == "torch":
        value = torch.ones((2, 3))
    else:
        value = np.ones((2, 3), dtype=np.float32)
    converted = ivy.transpile(model.forward, source="torch", target=target)
    output = converted(value)
    np.testing.assert_allclose(np.asarray(output), np.full((2, 3), 2))


def test_torch_linear_to_equinox_module():
    eqx = pytest.importorskip("equinox")
    jnp = pytest.importorskip("jax.numpy")
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 3)
    with torch.no_grad():
        model.weight.copy_(torch.eye(3))
        model.bias.fill_(1)
    converted = ivy.to_equinox_module(model, source="torch")
    assert isinstance(converted, eqx.Module)
    output = converted(jnp.ones((2, 3)))
    np.testing.assert_allclose(np.asarray(output), np.full((2, 3), 2))


def test_custom_torch_module_to_equinox_and_stateless_wrapper():
    eqx = pytest.importorskip("equinox")
    torch = pytest.importorskip("torch")
    model = TorchLikeModule(torch.eye(3), torch.ones(3))
    converted = ivy.to_equinox_module(model, source="torch")
    assert isinstance(converted, eqx.Module)
    output = converted(np.ones((2, 3), dtype=np.float32))
    np.testing.assert_allclose(np.asarray(output), np.full((2, 3), 2))

    stateless = ivy.to_equinox_module(StatelessTorchModule(), source="torch")
    assert isinstance(stateless, eqx.Module)
    np.testing.assert_allclose(np.asarray(stateless(np.array([-1.0, 2.0]))), [0.0, 2.0])


def test_equinox_serialization_uses_native_tree_leaves(tmp_path):
    eqx = pytest.importorskip("equinox")

    class Scale(eqx.Module):
        weight: object

    original = Scale(weight=np.array(2.0, dtype=np.float32))
    path = tmp_path / "scale.eqx"
    ivy.save_equinox(original, path)
    assert path.with_suffix(".eqx.json").is_file()
    restored = ivy.load_equinox(path, original)
    np.testing.assert_allclose(np.asarray(restored.weight), 2.0)


@pytest.mark.parametrize("target", ["numpy", "jax", "torch", "tensorflow"])
def test_extended_torch_methods_and_dtype_constants(target):
    if target == "numpy":
        value = np.arange(-6, 6, dtype=np.float64).reshape(3, 4)
    elif target == "jax":
        value = pytest.importorskip("jax.numpy").arange(-6, 6, dtype=np.float32).reshape(3, 4)
    elif target == "torch":
        torch = pytest.importorskip("torch")
        value = torch.arange(-6, 6, dtype=torch.float64).reshape(3, 4)
    else:
        tensorflow = pytest.importorskip("tensorflow")
        value = tensorflow.reshape(tensorflow.range(-6, 6, dtype="float64"), (3, 4))
    converted = ivy.transpile(torch_extended_style, source="torch", target=target)
    result = converted(value)
    expected = np.clip(np.asarray(value), 0, 6).T.reshape(-1).astype(np.float32)
    np.testing.assert_allclose(np.asarray(result), expected)
