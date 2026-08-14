"""Convert a TensorFlow-shaped function into a NumPy callable."""

from __future__ import annotations

import numpy as np

import ivy


def source_fn(x, weight, bias):
    import tensorflow as tf

    return tf.nn.relu(tf.linalg.matmul(x, weight) + bias)


if __name__ == "__main__":
    converted = ivy.transpile(source_fn, source="tensorflow", target="numpy")
    print(
        converted(
            np.ones((2, 3), dtype=np.float32),
            np.eye(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )
    )
