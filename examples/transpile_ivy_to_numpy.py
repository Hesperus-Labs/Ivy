"""Use the Ivy compatibility source with a native NumPy target."""

from __future__ import annotations

import numpy as np

import ivy


def source_fn(x):
    import ivy

    return ivy.mean(ivy.relu(x), axis=-1, keepdims=True)


if __name__ == "__main__":
    converted = ivy.transpile(source_fn, source="ivy", target="numpy")
    print(converted(np.array([[-1.0, 2.0]], dtype=np.float32)))
