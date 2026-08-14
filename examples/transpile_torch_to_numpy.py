"""Convert a framework-shaped function into a NumPy callable."""

from __future__ import annotations

import numpy as np

import ivy


def source_fn(x, weight, bias):
    import torch

    return torch.relu(torch.matmul(x, weight) + bias)


def main() -> None:
    converted, report = ivy.transpile(
        source_fn,
        source="torch",
        target="numpy",
        return_report=True,
    )
    output = converted(
        np.ones((2, 3), dtype=np.float32),
        np.eye(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )
    print(output)
    print(report.to_json())


if __name__ == "__main__":
    main()
