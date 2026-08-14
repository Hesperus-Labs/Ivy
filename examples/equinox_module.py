"""Small Equinox example; install the JAX extra before running it."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class Scale(eqx.Module):
    weight: jnp.ndarray

    def __call__(self, x):
        return x * self.weight


if __name__ == "__main__":
    model = Scale(jnp.array(2.0))
    print(model(jnp.array(3.0)))
