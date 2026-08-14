# Command-line interface

Both `ivy` and `ivy-doctor` are installed as console scripts.

## Environment diagnostics

```bash
ivy doctor
# or
ivy-doctor
```

Outputs JSON containing package context, Python/platform/machine, installed
versions for NumPy, PyTorch, TensorFlow, JAX, Equinox, and Optax, plus best-
effort CUDA/JAX/NVML details. Optional-backend initialization errors are
captured as strings instead of crashing diagnostics.

## Cache location

```bash
ivy cache-info
```

Returns JSON with the resolved user cache path.

## Clear generated-source cache

```bash
ivy cache-clear
```

Prints the number of Hesperus cache files removed.

## Primitive coverage

```bash
ivy coverage
ivy coverage --source torch --target equinox
```

Outputs the complete JSON registry manifest or a framework-pair filter. Source
and target values should use canonical names; programmatic API normalization is
more permissive than the report filter.

## Support bundle

```bash
ivy doctor > ivy-doctor.json
ivy coverage --source tensorflow --target jax > ivy-coverage.json
```

Review diagnostics before sharing; device and platform details may be
sensitive in some environments.
