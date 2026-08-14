# Install and verify

## Supported environments

| Component | Supported baseline |
| --- | --- |
| Python | 3.12–3.13 |
| PyTorch | 2.13.x |
| TensorFlow | 2.21.x |
| JAX | 0.11.x |
| Equinox | 0.13.8.x |
| Optax | 0.2.8.x |

The project lockfile is the reproducible development environment. Published
Git installs resolve the selected extra using normal package metadata.

## CPU/reference installation

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install --torch-backend=cpu \
  "hesperus-ivy[all-cpu] @ git+https://github.com/Hesperus-Labs/Ivy.git@main"
```

`@main` tracks the maintained fork. Once a release tag is available, pin the
same URL to that tag for reproducible application deployments.

## NVIDIA installation

The primary personalized target is Linux x86_64 with an NVIDIA RTX 4060 and a
CUDA 13-compatible driver:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install --torch-backend=cu130 \
  "hesperus-ivy[nvidia] @ git+https://github.com/Hesperus-Labs/Ivy.git@main"
ivy doctor
```

JAX random operations use explicit keys, and TensorFlow’s CUDA dependencies
are installed through its `and-cuda` extra. The two accelerator stacks can
coexist, but a source environment should use one lock resolution at a time.

## Local development

```bash
git clone https://github.com/Hesperus-Labs/Ivy.git
cd Ivy
uv sync --python 3.13
uv run --python 3.13 pytest tests
uv run --python 3.13 --group docs mkdocs build --strict
```

## Troubleshooting

1. Run `ivy doctor` and save its JSON output to an issue.
2. Confirm `python --version` is 3.12 or 3.13.
3. For PyTorch, try `uv pip install torch --torch-backend=auto`.
4. For JAX, verify the installed driver meets the CUDA backend’s requirement.
5. If a conversion fails, attach `TranspileReport.to_json()` and the generated
   source directory, not a private model checkpoint.
