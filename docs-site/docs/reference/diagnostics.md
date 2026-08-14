# Diagnostics API

::: ivy.cli.doctor

Use the command line for a support bundle:

```bash
ivy doctor > ivy-doctor.json
```

When `nvidia-ml-py` is installed, the same document includes the driver,
device names, and total memory so an RTX 4060/CUDA installation can be checked
without importing a training script.
