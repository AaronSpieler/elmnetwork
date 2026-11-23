# jaxtriton: vendored copy of jax-triton

[jax-triton](https://github.com/jax-ml/jax-triton) calls Triton kernels from JAX. It is not on
conda-forge, so a copy lives here instead of a pip dependency; the fused ELM kernels
(`src/models/fused_*.py`) use it as `from src import jaxtriton as jt`.

- **Upstream:** commit `13fc05b` (16 November 2026, "Fixed kernel argument specialization in
  `triton_call`"), after release 0.5.0 (so `jt.__version__` still says 0.5.0). It includes two fixes
  not in that release: kernel argument specialization, and an error (instead of silently misbound
  buffers) for outputs declared before inputs.
- **License:** Apache-2.0, see `src/licences/jax-triton-Apache-2.0.txt` (unchanged).
- **The one change:** the 6 import lines in `__init__.py` are relative (`from .triton_lib import ...`)
  instead of absolute (`from jax_triton.triton_lib import ...`), so the package works under this name.
  `triton_lib.py` and `version.py` are byte-for-byte upstream.
- **Needs:** jax >= 0.11 and triton >= 3.7 (`elmnetwork_env_gpu.yml`: jax 0.12.0, triton 3.9.0).
- **Updating:** copy `jax_triton/{__init__,triton_lib,version}.py` from a newer upstream commit over
  these files, redo the 6 relative imports, update the commit above, run the kernel tests.

Do not edit these files otherwise; changes belong upstream.
