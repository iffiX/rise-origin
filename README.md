# RiseOrigin

GPU-accelerated voxel simulator for rigid and soft body simulations.

📖 **Documentation:** https://rise-origin.readthedocs.io — installation (incl.
offline & from-source), concepts, runnable examples, and the full API reference.

## Installation

Pick the backend matching your CUDA generation. `cuda12` is the default —
it covers Volta through Hopper. Use `cuda13` for newer hardware
such as Blackwell.

### PyPI

```bash
pip install "rise-origin[cuda12]"     # Volta (V100) -> Hopper  (default)
pip install "rise-origin[cuda13]"     # Turing (RTX 20xx) -> Blackwell (RTX 50xx)
```

The two backends are mutually exclusive. To switch:

```bash
pip install --upgrade "rise-origin[cuda13]"
```

### Conda (phenoweave channel)

Install [Miniforge](https://github.com/conda-forge/miniforge#install) first,
then:

```bash
mamba install -c phenoweave -c conda-forge -c nvidia rise-origin
```

### Offline / air-gapped

Download the frontend wheel plus the backend wheel for your CUDA generation
from a [GitHub Release](https://github.com/phenoweave/rise-origin/releases),
then on the target machine:

```bash
pip install rise_origin-*.whl rise_origin_core_cuda12-*.whl
```