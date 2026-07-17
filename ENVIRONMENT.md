# Reproducibility Environment

The package versions in this repository were locked from the author's
`yj_cbramod` environment.

## Core platform

- PyTorch: `2.9.0+cu126`
- Torchvision: `0.24.0+cu126`
- PyTorch wheel CUDA runtime: `12.6`
- Operating-system family: Windows (the full snapshot contains Windows-only packages)
- Python version: **not included in the supplied `pip list` output**. Before the
  public release, record it with `python --version` and add it here.

## Installation

For the dependencies required by the DB-EpiFM source code:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the complete package snapshot of the original experiment environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-full.txt
```

`requirements-full.txt` is platform-specific and contains development,
Jupyter, DataLad, AWS, and Windows support packages that are not required by
the core training code. For most users, `requirements.txt` is recommended.

## Verifying the environment

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -m pip check
```

The CUDA wheel tag records the runtime bundled with PyTorch; the NVIDIA driver
must independently support CUDA 12.6.
