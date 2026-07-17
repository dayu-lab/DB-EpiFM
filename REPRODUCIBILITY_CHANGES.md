# Reproducibility Changes

This package includes the following repository-preparation updates.

## Licensing and attribution

- Added the root MIT `LICENSE` for DB-EpiFM.
- Added `THIRD_PARTY_NOTICES.md` describing the relationship to CBraMod.
- Added the original CBraMod MIT license at `LICENSES/CBraMod-LICENSE`.

## Locked environment

- `requirements.txt` contains the exact versions of the runtime libraries used
  by the DB-EpiFM source code.
- `requirements-full.txt` contains the complete package snapshot supplied by
  the project author.
- `ENVIRONMENT.md` records PyTorch, CUDA wheel, installation, and verification
  information. The exact Python version was not present in the supplied
  package list and must be recorded before the public release.

## Portable paths

All personal Windows paths were replaced with repository-relative defaults.
The scripts assume commands are launched from the repository root and use this
layout by default:

```text
data/
├── raw/
└── processed/
checkpoints/
outputs/
```

Users can override paths through command-line arguments where those arguments
are available.

## Boolean command-line arguments

Unsafe `type=bool` declarations were replaced by a strict `str2bool` parser.
The affected arguments now accept both explicit values and bare flags:

```bash
python pretrain_main.py --parallel true --need_mask false
python pretrain_main.py --parallel
python finetune_main.py --frozen false --use_pretrained_weights true
```

Accepted values are `true/false`, `yes/no`, `on/off`, and `1/0`, without case
sensitivity.
