# DB-EpiFM

Official implementation of **DB-EpiFM: A Dual-Branch Foundation Model for Epilepsy-Related EEG Representation Learning**.

DB-EpiFM learns complementary EEG representations through:

- a **spatial-temporal (ST) branch** based on convolutional patch embedding and a criss-cross Transformer;
- a **spatial-frequency (SF) branch** based on interpretable frequency-band tokens;
- masked reconstruction objectives for both branches;
- a cross-branch alignment objective;
- fused representations for downstream EEG classification.

> **Release status:** Source code is available. Pretrained checkpoints, complete dataset instructions, and final citation information will be added after release.

## Overview

The main implementation is located in [`models/db_epifm.py`](models/db_epifm.py). The model expects segmented EEG signals in the form:

```text
(batch, channels, patches, samples_per_patch)
```

The current pretraining defaults are:

| Item | Default |
|---|---:|
| Samples per patch | 200 |
| Patches per pretraining sample | 30 |
| Sampling rate | 200 Hz |
| Hidden dimension | 200 |
| Transformer layers | 12 |
| Attention heads | 8 |
| ST mask ratio | 0.50 |
| SF mask ratio | 0.15 |
| Frequency bands | 0.5–4, 4–8, 8–13, 13–30, 30–45 Hz |

## Repository Structure

```text
DB-EpiFM/
├── datasets/                   # PyTorch datasets and data loaders
├── models/                     # DB-EpiFM backbone and downstream models
├── processing/                 # Dataset preprocessing scripts
├── supplementary/
│   └── data_leakage_audit/     # Data-leakage audit materials
├── tool/                       # Evaluation and analysis utilities
├── utils/                      # Shared utility functions
├── pretrain_main.py            # Pretraining entry point
├── pretrain_trainer.py         # Pretraining loop
├── finetune_main.py            # Fine-tuning entry point
├── finetune_trainer.py         # Fine-tuning and evaluation loop
├── ENVIRONMENT.md              # Environment details
├── requirements.txt            # Core dependencies
├── requirements-full.txt       # Full experiment environment snapshot
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/dayu-lab/DB-EpiFM.git
cd DB-EpiFM
```

### 2. Create an environment

Replace `3.X` with the Python version used for the original experiments:

```bash
conda create -n db-epifm python=3.X -y
conda activate db-epifm
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The original environment used:

```text
PyTorch 2.9.0+cu126
Torchvision 0.24.0+cu126
PyTorch CUDA runtime 12.6
```

The complete environment snapshot is provided in `requirements-full.txt`. It includes platform-specific development packages, so most users should install `requirements.txt` instead.

Verify the installation with:

```bash
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
python -m pip check
```

See [`ENVIRONMENT.md`](ENVIRONMENT.md) for additional details.

## Data Preparation

The original EEG datasets are **not distributed in this repository**. Users must obtain them from their official providers and comply with the applicable licenses and data-use requirements.

The preprocessing scripts are located in [`processing/`](processing/):

```text
processing/
├── TU_for_pretrain.py
├── processing_chbmit.py
├── processing_tuab.py
├── processing_tuev.py
└── processing_tuev_3class.py
```

### Expected processed-data layout

#### Pretraining

```text
data/processed/pretraining/
└── tuep_tusz_no_tuab_tuev_overlap.lmdb
```

#### TUAB

```text
data/processed/tuab/
├── train/
├── val/
└── test/
```

Each sample should be a pickle file containing:

```python
{"X": eeg_signal, "y": label}
```

The current loader reshapes each sample to `(16, 10, 200)`.

#### TUEV three-class setting

```text
data/processed/tuev_3class/
├── processed_train/
├── processed_eval/
└── processed_test/
```

Each sample should be a pickle file containing at least:

```python
{"signal": eeg_signal, "label": label}
```

The current loader expects a 10-second signal with shape `(16, 2000)` and reshapes it to `(16, 10, 200)`.

#### CHB-MIT

```text
data/processed/chbmit/
├── train/
├── val/
└── test/
```

Each sample should be a pickle file containing:

```python
{"X": eeg_signal, "y": label}
```

The current loader resamples each sample to 2,000 time points and reshapes it to `(16, 10, 200)`.

> Before the final public release, add the exact dataset versions, channel montage, channel order, filtering, resampling, windowing, normalization, label mapping, and subject-level split rules.

## Pretraining

```bash
python pretrain_main.py \
    --dataset_dir ./data/processed/pretraining/tuep_tusz_no_tuab_tuev_overlap.lmdb \
    --model_dir ./outputs/pretraining \
    --cuda 0 \
    --epochs 40 \
    --batch_size 128 \
    --lr 5e-5 \
    --weight_decay 5e-2 \
    --mask_ratio 0.50 \
    --freq_mask_ratio 0.15 \
    --sf_fs 200 \
    --sf_bands "0.5-4,4-8,8-13,13-30,30-45" \
    --lambda_f 1.0 \
    --lambda_align 0.15 \
    --parallel false
```

Important arguments:

| Argument | Description |
|---|---|
| `--dataset_dir` | Path to the pretraining LMDB database |
| `--model_dir` | Output directory for checkpoints and logs |
| `--mask_ratio` | Spatial-temporal mask ratio |
| `--freq_mask_ratio` | Spatial-frequency mask ratio |
| `--sf_fs` | Sampling rate used for frequency-band features |
| `--sf_bands` | Comma-separated frequency bands |
| `--lambda_f` | Frequency reconstruction loss weight |
| `--lambda_align` | ST–SF alignment loss weight |
| `--parallel` | Enable or disable data parallelism |

Boolean arguments accept `true/false`, `yes/no`, `on/off`, or `1/0`.

## Fine-Tuning

Place the pretrained checkpoint at:

```text
./checkpoints/db_epifm_pretrained.pth
```

### TUEV three-class classification

```bash
python finetune_main.py \
    --downstream_dataset TUEV \
    --datasets_dir ./data/processed/tuev_3class \
    --num_of_classes 3 \
    --foundation_dir ./checkpoints/db_epifm_pretrained.pth \
    --model_dir ./outputs/finetuning/tuev \
    --cuda 0 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-5 \
    --weight_decay 1e-2 \
    --use_pretrained_weights true \
    --frozen false \
    --multi_lr true
```

### TUAB binary classification

```bash
python finetune_main.py \
    --downstream_dataset TUAB \
    --datasets_dir ./data/processed/tuab \
    --num_of_classes 2 \
    --foundation_dir ./checkpoints/db_epifm_pretrained.pth \
    --model_dir ./outputs/finetuning/tuab \
    --cuda 0 \
    --epochs 50 \
    --batch_size 32 \
    --use_pretrained_weights true \
    --frozen false \
    --multi_lr true
```

### CHB-MIT binary classification

```bash
python finetune_main.py \
    --downstream_dataset CHB-MIT \
    --datasets_dir ./data/processed/chbmit \
    --num_of_classes 2 \
    --foundation_dir ./checkpoints/db_epifm_pretrained.pth \
    --model_dir ./outputs/finetuning/chbmit \
    --cuda 0 \
    --epochs 50 \
    --batch_size 32 \
    --use_pretrained_weights true \
    --frozen false \
    --multi_lr true
```

## Pretrained Checkpoints
# Pretrained Weights

The pretrained weights of DB-EpiFM are available on
[Hugging Face](https://huggingface.co/dayu-lab/DB-EpiFM).

| Checkpoint | Download | SHA-256 |
|---|---|---|
| DB-EpiFM pretrained backbone | Coming soon | Coming soon |
| TUEV fine-tuned model | Coming soon | Coming soon |
| TUAB fine-tuned model | Coming soon | Coming soon |
| CHB-MIT fine-tuned model | Coming soon | Coming soon |

Large checkpoints should be distributed through a GitHub Release, Hugging Face, Zenodo, or Git LFS rather than committed directly to the main Git history.

## Reproducing the Paper Results

Add the final experimental protocol and results here:

| Dataset | Task | Metrics | Reported result |
|---|---|---|---|
| TUAB | Normal/abnormal classification | Balanced Accuracy, AUROC | To be added |
| TUEV | Three-class event classification | Balanced Accuracy, Cohen's Kappa | To be added |
| CHB-MIT | Seizure/non-seizure classification | Balanced Accuracy, AUROC | To be added |

For every result, report the dataset version, subject-level split, random seeds, number of runs, checkpoint-selection rule, mean, standard deviation, and exact evaluation command.

## Data-Leakage Audit

Data-leakage audit materials are provided in:

```text
supplementary/data_leakage_audit/
```

The final documentation should explain the patient/session exclusion procedure and how to reproduce the audit.

## Acknowledgements

Parts of this repository are adapted from the official implementation of **CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding**.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSES/CBraMod-LICENSE`](LICENSES/CBraMod-LICENSE) for attribution and license details.

Please also cite CBraMod:

```bibtex
@inproceedings{wang2025cbramod,
  title={{CB}raMod: A Criss-Cross Brain Foundation Model for {EEG} Decoding},
  author={Jiquan Wang and Sha Zhao and Zhiling Luo and Yangxuan Zhou and Haiteng Jiang and Shijian Li and Tao Li and Gang Pan},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=NPNUHgHF2w}
}
```

## Citation

The complete DB-EpiFM citation will be added after the paper metadata is finalized.

```bibtex
% TODO: Replace this placeholder with the final DB-EpiFM citation.
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).

Third-party notices and the original CBraMod license are provided in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSES/CBraMod-LICENSE`](LICENSES/CBraMod-LICENSE).

## Contact

For code and reproducibility questions, please open a GitHub issue in this repository.

<!--
Final-release checklist:
1. Add the exact Python version.
2. Upload requirements.txt if it is not yet in the repository.
3. Add official dataset pages and dataset versions.
4. Document channel names/order, montage, filters, sampling rate, windowing,
   normalization, and label mapping.
5. Add fixed subject-level split files.
6. Publish pretrained checkpoints and SHA-256 checksums.
7. Add the model architecture figure.
8. Add final paper results and citation.
9. Add missing TUSL files or remove TUSL imports/support claims.
-->
