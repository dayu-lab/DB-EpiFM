# DB-EpiFM

Official implementation of **DB-EpiFM: A Dual-Branch Foundation Model for Epilepsy-related EEG Representation Learning**.

DB-EpiFM is an epilepsy-oriented EEG foundation model that learns complementary representations through two parallel branches:

- a **temporal-spatial (TS) branch** for waveform morphology, temporal evolution, and inter-channel dependencies;
- a **spectral-spatial (SS) branch** for explicit channel-band representation learning;
- branch-specific masked reconstruction objectives;
- a cross-branch representation alignment objective;
- residual fusion of the two pretrained representations for downstream EEG classification.

The current implementation supports self-supervised pretraining on **TUEP + TUSZ** and downstream evaluation on **TUAB**, **CHB-MIT**, and **TUEV**.

---

## Model Overview

### Temporal-spatial branch

Each EEG recording is divided into 1-s channel-wise patches. A convolutional patch encoder maps each patch to an embedding, after which asymmetric conditional positional encoding and criss-cross Transformer blocks are used to model temporal and spatial dependencies.

### Spectral-spatial branch

For each EEG channel, spectral features are computed over five frequency bands:

| Band | Frequency range |
|---|---:|
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–45 Hz |

Each channel-band token is constructed from absolute and relative band-power information and processed by a dedicated spectral-spatial encoder.

### Pretraining objective

DB-EpiFM is optimized with two masked reconstruction losses and a cross-branch alignment loss:

\[
L = L_{ts} + L_{ss} + \lambda_{align}L_{align}.
\]

The main experiments use:

- temporal-spatial mask ratio: **0.15**
- spectral-spatial mask ratio: **0.15**
- cross-branch alignment weight: **0.10**

### Downstream fusion

During fine-tuning, the reconstruction heads and alignment projection heads are removed.

The spectral-spatial representation is first averaged across the frequency-band dimension to obtain a channel-wise spectral representation. It is then broadcast across the temporal-patch dimension, scaled by a learnable fusion coefficient initialized to **0.1**, and residually added to the temporal-spatial representation. The fused representation is subsequently passed to the task-specific classifier.

---

## Main Pretraining Configuration

| Setting | Value |
|---|---:|
| Sampling rate | 200 Hz |
| Channels | 16 |
| Pretraining window | 30 s |
| Patch duration | 1 s |
| Input shape | 16 × 30 × 200 |
| Hidden dimension | 200 |
| Feed-forward dimension | 800 |
| Transformer layers | 12 |
| Attention heads | 8 |
| TS mask ratio | 0.15 |
| SS mask ratio | 0.15 |
| Alignment weight | 0.10 |
| Batch size | 128 |
| Epochs | 40 |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 5e-2 |
| Scheduler | CosineAnnealingLR |
| Gradient clipping | 1.0 |

---

## Repository Structure

```text
DB-EpiFM/
├── datasets/                       # PyTorch datasets and data loaders
├── models/                         # DB-EpiFM backbone and downstream models
├── processing/                     # Dataset preprocessing scripts
├── supplementary/
│   └── data_leakage_audit/         # Patient-level leakage-audit materials
├── tool/                           # Evaluation and analysis utilities
├── utils/                          # Shared utility functions
├── pretrain_main.py                # Pretraining entry point
├── pretrain_trainer.py             # Pretraining loop
├── finetune_main.py                # Fine-tuning entry point
├── finetune_trainer.py             # Fine-tuning and evaluation loop
├── ENVIRONMENT.md                  # Software and hardware environment
├── requirements-full.txt           # Additional dependency snapshot
├── THIRD_PARTY_NOTICES.md          # Third-party attribution
├── LICENSES/                       # Third-party licenses
└── LICENSE                         # Project license
```

---

## Environment

The experiments reported in the manuscript were conducted with:

- Ubuntu 22.04.5 LTS
- Python 3.11.7
- PyTorch 2.1.2
- CUDA 12.1
- NVIDIA GeForce RTX 4090 GPU with 24 GB memory
- Intel Core i9-13900K CPU
- 128 GB system memory

A minimal Python environment can be created as follows:

```bash
conda create -n db-epifm python=3.11.7 -y
conda activate db-epifm
```

Install PyTorch 2.1.2 for CUDA 12.1:

```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

Install the main runtime and preprocessing dependencies:

```bash
pip install lmdb mne pyEDFlib numpy scipy scikit-learn einops tqdm pandas matplotlib
```

See `ENVIRONMENT.md` for the reported software and hardware environment.

---

## Data

The original EEG recordings are **not redistributed** in this repository. Users must obtain TUEP, TUSZ, TUAB, TUEV, and CHB-MIT from their official providers and comply with the corresponding licenses and data-use requirements.

The provided preprocessing scripts are located in:

```text
processing/
├── TU_for_pretrain.py
├── processing_chbmit.py
├── processing_tuab.py
├── processing_tuev.py
└── processing_tuev_3class.py
```

The downstream preprocessing scripts are configured for the dataset layouts used in this project, including TUAB v3.0.1, TUEV v2.0.1, and CHB-MIT Scalp EEG Database v1.0.0.

---

## Common TUH Bipolar Montage

TUEP, TUSZ, TUAB, and TUEV are converted to the following common 16-channel TCP bipolar montage:

```text
FP1-F7
F7-T3
T3-T5
T5-O1
FP2-F8
F8-T4
T4-T6
T6-O2
FP1-F3
F3-C3
C3-P3
P3-O1
FP2-F4
F4-C4
C4-P4
P4-O2
```

CHB-MIT is converted to the corresponding fixed 16-channel bipolar montage defined in `processing/processing_chbmit.py`.

---

## Pretraining Data Preparation

DB-EpiFM is pretrained on **TUEP + TUSZ**.

Before signal preprocessing, patient identifiers appearing in downstream **TUAB** or **TUEV** are excluded from the pretraining pool at the patient level.

The pretraining preprocessing pipeline performs the following operations:

1. patient-level overlap removal between TUEP/TUSZ and TUAB/TUEV;
2. conversion to the common 16-channel TCP bipolar montage;
3. resampling to 200 Hz;
4. 0.3–75 Hz band-pass filtering;
5. 60 Hz notch filtering;
6. removal of the first 30 s of each recording;
7. segmentation into non-overlapping 30-s windows;
8. amplitude-based quality control;
9. scaling and storage as `float32`;
10. storage in LMDB format.

The preprocessing script can be run with:

```bash
python processing/TU_for_pretrain.py \
    --tuep_root ./data/raw/TUEP \
    --tusz_root ./data/raw/TUSZ \
    --tuab_root ./data/raw/TUAB \
    --tuev_root ./data/raw/TUEV \
    --output_lmdb ./data/processed/pretraining/tuep_tusz_no_tuab_tuev_overlap.lmdb
```

The resulting LMDB is expected at:

```text
data/processed/pretraining/
└── tuep_tusz_no_tuab_tuev_overlap.lmdb
```

Each pretraining sample has shape:

```text
(16, 30, 200)
```

corresponding to:

```text
(channels, temporal patches, samples per patch)
```

---

## Patient-Level Data-Leakage Audit

Patient-level independence between the TUEP/TUSZ pretraining corpus and the downstream TUAB/TUEV datasets was explicitly audited before signal preprocessing.

The audit identified:

- **887** downstream patient identifiers in the exclusion list;
- **101** overlapping pretraining patients actually removed from TUEP/TUSZ;
- **2,195** pretraining EDF recordings removed because of patient overlap;
- **162,617** retained non-overlapping 30-s pretraining windows;
- approximately **1,355.1 h** of EEG after preprocessing.

The released audit materials are available under:

```text
supplementary/data_leakage_audit/
├── Data_Leakage_Audit_Report.md
├── Data_Leakage_Audit_Report.pdf
├── Data_Leakage_Audit_Report.tex
├── audit_summary.json
├── tables/
└── manifests_sanitized/
```

The released manifests use pseudonymous patient hashes and dataset-relative paths. Raw patient identifiers, raw EDF recordings, derived EEG windows, local absolute paths, and LMDB databases are not redistributed.

---

## Pretraining

After preparing the LMDB database, run:

```bash
python pretrain_main.py \
    --dataset_dir ./data/processed/pretraining/tuep_tusz_no_tuab_tuev_overlap.lmdb \
    --model_dir ./outputs/pretraining \
    --cuda 0 \
    --epochs 40 \
    --batch_size 128 \
    --lr 5e-5 \
    --weight_decay 5e-2 \
    --mask_ratio 0.15 \
    --freq_mask_ratio 0.15 \
    --sf_fs 200 \
    --sf_bands "0.5-4,4-8,8-13,13-30,30-45" \
    --lambda_align 0.10 \
    --parallel false
```

Important arguments:

| Argument | Description |
|---|---|
| `--dataset_dir` | Path to the pretraining LMDB database |
| `--model_dir` | Directory for checkpoints and logs |
| `--mask_ratio` | Temporal-spatial masked reconstruction ratio |
| `--freq_mask_ratio` | Spectral-spatial masked reconstruction ratio |
| `--sf_fs` | Sampling rate for spectral feature extraction |
| `--sf_bands` | Frequency-band definitions |
| `--lambda_f` | Spectral reconstruction loss weight |
| `--lambda_align` | Cross-branch alignment loss weight |
| `--parallel` | Enable or disable data parallelism |

Boolean arguments accept `true/false`, `yes/no`, `on/off`, or `1/0`.

---

## Pretrained Weights

The DB-EpiFM pretrained checkpoint is available from Hugging Face:

```text
https://huggingface.co/dayu-lab/DB-EpiFM
```

Checkpoint file:

```text
DB-EpiFM_pretrain.pth
```

For the default fine-tuning configuration, place the checkpoint at:

```text
DB-EpiFM/
└── checkpoints/
    └── DB-EpiFM_pretrain.pth
```

or provide its location explicitly using `--foundation_dir`.

---

## Downstream Tasks

DB-EpiFM is evaluated on three epilepsy-related clinical EEG tasks:

| Dataset | Task | Classes |
|---|---|---:|
| TUAB | Clinical abnormal EEG detection | Normal / Abnormal |
| CHB-MIT | Seizure detection | Non-seizure / Seizure |
| TUEV | Epileptiform EEG event classification | SPSW / GPED / PLED |

TUAB is treated as an **abnormal-versus-normal EEG classification task**, not as seizure detection.

---

## TUAB Preprocessing

The provided TUAB preprocessing script expects the raw dataset under:

```text
data/raw/TUAB/v3.0.1/edf/
```

Run:

```bash
python processing/processing_tuab.py
```

The processed data are stored as:

```text
data/processed/tuab/
├── train/
├── val/
└── test/
```

The official TUAB training and evaluation partitions are retained. The validation subset is constructed from the official training partition at the patient level so that windows from the same patient cannot cross the constructed subsets.

Signals are converted to the common 16-channel TCP bipolar montage, resampled to 200 Hz, filtered at 0.3–75 Hz with a 60 Hz notch filter, and divided into non-overlapping 10-s windows.

Each sample is stored as a pickle file containing:

```python
{"X": eeg_signal, "y": label}
```

---

## CHB-MIT Preprocessing

The provided CHB-MIT preprocessing script expects:

```text
data/raw/CHBMIT/chb-mit-scalp-eeg-database-1.0.0/
```

Run:

```bash
python processing/processing_chbmit.py
```

The final processed dataset is stored under:

```text
data/processed/chbmit/process_2/
├── train/
├── val/
└── test/
```

The patient-independent split is:

- training: `chb01`–`chb20`
- validation: `chb21`–`chb22`
- testing: `chb23`–`chb24`

EEG signals are segmented into 10-s windows. Windows overlapping an annotated seizure interval are labeled as seizure. Additional seizure windows are sampled with a 5-s stride around the annotated seizure interval, including 1 s before seizure onset and 1 s after seizure offset.

Each sample is stored as:

```python
{"X": eeg_signal, "y": label}
```

---

## TUEV Three-Class Preprocessing

DB-EpiFM uses the following three epileptiform-event categories from TUEV:

| Original TUEV label | Event | Model label |
|---:|---|---:|
| 1 | SPSW | 0 |
| 2 | GPED | 1 |
| 3 | PLED | 2 |

Eye movement, artifact, and background categories are excluded.

The preprocessing script expects:

```text
data/raw/TUEV/v2.0.1/edf/
```

Run:

```bash
python processing/processing_tuev_3class.py
```

The model-compatible output is stored under:

```text
data/processed/tuev_3class/processed/
├── processed_train/
├── processed_eval/
└── processed_test/
```

The official training partition is divided into patient-disjoint training and validation subsets using an 80:20 split. The official evaluation partition is retained as the test set.

Each sample contains:

```python
{"signal": eeg_signal, "label": label}
```

with an EEG signal shape of:

```text
(16, 2000)
```

which is reshaped by the data loader to:

```text
(16, 10, 200)
```

---

## Fine-Tuning

Download `DB-EpiFM_pretrain.pth` and place it under:

```text
./checkpoints/DB-EpiFM_pretrain.pth
```

### TUAB: clinical abnormal EEG detection

```bash
python finetune_main.py \
    --downstream_dataset TUAB \
    --datasets_dir ./data/processed/tuab \
    --num_of_classes 2 \
    --foundation_dir ./checkpoints/DB-EpiFM_pretrain.pth \
    --model_dir ./outputs/finetuning/tuab \
    --cuda 0 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-5 \
    --weight_decay 1e-2 \
    --use_pretrained_weights true \
    --frozen false \
    --multi_lr true
```

### CHB-MIT: seizure detection

```bash
python finetune_main.py \
    --downstream_dataset CHB-MIT \
    --datasets_dir ./data/processed/chbmit/process_2 \
    --num_of_classes 2 \
    --foundation_dir ./checkpoints/DB-EpiFM_pretrain.pth \
    --model_dir ./outputs/finetuning/chbmit \
    --cuda 0 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-5 \
    --weight_decay 1e-2 \
    --use_pretrained_weights true \
    --frozen false \
    --multi_lr true
```

### TUEV: three-class epileptiform-event classification

```bash
python finetune_main.py \
    --downstream_dataset TUEV \
    --datasets_dir ./data/processed/tuev_3class/processed \
    --num_of_classes 3 \
    --foundation_dir ./checkpoints/DB-EpiFM_pretrain.pth \
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

The random seed can be controlled with:

```bash
--seed <seed>
```

The main comparative experiments in the manuscript were evaluated over ten independent runs using the same set of random seeds across methods.

---

## Reproducibility

This repository provides:

- model source code;
- preprocessing scripts;
- pretraining and fine-tuning scripts;
- software and hardware environment information;
- patient-independent downstream splitting procedures;
- patient-level pretraining/downstream leakage-audit materials;
- sanitized audit manifests;
- pretrained DB-EpiFM weights.

Original EEG recordings are not redistributed.

For the patient-level leakage audit, see:

```text
supplementary/data_leakage_audit/
```

For the computational environment, see:

```text
ENVIRONMENT.md
```

---

## Acknowledgements

Parts of this implementation are adapted from the official implementation of **CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding**.

CBraMod repository:

```text
https://github.com/wjq-learning/CBraMod
```

Please see:

```text
THIRD_PARTY_NOTICES.md
LICENSES/CBraMod-LICENSE
```

for attribution and third-party license information.

Reference:

```bibtex
@inproceedings{wang2025cbramod,
  title={{CB}raMod: A Criss-Cross Brain Foundation Model for {EEG} Decoding},
  author={Jiquan Wang and Sha Zhao and Zhiling Luo and Yangxuan Zhou and Haiteng Jiang and Shijian Li and Tao Li and Gang Pan},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=NPNUHgHF2w}
}
```

---

## License

The source code in this GitHub repository is released under the MIT License. See `LICENSE`.

Third-party notices and the original CBraMod license are provided in:

```text
THIRD_PARTY_NOTICES.md
LICENSES/CBraMod-LICENSE
```

Pretrained weights are distributed separately through the DB-EpiFM Hugging Face repository; please refer to the model repository for the applicable model license.

---

## Contact

For questions about the code, preprocessing pipeline, or reproducibility, please open an issue in this GitHub repository.
