import argparse
import pickle
import random
import re
from pathlib import Path

import lmdb
import mne
import numpy as np
import torch
from tqdm import tqdm


"""
Preprocess TUEP + TUSZ for self-supervised pretraining.

The script excludes any patient whose ID also appears in the downstream TUAB
or TUEV roots, then writes unlabeled 30-second EEG windows to LMDB. The LMDB
format is compatible with datasets/pretraining_dataset.py:

    key -> np.ndarray with shape (16, 30, 200)
    __keys__ -> list[str]
    __stats__ -> dict

Edit the default paths below or pass them from the command line.
"""


# =========================
# CONFIG: edit these paths
# =========================
TUEP_ROOT_DIR = r"./data/raw/TUEP"
TUSZ_ROOT_DIR = r"./data/raw/TUSZ"
TUAB_ROOT_DIR = r"./data/raw/TUAB"
TUEV_ROOT_DIR = r"./data/raw/TUEV"
OUTPUT_LMDB_PATH = r"./data/processed/pretraining/tuep_tusz_no_tuab_tuev_overlap.lmdb"

TARGET_FS = 200
WINDOW_SEC = 30
STRIDE_SEC = 30
START_CUT_SEC = 30
L_FREQ = 0.3
H_FREQ = 75.0
NOTCH_FREQ = 60.0
AMPLITUDE_THRESHOLD_UV = 1000.0
RANDOM_SEED = 42
MAP_SIZE_GB = 200
OVERWRITE = False
# =========================


CHANNEL_DEFS = [
    ("FP1-F7", "FP1", "F7"),
    ("F7-T3", "F7", "T3"),
    ("T3-T5", "T3", "T5"),
    ("T5-O1", "T5", "O1"),
    ("FP2-F8", "FP2", "F8"),
    ("F8-T4", "F8", "T4"),
    ("T4-T6", "T4", "T6"),
    ("T6-O2", "T6", "O2"),
    ("FP1-F3", "FP1", "F3"),
    ("F3-C3", "F3", "C3"),
    ("C3-P3", "C3", "P3"),
    ("P3-O1", "P3", "O1"),
    ("FP2-F4", "FP2", "F4"),
    ("F4-C4", "F4", "C4"),
    ("C4-P4", "C4", "P4"),
    ("P4-O2", "P4", "O2"),
]
CHANNEL_ORDER = [x[0] for x in CHANNEL_DEFS]

CHANNEL_ALIASES = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}

GENERIC_PATH_PARTS = {
    "edf",
    "train",
    "dev",
    "eval",
    "test",
    "normal",
    "abnormal",
    "seizure",
    "seizures",
    "events",
    "tuep",
    "tusz",
    "tuab",
    "tuev",
    "raw",
    "data",
}


def setup_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_channel_name(name: str) -> str:
    name = name.upper().strip()
    if name.startswith("EEG "):
        name = name[4:]
    for suffix in ("-REF", "-LE", "-AR", "-A1", "-A2", "-AVG"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    name = name.replace(" ", "").replace("-", "")
    return CHANNEL_ALIASES.get(name, name)


def build_channel_index(ch_names: list[str]) -> dict[str, int]:
    channel_index = {}
    for idx, name in enumerate(ch_names):
        norm = normalize_channel_name(name)
        channel_index.setdefault(norm, idx)
    return channel_index


def apply_tcp_montage(raw_data_uv: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """Convert referential EEG channels to the 16-channel TCP bipolar montage."""
    channel_index = build_channel_index(ch_names)
    converted = []
    missing = []

    for out_name, ch_a, ch_b in CHANNEL_DEFS:
        a = normalize_channel_name(ch_a)
        b = normalize_channel_name(ch_b)
        if a not in channel_index or b not in channel_index:
            missing.append((out_name, ch_a, ch_b))
            continue
        converted.append(raw_data_uv[channel_index[a]] - raw_data_uv[channel_index[b]])

    if missing:
        raise KeyError(f"Missing channels for TCP montage: {missing}")

    return np.stack(converted, axis=0).astype(np.float32)


def normalize_patient_id(token: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", token).lower()


def looks_like_patient_id(token: str) -> bool:
    token = normalize_patient_id(token)
    if not token:
        return False
    if token in GENERIC_PATH_PARTS:
        return False
    if re.fullmatch(r"0[123]tcpar[a-z]?", token):
        return False
    if re.fullmatch(r"s\d+", token) or re.fullmatch(r"t\d+", token):
        return False
    if re.fullmatch(r"19\d{2}|20\d{2}", token):
        return False
    return len(token) >= 3 and token.isalnum()


def infer_patient_id(edf_path: Path) -> str:
    """
    TUH-style files usually start with the patient ID, e.g.
    aaaaaabc_s001_t000.edf. Some TUEV versions use parent folders or names
    such as bckg_032_a_.edf, so parent directories are used as a fallback.
    """
    stem_tokens = [normalize_patient_id(x) for x in edf_path.stem.split("_") if x]

    if stem_tokens:
        if stem_tokens[0] in {"bckg", "seiz", "event"} and len(stem_tokens) > 1:
            if looks_like_patient_id(stem_tokens[1]):
                return stem_tokens[1]
        if looks_like_patient_id(stem_tokens[0]):
            return stem_tokens[0]

    for part in reversed(edf_path.parts[:-1]):
        candidate = normalize_patient_id(part)
        if looks_like_patient_id(candidate):
            return candidate

    raise ValueError(f"Cannot infer patient ID from path: {edf_path}")


def collect_patient_ids(root_dirs: list[Path]) -> set[str]:
    patient_ids = set()
    for root_dir in root_dirs:
        if root_dir is None or not root_dir.exists():
            continue
        for edf_path in root_dir.rglob("*.edf"):
            try:
                patient_ids.add(infer_patient_id(edf_path))
            except Exception as exc:
                print(f"[Warn] Cannot infer downstream patient ID from {edf_path}: {exc}")
    return patient_ids


def collect_patient_ids_by_dataset(dataset_roots: dict[str, Path]) -> dict[str, set[str]]:
    dataset_to_patient_ids = {}
    for dataset_name, root_dir in dataset_roots.items():
        patient_ids = set()
        if root_dir is None or not root_dir.exists():
            print(f"[Warn] Missing {dataset_name} root: {root_dir}")
            dataset_to_patient_ids[dataset_name] = patient_ids
            continue

        for edf_path in root_dir.rglob("*.edf"):
            try:
                patient_ids.add(infer_patient_id(edf_path))
            except Exception as exc:
                print(f"[Warn] Cannot infer {dataset_name} patient ID from {edf_path}: {exc}")
        dataset_to_patient_ids[dataset_name] = patient_ids
    return dataset_to_patient_ids


def build_excluded_patient_source_map(dataset_to_patient_ids: dict[str, set[str]]) -> dict[str, list[str]]:
    patient_to_sources = {}
    for dataset_name, patient_ids in dataset_to_patient_ids.items():
        for patient_id in patient_ids:
            patient_to_sources.setdefault(patient_id, []).append(dataset_name)
    return {patient_id: sorted(sources) for patient_id, sources in patient_to_sources.items()}


def collect_pretrain_files(
    pretrain_roots: dict[str, Path],
    excluded_patient_source_map: dict[str, list[str]],
):
    kept = []
    skipped_overlap = []
    skipped_bad_id = []

    for corpus_name, root_dir in pretrain_roots.items():
        if root_dir is None or not root_dir.exists():
            print(f"[Warn] Missing {corpus_name} root: {root_dir}")
            continue

        for edf_path in sorted(root_dir.rglob("*.edf")):
            try:
                patient_id = infer_patient_id(edf_path)
            except Exception as exc:
                skipped_bad_id.append((str(edf_path), str(exc)))
                continue

            if patient_id in excluded_patient_source_map:
                overlap_with = "+".join(excluded_patient_source_map[patient_id])
                skipped_overlap.append((str(edf_path), patient_id, corpus_name, overlap_with))
                continue

            kept.append(
                {
                    "path": edf_path,
                    "patient_id": patient_id,
                    "corpus": corpus_name,
                }
            )

    return kept, skipped_overlap, skipped_bad_id


def read_and_preprocess_edf(edf_path: Path) -> np.ndarray:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, exclude=[])
    raw.resample(TARGET_FS, verbose="ERROR")
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose="ERROR")
    raw.notch_filter(freqs=NOTCH_FREQ, verbose="ERROR")

    raw_data_uv = raw.get_data(units="uV")
    return apply_tcp_montage(raw_data_uv, raw.ch_names)


def iter_pretrain_windows(signal_16ch: np.ndarray):
    window_samples = int(WINDOW_SEC * TARGET_FS)
    stride_samples = int(STRIDE_SEC * TARGET_FS)
    start_cut = int(START_CUT_SEC * TARGET_FS)

    n_points = signal_16ch.shape[1]
    if n_points < start_cut + window_samples:
        return

    start = start_cut
    while start + window_samples <= n_points:
        window = signal_16ch[:, start : start + window_samples]
        yield window.reshape(16, WINDOW_SEC, TARGET_FS).astype(np.float32)
        start += stride_samples


def save_manifest_text(path: Path, rows: list[tuple | list], header: str):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")


def save_patient_source_map(path: Path, patient_source_map: dict[str, list[str]]):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("patient_id\toverlap_with\n")
        for patient_id in sorted(patient_source_map):
            f.write(f"{patient_id}\t{'+'.join(patient_source_map[patient_id])}\n")


def create_lmdb(
    tuep_root: Path,
    tusz_root: Path,
    tuab_root: Path,
    tuev_root: Path,
    output_lmdb: Path,
    overwrite: bool = False,
    map_size_gb: int = 200,
):
    if output_lmdb.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_lmdb} already exists. Use --overwrite or change --output_lmdb."
            )
        import shutil

        shutil.rmtree(output_lmdb)

    ensure_parent(output_lmdb)

    downstream_dataset_to_patients = collect_patient_ids_by_dataset(
        {"TUAB": tuab_root, "TUEV": tuev_root}
    )
    excluded_patient_source_map = build_excluded_patient_source_map(downstream_dataset_to_patients)
    downstream_patient_ids = set(excluded_patient_source_map.keys())

    print(f"TUAB patients: {len(downstream_dataset_to_patients['TUAB'])}")
    print(f"TUEV patients: {len(downstream_dataset_to_patients['TUEV'])}")
    print(f"Downstream TUAB+TUEV unique patients to exclude: {len(downstream_patient_ids)}")

    pretrain_files, skipped_overlap, skipped_bad_id = collect_pretrain_files(
        pretrain_roots={"TUEP": tuep_root, "TUSZ": tusz_root},
        excluded_patient_source_map=excluded_patient_source_map,
    )
    random.shuffle(pretrain_files)

    print(f"TUEP/TUSZ EDF files kept for pretraining: {len(pretrain_files)}")
    print(f"TUEP/TUSZ EDF files excluded by patient overlap: {len(skipped_overlap)}")
    print(f"TUEP/TUSZ EDF files skipped by bad patient ID: {len(skipped_bad_id)}")

    manifest_dir = output_lmdb.parent / f"{output_lmdb.name}_manifests"
    save_manifest_text(
        manifest_dir / "excluded_overlap.tsv",
        skipped_overlap,
        "edf_path\tpatient_id\tcorpus\toverlap_with",
    )
    save_manifest_text(
        manifest_dir / "skipped_bad_patient_id.tsv",
        skipped_bad_id,
        "edf_path\terror",
    )
    save_patient_source_map(
        manifest_dir / "excluded_patient_ids.tsv",
        excluded_patient_source_map,
    )

    db = lmdb.open(str(output_lmdb), map_size=int(map_size_gb * 1024**3))
    file_key_list = []
    stats = {
        "config": {
            "tuep_root": str(tuep_root),
            "tusz_root": str(tusz_root),
            "tuab_root": str(tuab_root),
            "tuev_root": str(tuev_root),
            "target_fs": TARGET_FS,
            "window_sec": WINDOW_SEC,
            "stride_sec": STRIDE_SEC,
            "start_cut_sec": START_CUT_SEC,
            "l_freq": L_FREQ,
            "h_freq": H_FREQ,
            "notch_freq": NOTCH_FREQ,
            "amplitude_threshold_uv": AMPLITUDE_THRESHOLD_UV,
            "stored_unit": "uV/100, so PretrainingDataset and Trainer feed the same scale as downstream datasets",
            "channel_order": CHANNEL_ORDER,
        },
        "downstream_patients_by_dataset": {
            dataset_name: len(patient_ids)
            for dataset_name, patient_ids in downstream_dataset_to_patients.items()
        },
        "downstream_patients": len(downstream_patient_ids),
        "files_kept_before_signal_qc": len(pretrain_files),
        "files_excluded_by_overlap": len(skipped_overlap),
        "files_skipped_bad_patient_id": len(skipped_bad_id),
        "files_success": 0,
        "files_failed_signal": 0,
        "files_no_valid_window": 0,
        "samples_total": 0,
        "samples_rejected_amplitude": 0,
        "samples_by_corpus": {"TUEP": 0, "TUSZ": 0},
    }

    error_rows = []

    for item in tqdm(pretrain_files, desc="Preprocessing TUEP/TUSZ"):
        edf_path = item["path"]
        corpus = item["corpus"]
        patient_id = item["patient_id"]

        try:
            signal_16ch = read_and_preprocess_edf(edf_path)
        except Exception as exc:
            stats["files_failed_signal"] += 1
            error_rows.append((str(edf_path), patient_id, corpus, repr(exc)))
            continue

        valid_windows = 0
        for window_idx, sample in enumerate(iter_pretrain_windows(signal_16ch)):
            max_abs_uv = float(np.max(np.abs(sample)))
            if AMPLITUDE_THRESHOLD_UV is not None and max_abs_uv > AMPLITUDE_THRESHOLD_UV:
                stats["samples_rejected_amplitude"] += 1
                continue

            # Downstream TU datasets feed data as uV / 100. Store the same
            # normalized scale because PretrainingDataset and Trainer cancel
            # their internal *100 and /100 operations.
            sample_to_store = (sample / 100.0).astype(np.float32)
            key = f"{corpus.lower()}_{patient_id}_{edf_path.stem}_{window_idx:05d}"
            with db.begin(write=True) as txn:
                txn.put(key.encode(), pickle.dumps(sample_to_store, protocol=pickle.HIGHEST_PROTOCOL))

            file_key_list.append(key)
            valid_windows += 1
            stats["samples_total"] += 1
            stats["samples_by_corpus"][corpus] += 1

        if valid_windows == 0:
            stats["files_no_valid_window"] += 1
        else:
            stats["files_success"] += 1

    with db.begin(write=True) as txn:
        txn.put("__keys__".encode(), pickle.dumps(file_key_list, protocol=pickle.HIGHEST_PROTOCOL))
        txn.put("__stats__".encode(), pickle.dumps(stats, protocol=pickle.HIGHEST_PROTOCOL))
    db.close()

    save_manifest_text(
        manifest_dir / "signal_errors.tsv",
        error_rows,
        "edf_path\tpatient_id\tcorpus\terror",
    )

    print("\nDone.")
    print(f"LMDB saved to: {output_lmdb}")
    print(f"Manifest folder: {manifest_dir}")
    print(f"Samples: {stats['samples_total']}")
    print(f"Samples by corpus: {stats['samples_by_corpus']}")
    print(f"Signal-error files: {stats['files_failed_signal']}")
    print(f"Amplitude-rejected windows: {stats['samples_rejected_amplitude']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess non-overlapping TUEP + TUSZ windows for pretraining."
    )
    parser.add_argument("--tuep_root", type=str, default=TUEP_ROOT_DIR)
    parser.add_argument("--tusz_root", type=str, default=TUSZ_ROOT_DIR)
    parser.add_argument("--tuab_root", type=str, default=TUAB_ROOT_DIR)
    parser.add_argument("--tuev_root", type=str, default=TUEV_ROOT_DIR)
    parser.add_argument("--output_lmdb", type=str, default=OUTPUT_LMDB_PATH)
    parser.add_argument("--map_size_gb", type=int, default=MAP_SIZE_GB)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_seed(RANDOM_SEED)
    create_lmdb(
        tuep_root=Path(args.tuep_root),
        tusz_root=Path(args.tusz_root),
        tuab_root=Path(args.tuab_root),
        tuev_root=Path(args.tuev_root),
        output_lmdb=Path(args.output_lmdb),
        overwrite=args.overwrite,
        map_size_gb=args.map_size_gb,
    )
