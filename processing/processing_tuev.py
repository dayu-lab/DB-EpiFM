import os
import pickle
import shutil
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm


# =========================
# CONFIG: edit these paths
# =========================
ROOT_PATH = r"./data/raw/TUEV/v2.0.1/edf"
TARGET_PATH = r"./data/processed/tuev"

VAL_RATIO = 0.2
RANDOM_SEED = 42
NORMALIZE = False          # True: per-channel z-score for each 10 s window
CLIP_UV = None             # e.g. 800 ; None means no clipping
# =========================

TARGET_FS = 200
WINDOW_SEC = 10.0
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)
HALF_WINDOW = WINDOW_SAMPLES // 2

# Current project-compatible 16-channel subset.
CHANNEL_DEFS = [
    ("FP1-F7", "EEG FP1-REF", "EEG F7-REF"),
    ("F7-T3", "EEG F7-REF", "EEG T3-REF"),
    ("T3-T5", "EEG T3-REF", "EEG T5-REF"),
    ("T5-O1", "EEG T5-REF", "EEG O1-REF"),
    ("FP2-F8", "EEG FP2-REF", "EEG F8-REF"),
    ("F8-T4", "EEG F8-REF", "EEG T4-REF"),
    ("T4-T6", "EEG T4-REF", "EEG T6-REF"),
    ("T6-O2", "EEG T6-REF", "EEG O2-REF"),
    ("FP1-F3", "EEG FP1-REF", "EEG F3-REF"),
    ("F3-C3", "EEG F3-REF", "EEG C3-REF"),
    ("C3-P3", "EEG C3-REF", "EEG P3-REF"),
    ("P3-O1", "EEG P3-REF", "EEG O1-REF"),
    ("FP2-F4", "EEG FP2-REF", "EEG F4-REF"),
    ("F4-C4", "EEG F4-REF", "EEG C4-REF"),
    ("C4-P4", "EEG C4-REF", "EEG P4-REF"),
    ("P4-O2", "EEG P4-REF", "EEG O2-REF"),
]
CHANNEL_ORDER = [x[0] for x in CHANNEL_DEFS]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_pickle(obj, filename: Path):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_rec(rec_path: Path) -> np.ndarray:
    event_data = np.genfromtxt(rec_path, delimiter=",")
    if event_data.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if event_data.ndim == 1:
        event_data = event_data[None, :]
    return event_data.astype(np.float32)


def read_edf(file_path: Path):
    raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose="ERROR")
    raw.resample(TARGET_FS)
    raw.filter(l_freq=0.3, h_freq=75, verbose="ERROR")
    raw.notch_filter(freqs=60, verbose="ERROR")

    signals = raw.get_data(units="uV")
    ch_names = raw.info["ch_names"]
    sfreq = int(round(raw.info["sfreq"]))
    n_points = signals.shape[1]
    times = np.arange(n_points, dtype=np.float32) / float(sfreq)
    return signals, ch_names, times, raw


def convert_to_16ch(signals: np.ndarray, ch_names: list[str]) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(ch_names)}
    converted = []
    missing = []
    for out_name, a, b in CHANNEL_DEFS:
        if a not in name_to_idx or b not in name_to_idx:
            missing.append((out_name, a, b))
            continue
        converted.append(signals[name_to_idx[a]] - signals[name_to_idx[b]])

    if missing:
        raise KeyError(f"Missing required channels for 16-ch montage: {missing}")

    return np.stack(converted, axis=0).astype(np.float32)


def safe_extract_fixed_window(signals: np.ndarray, center_idx: int) -> np.ndarray:
    """Extract a fixed 10-second window centered at center_idx with edge padding."""
    n_ch, n_time = signals.shape
    start = center_idx - HALF_WINDOW
    end = center_idx + HALF_WINDOW

    pad_left = max(0, -start)
    pad_right = max(0, end - n_time)

    start = max(0, start)
    end = min(n_time, end)
    window = signals[:, start:end]

    if pad_left > 0 or pad_right > 0:
        window = np.pad(window, ((0, 0), (pad_left, pad_right)), mode="edge")

    if window.shape[1] != WINDOW_SAMPLES:
        raise RuntimeError(f"Unexpected window length: {window.shape[1]} != {WINDOW_SAMPLES}")
    return window.astype(np.float32)


def normalize_per_channel(window: np.ndarray) -> np.ndarray:
    out = window.copy()
    for i in range(out.shape[0]):
        std = float(np.std(out[i]))
        if std > 1e-6:
            out[i] = (out[i] - np.mean(out[i])) / std
        else:
            out[i] = out[i] - np.mean(out[i])
    return out.astype(np.float32)


def build_event_samples(signals: np.ndarray, times: np.ndarray, event_data: np.ndarray,
                        file_name: str, patient_id: str, split_name: str,
                        normalize: bool = False, clip_uv: float | None = None):
    samples = []
    if len(event_data) == 0:
        return samples

    n_time = signals.shape[1]
    for row_idx, row in enumerate(event_data):
        if len(row) < 4:
            continue

        offending_channel = int(row[0])
        start_sec = float(row[1])
        end_sec = float(row[2])
        label_tuev = int(row[3])

        # Remap 1..6 -> 0..5 for CrossEntropyLoss compatibility.
        label = label_tuev - 1
        if not (0 <= label <= 5):
            continue

        center_sec = 0.5 * (start_sec + end_sec)
        center_idx = int(round(center_sec * TARGET_FS))
        center_idx = min(max(center_idx, 0), n_time - 1)

        window = safe_extract_fixed_window(signals, center_idx)

        if clip_uv is not None:
            window = np.clip(window, -clip_uv, clip_uv)

        if normalize:
            window = normalize_per_channel(window)

        sample = {
            "signal": window.astype(np.float32),
            "label": int(label),
            "label_tuev": int(label_tuev),
            "offending_channel": int(offending_channel),
            "event_start_sec": float(start_sec),
            "event_end_sec": float(end_sec),
            "event_center_sec": float(center_sec),
            "file_name": file_name,
            "patient_id": patient_id,
            "split": split_name,
            "fs": TARGET_FS,
            "channel_order": CHANNEL_ORDER,
        }
        samples.append(sample)

    return samples


def infer_patient_id(edf_path: Path, root_split_dir: Path) -> str:
    """
    TUEV train example: root/train/00002275/00002275_00000001.edf -> patient_id=00002275
    TUEV eval example:  root/eval/032/bckg_032_a_.edf              -> patient_id=032
    """
    rel = edf_path.relative_to(root_split_dir)
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return edf_path.stem.split("_")[0]


def process_split(split_dir: Path, out_dir: Path, split_name: str,
                  normalize: bool = False, clip_uv: float | None = None):
    ensure_dir(out_dir)

    all_edf = sorted(split_dir.rglob("*.edf"))
    stats = {
        "files_total": 0,
        "files_success": 0,
        "files_skipped": 0,
        "samples_total": 0,
        "label_hist": {i: 0 for i in range(6)},
    }

    for edf_path in tqdm(all_edf, desc=f"Processing {split_name}"):
        stats["files_total"] += 1
        rec_path = edf_path.with_suffix(".rec")
        if not rec_path.exists():
            stats["files_skipped"] += 1
            print(f"[Skip] Missing rec: {rec_path}")
            continue

        try:
            signals, ch_names, times, _ = read_edf(edf_path)
            signals_16 = convert_to_16ch(signals, ch_names)
            event_data = load_rec(rec_path)
            patient_id = infer_patient_id(edf_path, split_dir)
            samples = build_event_samples(
                signals=signals_16,
                times=times,
                event_data=event_data,
                file_name=edf_path.name,
                patient_id=patient_id,
                split_name=split_name,
                normalize=normalize,
                clip_uv=clip_uv,
            )
        except Exception as e:
            stats["files_skipped"] += 1
            print(f"[Skip] {edf_path}: {e}")
            continue

        for idx, sample in enumerate(samples):
            out_name = f"{edf_path.stem}-event{idx:04d}.pkl"
            save_pickle(sample, out_dir / out_name)
            stats["samples_total"] += 1
            stats["label_hist"][sample["label"]] += 1

        stats["files_success"] += 1

    return stats


def split_train_val(processed_train_dir: Path, final_root: Path, val_ratio: float, seed: int):
    train_final = final_root / "processed_train"
    val_final = final_root / "processed_eval"
    test_final = final_root / "processed_test"
    ensure_dir(train_final)
    ensure_dir(val_final)
    ensure_dir(test_final)

    train_files = sorted([x for x in processed_train_dir.glob("*.pkl")])
    patient_ids = sorted({f.stem.split("-event")[0].split("_")[0] for f in train_files})

    rng = np.random.default_rng(seed)
    patient_ids = np.array(patient_ids)
    rng.shuffle(patient_ids)

    n_val = max(1, int(round(len(patient_ids) * val_ratio))) if len(patient_ids) > 1 else 0
    val_ids = set(patient_ids[:n_val].tolist())
    train_ids = set(patient_ids[n_val:].tolist()) if n_val > 0 else set(patient_ids.tolist())

    if len(train_ids) == 0 and len(val_ids) > 0:
        moved = next(iter(val_ids))
        val_ids.remove(moved)
        train_ids.add(moved)

    for src in tqdm(train_files, desc="Splitting original train -> train/val"):
        patient_id = src.stem.split("-event")[0].split("_")[0]
        dst_dir = val_final if patient_id in val_ids else train_final
        shutil.copy2(src, dst_dir / src.name)

    return {
        "num_train_patients": len(train_ids),
        "num_val_patients": len(val_ids),
        "train_patient_ids": sorted(train_ids),
        "val_patient_ids": sorted(val_ids),
    }


def copy_eval_to_test(processed_eval_dir: Path, final_root: Path):
    test_final = final_root / "processed_test"
    ensure_dir(test_final)
    eval_files = sorted(processed_eval_dir.glob("*.pkl"))
    for src in tqdm(eval_files, desc="Copying original eval -> test"):
        shutil.copy2(src, test_final / src.name)
    return {"num_test_files": len(eval_files)}


def main():
    root = Path(ROOT_PATH)
    target = Path(TARGET_PATH)

    train_src = root / "train"
    eval_src = root / "eval"
    if not train_src.exists() or not eval_src.exists():
        raise FileNotFoundError(
            f"Expected train/ and eval/ under {root}. Please modify ROOT_PATH in the script."
        )

    temp_train_dir = target / "processed_train_raw"
    temp_eval_dir = target / "processed_eval_raw"
    final_root = target / "processed"
    ensure_dir(temp_train_dir)
    ensure_dir(temp_eval_dir)
    ensure_dir(final_root)

    train_stats = process_split(
        split_dir=train_src,
        out_dir=temp_train_dir,
        split_name="processed_train_raw",
        normalize=NORMALIZE,
        clip_uv=CLIP_UV,
    )
    eval_stats = process_split(
        split_dir=eval_src,
        out_dir=temp_eval_dir,
        split_name="processed_eval_raw",
        normalize=NORMALIZE,
        clip_uv=CLIP_UV,
    )

    split_stats = split_train_val(
        processed_train_dir=temp_train_dir,
        final_root=final_root,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED,
    )
    test_stats = copy_eval_to_test(
        processed_eval_dir=temp_eval_dir,
        final_root=final_root,
    )

    summary = {
        "config": {
            "root": str(root),
            "target": str(target),
            "target_fs": TARGET_FS,
            "window_sec": WINDOW_SEC,
            "window_samples": WINDOW_SAMPLES,
            "channels": CHANNEL_ORDER,
            "normalize": NORMALIZE,
            "clip_uv": CLIP_UV,
            "val_ratio": VAL_RATIO,
            "seed": RANDOM_SEED,
            "label_map": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
        },
        "train_raw_stats": train_stats,
        "eval_raw_stats": eval_stats,
        "split_stats": split_stats,
        "test_stats": test_stats,
    }

    summary_path = target / "preprocess_summary.pkl"
    save_pickle(summary, summary_path)

    print("\nDone.")
    print(f"Summary saved to: {summary_path}")
    print(f"ROOT_PATH   = {root}")
    print(f"TARGET_PATH = {target}")
    print("Current model-compatible sample shape: (16, 2000) -> reshape(16, 10, 200)")
    print("Label map: 1..6 -> 0..5")


if __name__ == "__main__":
    main()
