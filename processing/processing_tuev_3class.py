import math
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
TARGET_PATH = r"./data/processed/tuev_3class"

VAL_RATIO = 0.2
RANDOM_SEED = 42
NORMALIZE = False          # True: per-channel z-score for each 10 s window
CLIP_UV = None             # e.g. 800 ; None means no clipping
# =========================


TARGET_FS = 200
WINDOW_SEC = 10.0
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)
HALF_WINDOW = WINDOW_SAMPLES // 2

# Keep only epileptiform TUEV events:
# 1=spsw, 2=gped, 3=pled. Remap to 0,1,2 for CrossEntropyLoss.
KEEP_TUEV_LABELS = {1: 0, 2: 1, 3: 2}
LABEL_NAMES = {
    0: "spsw",
    1: "gped",
    2: "pled",
}
REMOVED_TUEV_LABELS = {
    4: "eyem",
    5: "artf",
    6: "bckg",
}

EXPECTED_TUEV_COUNTS = {
    "processed_train": {
        0: 128,
        1: 1931,
        2: 1392,
        "total": 3451,
    },
    "processed_eval": {
        0: 13,
        1: 493,
        2: 637,
        "total": 1143,
    },
    "processed_test": {
        0: 46,
        1: 822,
        2: 707,
        "total": 1575,
    },
}


def canonical_center_second(start_sec: float, end_sec: float) -> int:
    """Return the integer event-center second used for deduplication.

    ``floor(x + 0.5)`` is used instead of Python's built-in ``round`` to
    avoid banker's rounding at values ending in exactly 0.5.
    """
    if not np.isfinite(start_sec) or not np.isfinite(end_sec):
        raise ValueError(
            f"Non-finite TUEV event interval: start={start_sec}, end={end_sec}"
        )

    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError(
            f"Invalid TUEV event interval: start={start_sec}, end={end_sec}"
        )

    center_sec = 0.5 * (start_sec + end_sec)
    return int(math.floor(center_sec + 0.5))


def deduplicate_events(
    event_data: np.ndarray,
    file_name: str,
):
    """Deduplicate channel-level TUEV annotations into event-level records.

    The manuscript defines one event by the pair:

        (recording file, event-center second)

    Multiple annotation rows from different channels that share this key
    are converted into one multichannel EEG sample.
    """
    unique_events = {}
    skipped_labels = {label: 0 for label in REMOVED_TUEV_LABELS}

    audit = {
        "annotation_rows_total": 0,
        "retained_three_class_rows": 0,
        "duplicate_channel_rows_removed": 0,
        "unique_events_after_deduplication": 0,
        "label_conflicts": [],
    }

    if len(event_data) == 0:
        return [], skipped_labels, audit

    for row_idx, row in enumerate(event_data):
        audit["annotation_rows_total"] += 1

        if len(row) < 4:
            continue

        offending_channel = int(row[0])
        start_sec = float(row[1])
        end_sec = float(row[2])
        label_tuev = int(row[3])

        if label_tuev not in KEEP_TUEV_LABELS:
            if label_tuev in skipped_labels:
                skipped_labels[label_tuev] += 1
            continue

        audit["retained_three_class_rows"] += 1

        center_sec = 0.5 * (start_sec + end_sec)
        center_second = canonical_center_second(
            start_sec=start_sec,
            end_sec=end_sec,
        )

        event_key = (file_name, center_second)

        candidate = {
            "file_name": file_name,
            "event_start_sec": start_sec,
            "event_end_sec": end_sec,
            "event_center_sec": center_sec,
            "event_center_second": center_second,
            "label_tuev": label_tuev,
            "offending_channels": [offending_channel],
            "source_annotation_rows": [row_idx],
        }

        if event_key not in unique_events:
            unique_events[event_key] = candidate
            continue

        existing = unique_events[event_key]

        if existing["label_tuev"] != label_tuev:
            conflict = {
                "file_name": file_name,
                "event_center_second": center_second,
                "existing_label": existing["label_tuev"],
                "conflicting_label": label_tuev,
                "existing_rows": existing["source_annotation_rows"],
                "conflicting_row": row_idx,
            }
            audit["label_conflicts"].append(conflict)

            raise ValueError(
                "Conflicting TUEV labels for the same event-level key: "
                f"{conflict}"
            )

        existing["offending_channels"].append(offending_channel)
        existing["source_annotation_rows"].append(row_idx)
        existing["event_start_sec"] = min(
            existing["event_start_sec"], start_sec
        )
        existing["event_end_sec"] = max(
            existing["event_end_sec"], end_sec
        )
        audit["duplicate_channel_rows_removed"] += 1

    unique_event_list = sorted(
        unique_events.values(),
        key=lambda event: (
            event["event_center_second"],
            event["label_tuev"],
        ),
    )

    for event in unique_event_list:
        event["offending_channels"] = sorted(set(event["offending_channels"]))

    audit["unique_events_after_deduplication"] = len(unique_event_list)
    return unique_event_list, skipped_labels, audit


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
    raw.resample(TARGET_FS, verbose="ERROR")
    raw.filter(l_freq=0.3, h_freq=75, verbose="ERROR")
    raw.notch_filter(freqs=60, verbose="ERROR")

    signals = raw.get_data(units="uV")
    ch_names = raw.info["ch_names"]
    sfreq = int(round(raw.info["sfreq"]))
    n_points = signals.shape[1]
    times = np.arange(n_points, dtype=np.float32) / float(sfreq)
    return signals, ch_names, times


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


def build_event_samples(
    signals: np.ndarray,
    event_data: np.ndarray,
    file_name: str,
    patient_id: str,
    split_name: str,
    normalize: bool = False,
    clip_uv: float | None = None,
):
    """Build one 16-channel sample for each unique TUEV event."""
    samples = []

    unique_events, skipped_labels, dedup_audit = deduplicate_events(
        event_data=event_data,
        file_name=file_name,
    )

    if not unique_events:
        return samples, skipped_labels, dedup_audit

    n_time = signals.shape[1]
    for event in unique_events:
        label_tuev = int(event["label_tuev"])
        label = KEEP_TUEV_LABELS[label_tuev]

        center_sec = float(event["event_center_sec"])
        center_idx = int(round(center_sec * TARGET_FS))
        center_idx = min(max(center_idx, 0), n_time - 1)

        window = safe_extract_fixed_window(signals, center_idx)

        if window.shape[1] != WINDOW_SAMPLES:
            raise RuntimeError(
                "Unexpected TUEV window length: "
                f"{window.shape[1]} != {WINDOW_SAMPLES}"
            )

        if clip_uv is not None:
            window = np.clip(window, -clip_uv, clip_uv)

        if normalize:
            window = normalize_per_channel(window)

        sample = {
            "signal": window.astype(np.float32, copy=False),
            "label": int(label),
            "label_name": LABEL_NAMES[int(label)],
            "label_tuev": label_tuev,
            "event_start_sec": float(event["event_start_sec"]),
            "event_end_sec": float(event["event_end_sec"]),
            "event_center_sec": center_sec,
            "event_center_second": int(event["event_center_second"]),
            "offending_channels": event["offending_channels"],
            "source_annotation_rows": event["source_annotation_rows"],
            "file_name": file_name,
            "patient_id": patient_id,
            "split": split_name,
            "fs": TARGET_FS,
            "channel_order": CHANNEL_ORDER,
        }
        samples.append(sample)

    return samples, skipped_labels, dedup_audit


def infer_patient_id(edf_path: Path, root_split_dir: Path) -> str:
    """
    TUEV train example: root/train/00002275/00002275_00000001.edf -> patient_id=00002275
    TUEV eval example:  root/eval/032/bckg_032_a_.edf              -> patient_id=032
    """
    rel = edf_path.relative_to(root_split_dir)
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return edf_path.stem.split("_")[0]


def process_split(
    split_dir: Path,
    out_dir: Path,
    split_name: str,
    normalize: bool = False,
    clip_uv: float | None = None,
):
    ensure_dir(out_dir)

    all_edf = sorted(split_dir.rglob("*.edf"))
    stats = {
        "files_total": 0,
        "files_success": 0,
        "files_skipped": 0,
        "files_no_kept_event": 0,
        "annotation_rows_total": 0,
        "retained_three_class_rows": 0,
        "duplicate_channel_rows_removed": 0,
        "unique_events_after_deduplication": 0,
        "samples_total": 0,
        "label_hist": {i: 0 for i in range(3)},
        "skipped_tuev_label_hist": {label: 0 for label in REMOVED_TUEV_LABELS},
        "label_conflicts": [],
    }

    for edf_path in tqdm(all_edf, desc=f"Processing {split_name}"):
        stats["files_total"] += 1
        rec_path = edf_path.with_suffix(".rec")
        if not rec_path.exists():
            stats["files_skipped"] += 1
            print(f"[Skip] Missing rec: {rec_path}")
            continue

        try:
            signals, ch_names, _ = read_edf(edf_path)
            signals_16 = convert_to_16ch(signals, ch_names)
            event_data = load_rec(rec_path)
            patient_id = infer_patient_id(edf_path, split_dir)
            samples, skipped_labels, dedup_audit = build_event_samples(
                signals=signals_16,
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

        for label_tuev, count in skipped_labels.items():
            stats["skipped_tuev_label_hist"][label_tuev] += count

        stats["annotation_rows_total"] += dedup_audit["annotation_rows_total"]
        stats["retained_three_class_rows"] += dedup_audit[
            "retained_three_class_rows"
        ]
        stats["duplicate_channel_rows_removed"] += dedup_audit[
            "duplicate_channel_rows_removed"
        ]
        stats["unique_events_after_deduplication"] += dedup_audit[
            "unique_events_after_deduplication"
        ]
        stats["label_conflicts"].extend(dedup_audit["label_conflicts"])

        if len(samples) == 0:
            stats["files_no_kept_event"] += 1
            continue

        for sample in samples:
            center_second = sample["event_center_second"]
            out_name = (
                f"{patient_id}_{edf_path.stem}"
                f"_center{center_second:06d}.pkl"
            )
            output_path = out_dir / out_name

            if output_path.exists():
                raise FileExistsError(
                    "Duplicate TUEV event output path: " f"{output_path}"
                )

            save_pickle(sample, output_path)
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
    patient_ids = sorted({f.name.split("_")[0] for f in train_files})

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
        patient_id = src.name.split("_")[0]
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


def count_final_tuev_samples(final_root: Path):
    counts = {}

    for split_name in (
        "processed_train",
        "processed_eval",
        "processed_test",
    ):
        split_dir = final_root / split_name
        label_hist = {0: 0, 1: 0, 2: 0}
        files = sorted(split_dir.glob("*.pkl"))

        for file_path in files:
            with file_path.open("rb") as stream:
                sample = pickle.load(stream)

            label = int(sample["label"])
            if label not in label_hist:
                raise ValueError(f"Unexpected TUEV label {label} in {file_path}")
            label_hist[label] += 1

        counts[split_name] = {
            **label_hist,
            "total": len(files),
        }

    return counts


def validate_final_tuev_counts(final_root: Path):
    actual = count_final_tuev_samples(final_root)
    errors = []

    for split_name, expected in EXPECTED_TUEV_COUNTS.items():
        observed = actual[split_name]

        for key, expected_value in expected.items():
            observed_value = observed[key]
            if observed_value != expected_value:
                errors.append(
                    f"{split_name}, {key}: "
                    f"expected={expected_value}, observed={observed_value}"
                )

    if errors:
        raise RuntimeError(
            "TUEV class distribution does not match the manuscript:\n"
            + "\n".join(errors)
        )

    return actual


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
    final_class_counts = validate_final_tuev_counts(final_root)

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
            "kept_tuev_label_map": KEEP_TUEV_LABELS,
            "label_names": LABEL_NAMES,
            "removed_tuev_labels": REMOVED_TUEV_LABELS,
        },
        "train_raw_stats": train_stats,
        "eval_raw_stats": eval_stats,
        "split_stats": split_stats,
        "test_stats": test_stats,
        "final_class_counts": final_class_counts,
    }

    summary_path = target / "preprocess_summary.pkl"
    save_pickle(summary, summary_path)

    print("\nDone.")
    print(f"Summary saved to: {summary_path}")
    print(f"ROOT_PATH   = {root}")
    print(f"TARGET_PATH = {target}")
    print("Kept TUEV labels: 1=spsw->0, 2=gped->1, 3=pled->2")
    print("Removed TUEV labels: 4=eyem, 5=artf, 6=bckg")
    print("Current model-compatible sample shape: (16, 2000) -> reshape(16, 10, 200)")


if __name__ == "__main__":
    main()
