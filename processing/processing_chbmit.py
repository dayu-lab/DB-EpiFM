"""Single-stage CHB-MIT preprocessing for DB-EpiFM.

Pipeline: EDF -> fixed 16-channel bipolar montage -> resample to 200 Hz ->
0.3-75 Hz band-pass -> 60 Hz notch -> non-overlapping 10-s windows.
Windows are stored by case; patient-level folds are applied by the data loader.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm


DEFAULT_ROOT = Path("./data/raw/CHBMIT/chb-mit-scalp-eeg-database-1.0.0")
DEFAULT_OUTPUT = Path("./data/processed/chbmit/process_2")
TARGET_FS = 200
WINDOW_SEC = 10.0
LOW_FREQ = 0.3
HIGH_FREQ = 75.0
NOTCH_FREQ = 60.0
AMPLITUDE_THRESHOLD_UV = 1000.0

CHANNEL_CANDIDATES = (
    ("FP1-F7", ("FP1-F7",)),
    ("F7-T3", ("F7-T7", "F7-T3")),
    ("T3-T5", ("T7-P7", "T3-T5")),
    ("T5-O1", ("P7-O1", "T5-O1")),
    ("FP2-F8", ("FP2-F8",)),
    ("F8-T4", ("F8-T8", "F8-T4")),
    ("T4-T6", ("T8-P8", "T8-P8-0", "T4-T6")),
    ("T6-O2", ("P8-O2", "T6-O2")),
    ("FP1-F3", ("FP1-F3",)),
    ("F3-C3", ("F3-C3",)),
    ("C3-P3", ("C3-P3",)),
    ("P3-O1", ("P3-O1",)),
    ("FP2-F4", ("FP2-F4",)),
    ("F4-C4", ("F4-C4",)),
    ("C4-P4", ("C4-P4",)),
    ("P4-O2", ("P4-O2",)),
)
CHANNEL_ORDER = tuple(item[0] for item in CHANNEL_CANDIDATES)

ELECTRODE_PAIRS = (
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
)
ELECTRODE_ALIASES = {
    "T7": ("T7", "T3"), "P7": ("P7", "T5"),
    "T8": ("T8", "T4"), "P8": ("P8", "T6"),
    "O1": ("O1", "01"),
}


def parse_summary(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Return EDF filename -> seizure intervals in seconds."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"(?=File\s+Name\s*:)", text, flags=re.I)
    result = {}
    for block in blocks:
        file_match = re.search(r"File\s+Name\s*:\s*([^\r\n]+)", block, re.I)
        if not file_match:
            continue
        starts = [float(value) for value in re.findall(
            r"Seizure(?:\s+\d+)?\s+Start\s+Time\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            block, re.I,
        )]
        ends = [float(value) for value in re.findall(
            r"Seizure(?:\s+\d+)?\s+End\s+Time\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            block, re.I,
        )]
        count_match = re.search(
            r"Number\s+of\s+Seizures\s+in\s+File\s*:\s*(\d+)", block, re.I
        )
        if (not count_match or len(starts) != len(ends)
                or len(starts) != int(count_match.group(1))):
            raise ValueError(f"Invalid seizure metadata in {path}")
        intervals = list(zip(starts, ends))
        if any(not 0 <= start < end for start, end in intervals):
            raise ValueError(f"Invalid seizure interval in {path}")
        result[file_match.group(1).strip()] = intervals
    return result


def select_channels(ch_names: list[str]) -> list[str]:
    available = {name.upper(): name for name in ch_names}
    selected, missing = [], []
    for output_name, candidates in CHANNEL_CANDIDATES:
        match = next(
            (available[name.upper()] for name in candidates
             if name.upper() in available), None
        )
        if match is None:
            missing.append((output_name, candidates))
        else:
            selected.append(match)
    if missing:
        raise KeyError(f"Missing required bipolar channels: {missing}")
    return selected


def _electrode_candidates(electrode: str, reference: str | None):
    aliases = ELECTRODE_ALIASES.get(electrode, (electrode,))
    if reference is None:
        return aliases
    return tuple(f"{alias}-{reference}" for alias in aliases)


def _resolve_electrode(available, electrode, reference):
    return next((
        available[candidate.upper()]
        for candidate in _electrode_candidates(electrode, reference)
        if candidate.upper() in available
    ), None)


def build_bipolar_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Build the fixed montage from direct bipolar or referenced channels."""
    try:
        selected = select_channels(raw.ch_names)
    except KeyError:
        selected = None
    if selected is not None:
        result = raw.copy().pick(selected)
        result.reorder_channels(selected)
        result.rename_channels(dict(zip(selected, CHANNEL_ORDER)))
        return result

    available = {name.upper(): name for name in raw.ch_names}
    for reference in ("CS2", None):
        resolved = []
        for anode, cathode in ELECTRODE_PAIRS:
            anode_name = _resolve_electrode(available, anode, reference)
            cathode_name = _resolve_electrode(available, cathode, reference)
            if anode_name is None or cathode_name is None:
                break
            resolved.append((anode_name, cathode_name))
        if len(resolved) != len(ELECTRODE_PAIRS):
            continue
        source = raw.get_data()
        indices = {name: raw.ch_names.index(name) for pair in resolved for name in pair}
        data = np.stack([
            source[indices[anode]] - source[indices[cathode]]
            for anode, cathode in resolved
        ])
        info = mne.create_info(CHANNEL_ORDER, raw.info["sfreq"], ch_types="eeg")
        return mne.io.RawArray(data, info, verbose="ERROR")
    raise KeyError(f"Cannot build the required montage from {raw.ch_names}")


def read_and_preprocess_edf(edf_path: Path) -> np.ndarray:
    """Return filtered 16-channel data in microvolts at 200 Hz."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw = build_bipolar_raw(raw)
    raw.resample(TARGET_FS, verbose="ERROR")
    raw.filter(LOW_FREQ, HIGH_FREQ, verbose="ERROR")
    raw.notch_filter(freqs=NOTCH_FREQ, verbose="ERROR")
    data = raw.get_data(units="uV").astype(np.float32)
    if data.shape[0] != 16 or not np.isfinite(data).all():
        raise ValueError(f"Invalid preprocessed data: {data.shape}")
    return data


def overlap_seconds(start, end, intervals):
    return sum(max(0.0, min(end, right) - max(start, left))
               for left, right in intervals)


def process_one_file(task):
    edf_path = Path(task["edf_path"])
    output_dir = Path(task["output_dir"])
    patient_id = task["patient_id"]
    intervals = [tuple(item) for item in task["intervals"]]
    threshold = task["amplitude_threshold_uv"]
    data = read_and_preprocess_edf(edf_path)
    window_samples = int(WINDOW_SEC * TARGET_FS)
    case_dir = output_dir / "by_subject" / patient_id
    case_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()

    starts = range(0, data.shape[1] - window_samples + 1, window_samples)
    for index, start_sample in enumerate(starts):
        end_sample = start_sample + window_samples
        start_sec = start_sample / TARGET_FS
        end_sec = end_sample / TARGET_FS
        window = data[:, start_sample:end_sample]
        overlap = overlap_seconds(start_sec, end_sec, intervals)
        label = int(overlap > 0.0)
        if threshold is not None and np.max(np.abs(window)) > threshold:
            stats[f"rejected_label_{label}"] += 1
            continue
        record = {
            "X": window.astype(np.float32, copy=False),
            "y": label,
            "case_id": patient_id,
            "recording_id": edf_path.stem,
            "source_file": edf_path.name,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "seizure_overlap_sec": overlap,
            "sampling_rate": TARGET_FS,
            "channel_names": CHANNEL_ORDER,
            "unit": "uV",
        }
        with (case_dir / f"{edf_path.stem}_{index:05d}.pkl").open("wb") as stream:
            pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stats[f"saved_label_{label}"] += 1
    return {"file": edf_path.name, "patient_id": patient_id, **stats}


def _data_layout(root: Path):
    """Support both the official layout and the normalized server layout."""
    if (root / "recordings").is_dir():
        return root / "recordings", root / "summaries"
    return root, None


def build_tasks(root, output_dir, threshold):
    recordings_root, summaries_root = _data_layout(root)
    tasks, missing_summary_entries = [], []
    patient_dirs = sorted(recordings_root.glob("chb??"))
    if not patient_dirs:
        raise FileNotFoundError(f"No chb?? case directories found under {recordings_root}")
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        summary_path = (
            summaries_root / f"{patient_id}-summary.txt"
            if summaries_root else patient_dir / f"{patient_id}-summary.txt"
        )
        if not summary_path.is_file():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        annotations = parse_summary(summary_path)
        for edf_path in sorted(patient_dir.glob("*.edf")):
            if edf_path.name not in annotations:
                missing_summary_entries.append(str(edf_path))
                continue
            tasks.append({
                "edf_path": str(edf_path),
                "output_dir": str(output_dir),
                "patient_id": patient_id,
                "intervals": annotations[edf_path.name],
                "amplitude_threshold_uv": threshold,
            })
    return tasks, missing_summary_entries


def get_args():
    parser = argparse.ArgumentParser(description="CHB-MIT preprocessing")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amplitude-threshold-uv", type=float,
                        default=AMPLITUDE_THRESHOLD_UV)
    parser.add_argument("--no-amplitude-rejection", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"{args.output_dir} is not empty; use a new output directory"
        )
    threshold = None if args.no_amplitude_rejection else args.amplitude_threshold_uv
    tasks, missing = build_tasks(args.root, args.output_dir, threshold)
    results, errors = [], []
    if args.workers == 1:
        for task in tqdm(tasks, desc="CHB-MIT EDF files"):
            try:
                results.append(process_one_file(task))
            except Exception as exc:
                errors.append({"file": task["edf_path"], "error": repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one_file, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="CHB-MIT EDF files"):
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append({"file": futures[future]["edf_path"],
                                   "error": repr(exc)})

    case_ids = sorted({row["patient_id"] for row in results})
    summary = {
        "config": {
            "target_fs": TARGET_FS,
            "window_sec": WINDOW_SEC,
            "low_freq": LOW_FREQ,
            "high_freq": HIGH_FREQ,
            "notch_freq": NOTCH_FREQ,
            "amplitude_threshold_uv": threshold,
            "channel_order": CHANNEL_ORDER,
        },
        "case_ids": case_ids,
        "case_count": len(case_ids),
        "files_scheduled": len(tasks),
        "files_succeeded": len(results),
        "files_failed": len(errors),
        "edf_without_summary_entry": missing,
        "errors": errors,
    }
    with (args.output_dir / "preprocess_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError(f"{len(errors)} EDF files failed")


if __name__ == "__main__":
    main()
