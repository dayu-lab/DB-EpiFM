import os
import pickle
import shutil
import random
from collections import Counter

import numpy as np

# =========================
# 你只需要改这里
# =========================
VAL_DIR = r"./data/processed/tusz/val"          # 原验证集目录
OUT_DIR = r"./data/processed/tusz/val_1to2"    # 输出新验证集目录（建议新建）
RATIO_ZERO_PER_ONE = 2   # 目标：label0 = label1 * 2  (等价于 label1:label0 = 1:2)
RANDOM_SEED = 3407
COPY_MODE = True         # True=复制到 OUT_DIR；False=移动到 OUT_DIR（会改动原数据）
# =========================


DEFAULT_LABEL_KEYS = [
    "label", "y", "target", "gt", "cls", "class",
    "labels", "targets", "y_true", "anno", "annotation"
]


def to_int01(v):
    if v is None:
        return None

    # torch-like -> numpy
    if hasattr(v, "detach"):
        v = v.detach()
    if hasattr(v, "cpu"):
        v = v.cpu()
    if hasattr(v, "numpy"):
        try:
            v = v.numpy()
        except Exception:
            pass

    if isinstance(v, np.ndarray):
        if v.size == 1:
            v = v.reshape(-1)[0]
        else:
            if v.ndim == 1 and v.shape[0] == 2:
                return int(np.argmax(v))
            return None

    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return to_int01(v[0])
        cand = to_int01(v[-1])
        if cand is not None:
            return cand
        if len(v) >= 2:
            return to_int01(v[1])
        return None

    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, np.integer)):
        iv = int(v)
        return iv if iv in (0, 1) else None
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        iv = int(round(fv))
        return iv if iv in (0, 1) and abs(fv - iv) < 1e-6 else None
    if isinstance(v, str):
        s = v.strip()
        if s in ("0", "1"):
            return int(s)
    return None


def extract_label(obj, candidate_keys=None):
    candidate_keys = candidate_keys or DEFAULT_LABEL_KEYS

    if isinstance(obj, dict):
        for k in candidate_keys:
            if k in obj:
                lab = to_int01(obj[k])
                if lab is not None:
                    return lab
        # nested
        for v in obj.values():
            if isinstance(v, (dict, list, tuple)):
                lab = extract_label(v, candidate_keys=candidate_keys)
                if lab is not None:
                    return lab
        return None

    if isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            lab = to_int01(obj[-1])
            if lab is not None:
                return lab
        if len(obj) >= 2:
            lab = to_int01(obj[1])
            if lab is not None:
                return lab
        for item in obj:
            lab = extract_label(item, candidate_keys=candidate_keys)
            if lab is not None:
                return lab
        return None

    return to_int01(obj)


def load_pkl(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def iter_pkl_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith(".pkl"):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, root_dir)
                yield full, rel


def ensure_parent(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    random.seed(RANDOM_SEED)

    zeros, ones = [], []
    failed = []

    print(f"Scanning val dir: {VAL_DIR}")
    for full, rel in iter_pkl_files(VAL_DIR):
        try:
            obj = load_pkl(full)
            lab = extract_label(obj)
            if lab == 0:
                zeros.append((full, rel))
            elif lab == 1:
                ones.append((full, rel))
            else:
                failed.append((full, "label_not_01_or_not_found"))
        except Exception as e:
            failed.append((full, f"load_error: {type(e).__name__}: {e}"))

    c0, c1 = len(zeros), len(ones)
    print("\n========== Original Val Stats ==========")
    print(f"Total pkl     : {c0 + c1 + len(failed)}")
    print(f"Label 0 count : {c0}")
    print(f"Label 1 count : {c1}")
    if c0 + c1 > 0:
        print(f"Ratio 0       : {c0/(c0+c1):.4f}")
        print(f"Ratio 1       : {c1/(c0+c1):.4f}")
    if failed:
        print(f"Unparsed/failed: {len(failed)} (will be ignored)")

    # Target keep count: label0 = label1 * RATIO_ZERO_PER_ONE
    target_c0 = c1 * RATIO_ZERO_PER_ONE

    if c1 == 0:
        print("\n[ERROR] No label=1 samples found. Cannot build 1:2 (label1:label0) set.")
        return

    if c0 <= target_c0:
        print("\n[WARN] label0 数量不够多（或已经<=目标），无法通过丢弃0达到精确 1:2。")
        keep_zeros = zeros  # keep all zeros
    else:
        keep_zeros = random.sample(zeros, target_c0)

    keep_ones = ones
    kept = keep_zeros + keep_ones
    random.shuffle(kept)

    new_c0, new_c1 = len(keep_zeros), len(keep_ones)
    print("\n========== New Val Stats (Target label1:label0 = 1:2) ==========")
    print(f"Keep label0   : {new_c0}")
    print(f"Keep label1   : {new_c1}")
    print(f"Total kept    : {new_c0 + new_c1}")
    print(f"New ratio 0   : {new_c0/(new_c0+new_c1):.4f}")
    print(f"New ratio 1   : {new_c1/(new_c0+new_c1):.4f}")
    print(f"Check 0 ~= 2*1: {new_c0} vs {2*new_c1}")

    # Write out
    os.makedirs(OUT_DIR, exist_ok=True)

    op = "COPY" if COPY_MODE else "MOVE"
    print(f"\nStart to {op} files to: {OUT_DIR}")

    for src_full, rel in kept:
        dst_full = os.path.join(OUT_DIR, rel)
        ensure_parent(dst_full)
        if COPY_MODE:
            shutil.copy2(src_full, dst_full)
        else:
            shutil.move(src_full, dst_full)

    # Save a manifest for traceability
    manifest_path = os.path.join(OUT_DIR, "manifest_kept.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for _, rel in kept:
            f.write(rel.replace("\\", "/") + "\n")

    print("\nDone.")
    print(f"- Output dir      : {OUT_DIR}")
    print(f"- Manifest saved  : {manifest_path}")


if __name__ == "__main__":
    main()
