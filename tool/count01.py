import os
import pickle
from collections import Counter

import numpy as np

# =========================
# 你只需要改这里
# =========================
TRAIN_DIR = r"./data/processed/tusz/val_1to2"   # 改成你的 train 文件夹路径
LABEL_KEY = None  # 如果你知道标签字段名，比如 "label" / "y" / "target"，就写成字符串；不知道就保持 None
SHOW_FAILED_EXAMPLES = 20
# =========================


DEFAULT_LABEL_KEYS = [
    "label", "y", "target", "gt", "cls", "class",
    "labels", "targets", "y_true", "anno", "annotation"
]


def to_int01(v):
    """Convert various types to int label {0,1}. Return None if not possible."""
    if v is None:
        return None

    # torch tensor -> numpy (without importing torch)
    if hasattr(v, "detach"):
        v = v.detach()
    if hasattr(v, "cpu"):
        v = v.cpu()
    if hasattr(v, "numpy"):
        try:
            v = v.numpy()
        except Exception:
            pass

    # numpy array
    if isinstance(v, np.ndarray):
        if v.size == 1:
            v = v.reshape(-1)[0]
        else:
            # one-hot/prob of 2 classes
            if v.ndim == 1 and v.shape[0] == 2:
                return int(np.argmax(v))
            return None

    # list/tuple
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return to_int01(v[0])
        cand = to_int01(v[-1])  # (..., y)
        if cand is not None:
            return cand
        if len(v) >= 2:
            return to_int01(v[1])  # (x, y)
        return None

    # bool / int
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, np.integer)):
        iv = int(v)
        return iv if iv in (0, 1) else None

    # float (0.0/1.0)
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        iv = int(round(fv))
        return iv if iv in (0, 1) and abs(fv - iv) < 1e-6 else None

    # str "0"/"1"
    if isinstance(v, str):
        s = v.strip()
        if s in ("0", "1"):
            return int(s)

    return None


def extract_label(obj, label_key=None, candidate_keys=None):
    """Try to extract label from pkl object. Return int(0/1) or None."""
    candidate_keys = candidate_keys or DEFAULT_LABEL_KEYS

    # 1) user specified key
    if label_key is not None:
        if isinstance(obj, dict) and label_key in obj:
            return to_int01(obj[label_key])
        if isinstance(obj, (list, tuple)):
            for item in obj:
                lab = extract_label(item, label_key=label_key, candidate_keys=candidate_keys)
                if lab is not None:
                    return lab

    # 2) dict common keys
    if isinstance(obj, dict):
        for k in candidate_keys:
            if k in obj:
                lab = to_int01(obj[k])
                if lab is not None:
                    return lab
        # nested search
        for v in obj.values():
            if isinstance(v, (dict, list, tuple)):
                lab = extract_label(v, label_key=None, candidate_keys=candidate_keys)
                if lab is not None:
                    return lab
        return None

    # 3) tuple/list common conventions
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
            lab = extract_label(item, label_key=None, candidate_keys=candidate_keys)
            if lab is not None:
                return lab
        return None

    # 4) scalar
    return to_int01(obj)


def load_pkl(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def main():
    counter = Counter()
    total = 0
    failed = []

    for root, _, files in os.walk(TRAIN_DIR):
        for fn in files:
            if not fn.endswith(".pkl"):
                continue
            total += 1
            p = os.path.join(root, fn)
            try:
                obj = load_pkl(p)
                lab = extract_label(obj, label_key=LABEL_KEY)
                if lab in (0, 1):
                    counter[lab] += 1
                else:
                    failed.append((p, "label_not_found_or_not_01"))
            except Exception as e:
                failed.append((p, f"load_error: {type(e).__name__}: {e}"))

    n0 = counter.get(0, 0)
    n1 = counter.get(1, 0)
    known = n0 + n1

    print("========== TUSZ Train Label Stats ==========")
    print(f"Train dir     : {TRAIN_DIR}")
    print(f"Total pkl     : {total}")
    print(f"Count(label=0): {n0}")
    print(f"Count(label=1): {n1}")
    print(f"Known labels  : {known} (parsed as 0/1)")
    if known > 0:
        print(f"Ratio 0       : {n0/known:.4f}")
        print(f"Ratio 1       : {n1/known:.4f}")
        print(f"Imbalance 1/0 : {(n1/max(n0,1)):.4f}")
    else:
        print("No 0/1 labels parsed.")
        print("-> 你可以把 LABEL_KEY 设置成你的标签字段名，比如 LABEL_KEY='label'。")

    if failed:
        print("\n========== Failed / Unparsed Files ==========")
        print(f"Failed count  : {len(failed)}")
        for p, msg in failed[:SHOW_FAILED_EXAMPLES]:
            print(f"- {p}  [{msg}]")
        if len(failed) > SHOW_FAILED_EXAMPLES:
            print(f"... ({len(failed) - SHOW_FAILED_EXAMPLES} more not shown)")


if __name__ == "__main__":
    main()
