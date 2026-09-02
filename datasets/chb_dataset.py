import json
import os
import pickle

import numpy as np
from scipy import signal
from torch.utils.data import DataLoader, Dataset

from utils.util import to_tensor


def _default_cases(subject_id):
    return [f"chb{int(subject_id.split('_')[-1]):02d}"]


def load_fold(split_manifest, fold_id):
    with open(split_manifest, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    folds = {int(item["fold"]): item for item in manifest["folds"]}
    if fold_id not in folds:
        raise ValueError(f"Fold {fold_id} is not present in {split_manifest}")

    fold = folds[fold_id]
    mapping = manifest.get("subject_mapping", {})
    partitions = {name: set(fold[name]) for name in ("train", "val", "test")}
    if partitions["train"] & partitions["val"]:
        raise ValueError("CHB-MIT train and validation patient groups overlap")
    if partitions["train"] & partitions["test"]:
        raise ValueError("CHB-MIT train and test patient groups overlap")
    if partitions["val"] & partitions["test"]:
        raise ValueError("CHB-MIT validation and test patient groups overlap")

    def expand(subjects):
        cases = []
        for subject in subjects:
            cases.extend(mapping.get(subject, _default_cases(subject)))
        return cases

    return {name: expand(fold[name]) for name in ("train", "val", "test")}


class CustomDataset(Dataset):
    def __init__(self, data_dir, cases):
        super().__init__()
        self.files = []
        subject_root = os.path.join(data_dir, "by_subject")
        for case_id in cases:
            case_dir = os.path.join(subject_root, case_id)
            if not os.path.isdir(case_dir):
                raise FileNotFoundError(
                    f"Processed CHB-MIT case directory not found: {case_dir}"
                )
            self.files.extend(
                os.path.join(case_dir, filename)
                for filename in sorted(os.listdir(case_dir))
                if filename.endswith(".pkl")
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as stream:
            data_dict = pickle.load(stream)
        data = signal.resample(data_dict["X"], 2000, axis=1)
        return data.reshape(16, 10, 200) / 100, data_dict["y"]

    @staticmethod
    def collate(batch):
        return (
            to_tensor(np.asarray([item[0] for item in batch])),
            to_tensor(np.asarray([item[1] for item in batch])),
        )


class LoadDataset:
    def __init__(self, params):
        self.params = params

    def get_data_loader(self):
        cases = load_fold(self.params.split_manifest, self.params.fold)
        datasets = {
            name: CustomDataset(self.params.datasets_dir, cases[name])
            for name in ("train", "val", "test")
        }
        print(
            f"CHB-MIT fold {self.params.fold}: train={len(datasets['train'])}, "
            f"val={len(datasets['val'])}, test={len(datasets['test'])}"
        )
        return {
            "train": DataLoader(datasets["train"], batch_size=self.params.batch_size,
                                collate_fn=datasets["train"].collate, shuffle=True,
                                num_workers=self.params.num_workers),
            "val": DataLoader(datasets["val"], batch_size=self.params.batch_size,
                              collate_fn=datasets["val"].collate, shuffle=False,
                              num_workers=self.params.num_workers),
            "test": DataLoader(datasets["test"], batch_size=self.params.batch_size,
                               collate_fn=datasets["test"].collate, shuffle=False,
                               num_workers=self.params.num_workers),
        }
