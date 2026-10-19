#!/usr/bin/env python3
"""Reproduce the manuscript's run-level comparisons from seed-level scores.

The input must contain measured run-level scores. This script never generates,
reorders, or changes scores to match published tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from itertools import product
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import scipy
from scipy.stats import friedmanchisquare, rankdata


DATASET_METRICS = {
    "TUAB": ("Balanced Accuracy", "AUC-PR", "AUROC"),
    "CHB-MIT": ("Balanced Accuracy", "Sensitivity", "Specificity"),
    "TUEV": ("Balanced Accuracy", "Cohen's Kappa", "Weighted F1"),
}
BASELINES = (
    "EEGNet", "ST-Transformer", "BIOT", "EEGPT",
    "LaBraM", "CBraMod", "EpilepsyFM",
)
TARGET = "DB-EpiFM"
MODELS = (*BASELINES, TARGET)
NORMAL = NormalDist()


def canonical_model(value: object) -> str:
    name = str(value).strip()
    aliases = {"EEGNET": "EEGNet", "ST-Transforme": "ST-Transformer"}
    return aliases.get(name, name)


def canonical_seed(value: object) -> str:
    label = str(value).strip()
    if label.lower().startswith("seed "):
        label = label[5:].strip()
    if not label:
        raise ValueError("Empty seed label")
    return label


def load_csv(path: Path, unit: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    data = pd.read_csv(
        path, dtype={"dataset": str, "model": str, "metric": str, "seed": str},
        float_precision="round_trip",
    )
    required = {"dataset", "model", "metric", "seed", "score"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    data = data[["dataset", "model", "metric", "seed", "score"]].copy()
    data["seed"] = data["seed"].map(canonical_seed)
    data["model"] = data["model"].map(canonical_model)
    data["score"] = pd.to_numeric(data["score"], errors="raise")
    if unit == "fraction":
        data["score"] *= 100.0
    seeds = list(dict.fromkeys(data["seed"].tolist()))
    return data, seeds, []


def load_wide_xlsx(path: Path, mean_tolerance: float) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Read 13-column seed scores, optionally with a Mean column.

    TUAB scores are fractions in the source XLSX; CHB-MIT and TUEV are percent.
    An optional Mean column is checked but never used in a calculation.
    """
    frame = pd.read_excel(path, sheet_name=0, engine="openpyxl", header=0)
    if list(frame.columns[:3]) != ["Dataset", "Model", "Metric"]:
        raise ValueError("Expected Dataset, Model, Metric as the first three columns")
    has_mean = len(frame.columns) == 14 and frame.columns[3] == "Mean"
    if len(frame.columns) != (14 if has_mean else 13):
        raise ValueError("Expected ten seed columns after Dataset, Model, Metric, and optional Mean")
    seed_start = 4 if has_mean else 3
    seed_columns = list(frame.columns[seed_start:])
    seeds = [canonical_seed(value) for value in seed_columns]
    if len(set(seeds)) != 10:
        raise ValueError("The ten seed column labels must be unique")
    records: list[dict] = []
    mismatches: list[str] = []
    for row in frame.itertuples(index=False, name=None):
        dataset, model, metric = row[:3]
        if pd.isna(dataset):
            continue
        dataset, model, metric = str(dataset).strip(), canonical_model(model), str(metric).strip()
        scale = 100.0 if dataset == "TUAB" else 1.0
        scores = np.asarray(row[seed_start:], dtype=float) * scale
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"Missing/nonfinite score: {dataset}/{model}/{metric}")
        if has_mean and pd.notna(row[3]):
            claimed = float(row[3]) * scale
            computed = float(scores.mean())
            if abs(claimed - computed) > mean_tolerance:
                mismatches.append(
                    f"{dataset}/{model}/{metric}: Mean cell={claimed:.6f}, "
                    f"seed mean={computed:.6f} percentage points"
                )
        records.extend(
            {"dataset": dataset, "model": model, "metric": metric,
             "seed": seed, "score": float(score)}
            for seed, score in zip(seeds, scores)
        )
    return pd.DataFrame.from_records(records), seeds, mismatches


def validate(data: pd.DataFrame, seeds: list[str]) -> None:
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("Exactly ten distinct seed labels are required")
    duplicates = data.duplicated(["dataset", "model", "metric", "seed"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate dataset/model/metric/seed records:\n"
                         + data.loc[duplicates].to_string(index=False))
    if not np.isfinite(data["score"].to_numpy(dtype=float)).all():
        raise ValueError("All scores must be finite")
    if set(data["dataset"]) != set(DATASET_METRICS):
        raise ValueError(f"Expected datasets {list(DATASET_METRICS)}, got {sorted(set(data['dataset']))}")
    for dataset, metrics in DATASET_METRICS.items():
        for metric in metrics:
            subset = data[(data.dataset == dataset) & (data.metric == metric)]
            if set(subset.model) != set(MODELS):
                raise ValueError(f"{dataset}/{metric}: expected exactly {MODELS}; "
                                 f"found {sorted(set(subset.model))}")
            for model in MODELS:
                found = set(subset.loc[subset.model == model, "seed"])
                if found != set(seeds):
                    raise ValueError(f"{dataset}/{metric}/{model}: missing or extra seeds: {found ^ set(seeds)}")
    if len(data) != 3 * 3 * 8 * 10:
        raise ValueError(f"Expected 720 seed-level records; found {len(data)}")


def exact_signed_rank(differences: np.ndarray) -> dict[str, float | int]:
    """Two-sided sign-enumeration p-value with midranks and Wilcox zeros.

    The two-sided convention is 2*min(lower_tail, upper_tail), capped at 1.
    For ten nonzero pairs this enumerates exactly 2**10 sign assignments.
    """
    values = np.asarray(differences, dtype=float)
    nonzero = values[values != 0.0]
    n = len(nonzero)
    if n == 0:
        return {"p_exact": 1.0, "rrb": float("nan"), "w_plus": 0.0,
                "w_minus": 0.0, "n_nonzero": 0, "n_zero": len(values)}
    ranks = rankdata(np.abs(nonzero), method="average")
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    sign_sums = np.asarray(
        [np.dot(ranks, signs) for signs in product((-1.0, 1.0), repeat=n)]
    )
    possible_w_plus = (sign_sums + ranks.sum()) / 2.0
    p_low = np.mean(possible_w_plus <= w_plus + 1e-12)
    p_high = np.mean(possible_w_plus >= w_plus - 1e-12)
    return {"p_exact": float(min(1.0, 2.0 * min(p_low, p_high))),
            "rrb": float((w_plus - w_minus) / (w_plus + w_minus)),
            "w_plus": w_plus, "w_minus": w_minus,
            "n_nonzero": n, "n_zero": len(values) - n}


def holm_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.empty_like(values)
    running = 0.0
    for i, index in enumerate(np.argsort(values, kind="stable")):
        running = max(running, min(1.0, (len(values) - i) * values[index]))
        result[index] = running
    return result


def bca_mean_ci(differences: np.ndarray, rng: np.random.Generator,
                resamples: int) -> tuple[float, float]:
    """95% BCa interval for the paired mean using run-level resampling.

    One shared RNG advances through all comparisons in fixed manuscript order.
    This reproduces the manuscript's historical 10,000-resample calculation.
    """
    values = np.asarray(differences, dtype=float)
    n = len(values)
    observed = float(values.mean())
    boot = values[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
    if np.all(values == values[0]):
        return observed, observed
    proportion = float(np.mean(boot < observed))
    proportion = min(max(proportion, 1.0 / (2 * resamples)), 1.0 - 1.0 / (2 * resamples))
    z0 = NORMAL.inv_cdf(proportion)
    jack = np.asarray([(values.sum() - values[i]) / (n - 1) for i in range(n)])
    u = jack.mean() - jack
    denominator = 6.0 * (np.sum(u * u) ** 1.5)
    acceleration = 0.0 if denominator == 0 else float(np.sum(u ** 3) / denominator)
    quantiles = []
    for alpha in (0.025, 0.975):
        z = NORMAL.inv_cdf(alpha)
        numerator = z0 + z
        quantiles.append(NORMAL.cdf(z0 + numerator / (1.0 - acceleration * numerator)))
    ci = np.quantile(boot, np.clip(quantiles, 0.0, 1.0), method="linear")
    return float(ci[0]), float(ci[1])


def run_analysis(data: pd.DataFrame, seeds: list[str], resamples: int,
                 bootstrap_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Match the previously declared seed order and model comparison order.
    rng = np.random.default_rng(bootstrap_seed)
    summaries: list[dict] = []
    all_comparisons: list[dict] = []
    displayed: list[dict] = []
    for dataset, metrics in DATASET_METRICS.items():
        for metric in metrics:
            subset = data[(data.dataset == dataset) & (data.metric == metric)]
            scores = {
                model: subset.loc[subset.model == model]
                .set_index("seed").loc[seeds, "score"].to_numpy(dtype=float)
                for model in MODELS
            }
            for model in MODELS:
                values = scores[model]
                summaries.append({"dataset": dataset, "model": model, "metric": metric,
                                  "n_runs": len(values), "mean_percent": float(values.mean()),
                                  "sd_percent": float(values.std(ddof=1))})
            statistic, omnibus_p = friedmanchisquare(*(scores[m] for m in MODELS))
            kendall_w = float(statistic / (len(seeds) * (len(MODELS) - 1)))
            strongest = max(BASELINES, key=lambda model: float(scores[model].mean()))
            comparisons: list[dict] = []
            for baseline in BASELINES:
                d = scores[TARGET] - scores[baseline]
                test = exact_signed_rank(d)
                ci_low, ci_high = bca_mean_ci(d, rng, resamples)
                correlation = float(np.corrcoef(scores[TARGET], scores[baseline])[0, 1])
                comparisons.append({
                    "dataset": dataset, "metric": metric, "baseline": baseline,
                    "delta_pp": float(d.mean()), "ci_low_pp": ci_low,
                    "ci_high_pp": ci_high, "p_exact": test["p_exact"],
                    "rrb": test["rrb"], "w_plus": test["w_plus"],
                    "w_minus": test["w_minus"], "n_nonzero": test["n_nonzero"],
                    "n_zero": test["n_zero"], "paired_sd_pp": float(d.std(ddof=1)),
                    "seed_score_correlation": correlation,
                    "friedman_chi2": float(statistic), "friedman_p": float(omnibus_p),
                    "kendall_w": kendall_w,
                })
            adjusted = holm_adjust([row["p_exact"] for row in comparisons])
            for row, p_adj in zip(comparisons, adjusted):
                row["p_holm"] = float(p_adj)
                row["omnibus_significant"] = bool(omnibus_p < 0.05)
                row["posthoc_significant"] = bool(omnibus_p < 0.05 and p_adj < 0.05)
                all_comparisons.append(row)
                if row["baseline"] == strongest:
                    displayed.append(row.copy())
    return pd.DataFrame(summaries), pd.DataFrame(all_comparisons), pd.DataFrame(displayed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    analysis_dir = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=analysis_dir / "seed_level_scores.csv",
                        help="Long CSV or 13/14-column wide XLSX of measured seed-level scores")
    parser.add_argument("--output-dir", type=Path, default=analysis_dir / "analysis_outputs")
    parser.add_argument("--csv-unit", choices=("percent", "fraction"), default="percent",
                        help="Unit of the CSV score column; XLSX uses its documented mixed units")
    parser.add_argument("--bootstrap-seed", type=int, default=20261006)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--mean-tolerance-pp", type=float, default=0.005,
                        help="Warn if an XLSX Mean cell differs from seed mean by more than this")
    args = parser.parse_args()
    if args.resamples < 1000 or args.mean_tolerance_pp < 0:
        parser.error("--resamples must be >= 1000 and --mean-tolerance-pp must be nonnegative")
    if args.input.suffix.lower() == ".csv":
        data, seeds, mismatches = load_csv(args.input, args.csv_unit)
    elif args.input.suffix.lower() == ".xlsx":
        data, seeds, mismatches = load_wide_xlsx(args.input, args.mean_tolerance_pp)
    else:
        parser.error("Input must be .csv or .xlsx")
    validate(data, seeds)
    summaries, comparisons, table7 = run_analysis(data, seeds, args.resamples, args.bootstrap_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Publish these files only after verifying the scores against run logs.
    data.to_csv(args.output_dir / "seed_level_scores.csv", index=False, float_format="%.10g")
    summaries.to_csv(args.output_dir / "model_mean_sd.csv", index=False, float_format="%.10g")
    comparisons.to_csv(args.output_dir / "all_pairwise_comparisons.csv", index=False,
                       float_format="%.10g")
    table7.to_csv(args.output_dir / "table7_comparisons.csv", index=False, float_format="%.10g")
    metadata = {
        "input_file": args.input.name,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "seed_order": seeds, "model_order": list(MODELS),
        "score_unit_in_outputs": "percentage points",
        "bootstrap_resamples": args.resamples, "bootstrap_rng_seed": args.bootstrap_seed,
        "exact_test": "two-sided, 2*min(tails), full sign enumeration, average ranks, Wilcox zeros",
        "multiple_testing": "Holm adjustment over seven DB-EpiFM versus baseline comparisons per dataset-metric",
        "mean_cell_mismatches": mismatches,
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for mismatch in mismatches:
        warnings.warn("XLSX Mean cell ignored: " + mismatch, stacklevel=1)
    print(table7[["dataset", "metric", "baseline", "delta_pp", "ci_low_pp",
                  "ci_high_pp", "p_holm", "rrb", "seed_score_correlation"]]
          .to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nWrote analysis to {args.output_dir}")
    print(f"XLSX Mean-cell mismatches: {len(mismatches)}")


if __name__ == "__main__":
    main()
