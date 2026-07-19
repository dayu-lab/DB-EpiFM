# Data Leakage Audit Supplement

This folder contains the patient-level data leakage audit for DB-EpiFM.

## What is included

- `Data_Leakage_Audit_Report.md`: GitHub-readable audit report.
- `Data_Leakage_Audit_Report.pdf`: PDF version of the audit report.
- `Data_Leakage_Audit_Report.tex`: LaTeX source of the audit report.
- `audit_summary.json`: machine-readable summary statistics.
- `tables/`: aggregate audit tables in TSV and CSV formats.
- `manifests_sanitized/`: sanitized patient-hash and dataset-relative path manifests.

## What is not included

This repository does **not** redistribute TUH raw EDF data, derived EEG windows, LMDB databases, raw patient identifiers, or local absolute paths.

## Key result

After excluding patients overlapping with TUAB or TUEV, the final TUEP/TUSZ pretraining corpus contained **162,617** non-overlapping 30-s EEG windows, corresponding to approximately **1355.1 h** of recordings.
