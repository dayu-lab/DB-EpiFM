# Supplementary Data Leakage Audit for DB-EpiFM

This document summarizes the patient-level overlap audit used to prevent leakage between the self-supervised pretraining corpus and the downstream TUH-family evaluation datasets.

## Scope

- Pretraining sources: TUEP and TUSZ.
- Downstream TUH-family datasets audited for patient overlap: TUAB and TUEV.
- Exclusion rule: any patient appearing in TUAB or TUEV was completely excluded from the TUEP/TUSZ pretraining corpus before signal preprocessing.
- Unit of exclusion: patient level, not recording level or window level.

## Patient-level audit protocol

Patient identifiers were extracted from the TUH directory structure and EDF file names. The exclusion list was constructed as the union of patient identifiers appearing in TUAB and TUEV. For TUEP and TUSZ, every EDF recording whose patient identifier appeared in this downstream exclusion list was removed before signal preprocessing. This procedure ensures strict patient-level independence between the pretraining corpus and the downstream TUH-family evaluation datasets.

## Key audit results

- Downstream patient exclusion list: **887** patient identifiers.
- Actual overlapping pretraining patients removed from TUEP/TUSZ: **101** patients.
- EDF recordings removed from the pretraining corpus: **2195**.
- Files with unrecognized patient identifiers: **0**.
- Signal-processing errors in the final preprocessing run: **0**.
- Final pretraining corpus: **162,617** non-overlapping 30-s windows.
- Estimated EEG duration after preprocessing: **1355.1 h** (56.46 days).

## Table S1. Patient-level overlap removal by pretraining source

| corpus | excluded_edf_recordings | unique_excluded_pretraining_patients |
| --- | --- | --- |
| TUEP | 945 | 28 |
| TUSZ | 1250 | 81 |
| TUEP+TUSZ | 2195 | 101 |


## Table S2. Source of patient-level overlap

| overlap_with | excluded_edf_recordings | unique_excluded_pretraining_patients |
| --- | --- | --- |
| TUAB | 2143 | 89 |
| TUEV | 25 | 7 |
| TUAB+TUEV | 27 | 5 |


## Table S3. Final pretraining corpus summary

| item | value |
| --- | --- |
| downstream_patient_ids_exclusion_list | 887 |
| actual_excluded_pretraining_patients | 101 |
| files_excluded_by_patient_overlap | 2195 |
| files_kept_before_signal_qc | 7990 |
| files_skipped_bad_patient_id | 0 |
| files_successfully_processed | 7217 |
| files_failed_signal_processing | 0 |
| files_with_no_valid_window | 773 |
| samples_total_30s_windows | 162617 |
| samples_rejected_by_amplitude_qc | 13529 |
| samples_TUEP_30s_windows | 37672 |
| samples_TUSZ_30s_windows | 124945 |
| total_duration_hours | 1355.14 |
| total_duration_days | 56.46 |


## Table S4. Excluded EDF recordings by pretraining source and downstream overlap source

| corpus | TUAB | TUEV | TUAB+TUEV | total |
| --- | --- | --- | --- | --- |
| TUEP | 941 | 0 | 4 | 945 |
| TUSZ | 1202 | 25 | 23 | 1250 |


## Released sanitized manifests

The folder `manifests_sanitized/` contains three release-safe TSV files:

1. `excluded_patient_hashes.tsv`: downstream exclusion-list patients represented as within-audit pseudonymous hashes.
2. `excluded_overlap_sanitized.tsv`: EDF-level exclusion manifest for TUEP/TUSZ after replacing raw patient identifiers and local absolute paths with sanitized values.
3. `excluded_dataset_relative_paths_sanitized.tsv`: dataset-relative EDF path list for excluded pretraining records, with patient identifiers replaced by pseudonymous patient hashes.

Raw patient identifiers, local absolute paths, raw EDF files, and preprocessed LMDB databases are not redistributed in this repository.

## Notes on de-identification

The patient hashes are pseudonymous identifiers generated for this audit release. They are intended to preserve within-audit consistency across tables and manifests while avoiding publication of raw patient identifiers. The private audit salt used to generate these hashes is not included in the repository.
