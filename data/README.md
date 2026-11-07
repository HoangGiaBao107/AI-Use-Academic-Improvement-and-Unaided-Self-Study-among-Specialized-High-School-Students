# Input data

Place the original **`data raw eng.csv`** in this folder, or supply its path with `--data`.

The reviewed input has 324 rows and 58 columns. One record lacks all five CAU items. The complete-case analytical cohort contains 323 students. The original CSV is never overwritten.

Expected SHA-256:

```text
ded803a8b38c1e14b31835655c1ad2b54854e21ec05c191fdd5dd7bead471624
```

The pipeline stops if the bytes do not match this reviewed version. Copy the original file without resaving it through spreadsheet software.

## Access and scope

Respondent-level survey data are not distributed in this public repository. Authorized users can obtain the input from the study author. Git ignores this CSV and all row-level intermediates in `.work/`. Published results are aggregate tables, resampling summaries, and figures.

The organizer reported 522 invitations and 358 responses. Earlier qualitative exclusions were deleted before the retained export was finalized. This repository reproduces analysis from the retained English export; it cannot reconstruct the original 358 submissions or their individual exclusion reasons.

Students answered a self-developed Vietnamese questionnaire. English translations used in the export are listed in `results/tables/01_data/english_item_dictionary.csv` after running Phase 1. N3 concerns change in the ability to study without help, not the amount of study time. A7 refers to the previous school year, not a verified period before AI use.
