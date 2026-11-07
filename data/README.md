# Input data

The reviewed **`data raw eng.csv`** is already included in this folder. Cloning or downloading the complete repository includes this input; no second upload is required. The `--data` option is only needed to select a different local file location.

The reviewed input has 324 rows and 58 columns. One record lacks all five CAU items. The complete-case analytical cohort contains 323 students. The original CSV is never overwritten.

Expected SHA-256:

```text
ded803a8b38c1e14b31835655c1ad2b54854e21ec05c191fdd5dd7bead471624
```

The pipeline stops if the bytes do not match this reviewed version. Copy the original file without resaving it through spreadsheet software.

## Access and scope

The study author has included the reviewed survey export in this public repository. Only this explicitly named input is allowed by the data-folder Git exception; additional CSV exports and generated row-level intermediates in `.work/` remain ignored. Published analysis outputs under `results/` are aggregate tables, resampling summaries, and figures.

The organizer reported 522 invitations and 358 responses. Earlier qualitative exclusions were deleted before the retained export was finalized. This repository reproduces analysis from the retained English export; it cannot reconstruct the original 358 submissions or their individual exclusion reasons.

Students answered a self-developed Vietnamese questionnaire. English translations used in the export are listed in `results/tables/01_data/english_item_dictionary.csv` after running Phase 1. N3 concerns change in the ability to study without help, not the amount of study time. A7 refers to the previous school year, not a verified period before AI use.
