# Claim-to-code traceability

This map concerns the current report's conclusions and the numerical support required to evaluate them. All paths are relative to the repository root.

| Report claim or evidence | Source module / phase | Recomputed output |
|---|---|---|
| Participant flow, 522 → 358 → 323 | `measurement.prepare`, Phase 1 | `results/tables/01_data/participant_flow.csv`; Figure 01. Recruitment counts are organizer-supplied metadata. |
| Cohort, academic-rating concentration, specialization distribution | `measurement.describe`, Phase 2 | `sample_distributions.csv`, `sparse_covariate_cells.csv`. |
| Outcome mismatch and paired combinations | Phase 2 | `outcome_distributions.csv`, `joint_improvement.csv`, `joint_decline.csv`; Figure 03. |
| English item wording and analytic variable roles | Phase 1 and `data.py` | `english_item_dictionary.csv`; administered language and A7 interpretation in `data_manifest.json`. |
| Quality flags and sensitivity samples | `data.py`, `measurement.py` | `quality_flags.csv`; initial qualitative exclusions cannot be reconstructed. |
| Reliability and the 24 descriptive correlations | Phase 2 | `scale_reliability.csv`, `descriptive_correlations.csv`. |
| C1a / C1b association and sensitivity | Phase 3 | `cau_sensitivity.csv`, `main_models.csv`. |
| C2 whole-cohort and conditional associations | Phase 3 | `c2_sensitivity.csv`. |
| C3 continuous-AOR association and ten specifications | Phase 3 | `c3_sensitivity.csv`, `c3_coefficients.csv`. |
| C3 component models | Phases 3–4 | `c3_components.csv`, `component_results.csv`. No formal between-outcome coefficient-difference claim. |
| Standardized probabilities and bootstrap intervals | Phase 4 | `bootstrap_intervals.csv`, `07_report_outputs/main_results.csv`; Figures 02 and 05. |
| CAU–AOR co-occurrence | Phases 2 and 4 | `cau_aor_correlation.csv`, `cau_aor_score_pairs.csv`; Figure 04. |
| C3 plug-in null simulation | Phase 4 | `c3_parametric_null.csv`. |
| Multiplicity scopes and historical sensitivity | Phase 3 | `multiplicity_registry.csv`, `association_specification.json`; frozen out-of-scope p-values in `provenance/historical_tests.csv`. |
| C4 primary ten and broader 45 log-loss comparisons | Phase 5 | `primary_logloss_comparisons.csv`, `paired_predictive_gains.csv`. |
| Nested selection and leakage checks | Phase 5 | `nested_selection_metrics.csv`, `nested_selected_candidates.csv`, `nested_leakage_audit.csv`, `fold_audit.csv`. |
| Calibration values and plot | Phase 5, separate diagnostic | `calibration.csv`, `calibration_bins.csv`; Figure 06. |
| C5 primary interaction, ten specifications and resampling | Phase 6 | `continuous_interaction_tests.csv`, `interaction_resampling.csv`. |
| C6 school/classroom items and practical priorities | Phases 2 and 6 | `school_classroom_context.csv`; Figure 07. These are perceptions, not intervention effects. |
| Final synthesis | Phase 7 | `evidence_summary.csv`; numerical assessments are checked against the outputs above. |

## Historical material outside this focused reproduction

The older three-profile clustering results and the RAI4 verification-behavior candidate remain historical supplementary context in the manuscript. This release does not refit those branches or present them as independent support for the conclusions. Their p-values are preserved only when needed for the declared historical BH-FDR scopes. A reader requiring full historical notebooks should request the original exploration archive separately.

The numerical reference snapshots under `provenance/reference/` serve validation only. Removing that directory does not prevent the analysis pipeline from computing results. It does prevent checks against the archived report evidence.

## Interpretation boundaries

The sample comprises competition-team students at one school. N1 and N3 are retrospective self-reports with different reference periods. AOR wording overlaps with perceived independence. Robustness checks reuse the same cohort. Statistical consistency does not establish causality, scale validity, representative sampling, or successful classroom intervention.
