# Feature selection and ablation

Selection used trip-grouped CV Macro F1 on the training trips only. Student-test data was not used to choose features.

- Full feature set: 10 features, CV Macro F1 = 0.8671
- Selected feature set: 6 features, CV Macro F1 = 0.8666
- Selected features: `distance_m, heading_diff_deg, candidate_rank, along_fraction, from_degree, to_degree`

## Leave-one-feature-out importance

Positive delta means removing the feature hurt CV Macro F1.

| Removed feature | CV Macro F1 without it | Delta vs full |
| :--- | ---: | ---: |
| heading_diff_deg | 0.7701 | 0.0970 |
| candidate_rank | 0.8460 | 0.0211 |
| along_fraction | 0.8499 | 0.0172 |
| from_degree | 0.8631 | 0.0039 |
| to_degree | 0.8640 | 0.0031 |
| time_gap_s | 0.8659 | 0.0012 |
| speed_consistency | 0.8660 | 0.0011 |
| arc_length_m | 0.8661 | 0.0009 |
| log_distance | 0.8664 | 0.0006 |
| distance_m | 0.8673 | -0.0002 |

## Cumulative feature groups

| Feature group reached | n features | CV Macro F1 |
| :--- | ---: | ---: |
| distance | 2 | 0.6413 |
| heading | 3 | 0.7473 |
| speed | 4 | 0.7580 |
| candidate_context | 7 | 0.8548 |
| temporal | 8 | 0.8561 |
| topology | 10 | 0.8671 |
