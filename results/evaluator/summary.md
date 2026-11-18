# Evaluator-partition run summary

Artifacts used unchanged: `artifacts/model.joblib`, `config.json`, `hmm.json` (beta=50). No training, tuning, feature selection, or threshold changes used evaluator data.

- Trips: 10
- Observations: 19,674
- Minimum GPS-to-route coverage: 99.6936%
- Reference-route discontinuities: 0
- Repeated-arc trips: 2
- API e2e: 10/10 HTTP 200; mean 2849 ms, max 6959 ms

## Mean decoder metrics (all 10 trips)

| method | point_accuracy | route_continuity | route_precision | route_recall | rmf | alignment_coverage | inference_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| classifier | 0.8275 | 0.9998 | 0.2241 | 0.9912 | 14.1316 | 0.9997 | 549.6578 |
| hmm | 0.8519 | 0.9998 | 0.5621 | 0.9921 | 6.2483 | 0.9997 | 2109.0814 |
| nearest | 0.5321 | 0.9999 | 0.1996 | 0.9914 | 11.7505 | 0.9997 | 542.1045 |

## Mean decoder metrics excluding 00000015 (reference-route length anomaly)

| method | point_accuracy | route_continuity | route_precision | route_recall | rmf | alignment_coverage | inference_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| classifier | 0.8612 | 0.9998 | 0.2461 | 0.991 | 8.017 | 1.0 | 602.2965 |
| hmm | 0.8891 | 0.9998 | 0.6212 | 0.9913 | 0.8988 | 1.0 | 2168.125 |
| nearest | 0.5487 | 0.9998 | 0.2126 | 0.9905 | 11.8633 | 1.0 | 595.3469 |

## Per-trip HMM

| trip_id | hmm_point_accuracy | route_continuity | alignment_coverage | inference_ms |
| --- | --- | --- | --- | --- |
| 00000015 | 0.5166 | 1.0 | 0.9969 | 3407.7521 |
| 00000017 | 0.8707 | 1.0 | 1.0 | 6958.5971 |
| 00000023 | 0.9236 | 1.0 | 1.0 | 686.3819 |
| 00000030 | 0.8226 | 0.9982 | 1.0 | 2372.75 |
| 00000049 | 0.9773 | 1.0 | 1.0 | 2177.3817 |
| 00000059 | 0.937 | 1.0 | 1.0 | 3278.6479 |
| 00000067 | 0.7904 | 1.0 | 1.0 | 638.1062 |
| 00000073 | 0.9221 | 1.0 | 1.0 | 1840.4832 |
| 00000087 | 0.9871 | 1.0 | 1.0 | 3409.2667 |
| 00000094 | 0.7712 | 1.0 | 1.0 | 3722.3636 |

## Anomaly flag

`00000015`: reference route 8.67 km vs GPS track 53.31 km (ratio 0.163); geometric coverage is 99.69% but the reference covers only about one traversal of a repeatedly driven route. This depresses point accuracy on later passes and is a label/reference limitation, not a model failure. It was not used to tune anything.
