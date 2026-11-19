# Decoder failure analysis

Comparison of classifier-only predictions with classifier+HMM decoding. Positive deltas mean HMM improved the metric.

| Trip | Classifier accuracy | HMM accuracy | Δ accuracy | Classifier continuity | HMM continuity | Δ continuity |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00000000 | 0.9221 | 0.9860 | +0.0639 | 0.9820 | 0.9907 | +0.0086 |
| 00000007 | 0.9010 | 1.0000 | +0.0990 | 0.9931 | 1.0000 | +0.0069 |
| 00000009 | 0.8093 | 0.9399 | +0.1306 | 1.0000 | 1.0000 | +0.0000 |
| 00000040 | 0.8661 | 0.9496 | +0.0835 | 1.0000 | 1.0000 | +0.0000 |
| 00000052 | 0.9198 | 1.0000 | +0.0802 | 1.0000 | 1.0000 | +0.0000 |
| 00000056 | 0.8527 | 0.9888 | +0.1361 | 0.9995 | 1.0000 | +0.0005 |
| 00000065 | 0.7734 | 0.9596 | +0.1862 | 1.0000 | 0.9951 | -0.0049 |
| 00000076 | 0.7631 | 0.9670 | +0.2039 | 1.0000 | 1.0000 | +0.0000 |
| 00000077 | 0.6374 | 0.8867 | +0.2493 | 1.0000 | 1.0000 | +0.0000 |
| 00000091 | 0.8009 | 0.9748 | +0.1739 | 1.0000 | 1.0000 | +0.0000 |

## Case study: HMM helps

Trip `00000077` gained +0.2493 point accuracy and +0.0000 route continuity. The transition term suppresses locally plausible but topologically isolated candidates, so the decoded sequence follows connected arcs instead of independent per-point snap decisions.

## Case study: HMM does not help

Trip `00000000` showed the smallest gain (+0.0639 point accuracy, continuity delta +0.0086). No student-test trip regressed in point accuracy; this trip is the weakest positive case, where connectivity constraints had little room to correct errors.
