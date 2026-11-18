# Decoder failure analysis

Comparison of classifier-only predictions with classifier+HMM decoding. Positive deltas mean HMM improved the metric.

| Trip | Classifier accuracy | HMM accuracy | Δ accuracy | Classifier continuity | HMM continuity | Δ continuity |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00000000 | 0.9205 | 0.9393 | +0.0188 | 0.9820 | 0.9940 | +0.0120 |
| 00000007 | 0.8966 | 0.9121 | +0.0155 | 0.9928 | 1.0000 | +0.0072 |
| 00000009 | 0.8103 | 0.8618 | +0.0515 | 1.0000 | 1.0000 | +0.0000 |
| 00000040 | 0.8644 | 0.8825 | +0.0180 | 1.0000 | 1.0000 | +0.0000 |
| 00000052 | 0.9160 | 0.9962 | +0.0802 | 1.0000 | 1.0000 | +0.0000 |
| 00000056 | 0.8532 | 0.8985 | +0.0453 | 0.9995 | 1.0000 | +0.0005 |
| 00000065 | 0.7708 | 0.7643 | -0.0065 | 1.0000 | 1.0000 | +0.0000 |
| 00000076 | 0.7651 | 0.8250 | +0.0599 | 1.0000 | 1.0000 | +0.0000 |
| 00000077 | 0.6392 | 0.6538 | +0.0146 | 1.0000 | 1.0000 | +0.0000 |
| 00000091 | 0.8043 | 0.8284 | +0.0240 | 1.0000 | 1.0000 | +0.0000 |

## Case study: HMM helps

Trip `00000052` gained +0.0802 point accuracy and +0.0000 route continuity. The transition term suppresses locally plausible but topologically isolated candidates, so the decoded sequence follows connected arcs instead of independent per-point snap decisions.

## Case study: HMM does not help

Trip `00000065` lost 0.0065 point accuracy (continuity delta +0.0000). When the GPS noise or a parallel-road ambiguity makes several candidates nearly equally likely, the transition penalty can prefer a connected but locally wrong arc, overriding the classifier's top choice.
