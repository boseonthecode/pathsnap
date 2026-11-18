# PathSnap — EDA data-quality report

Trips inspected: **90** (train=80, test=10).

## Inventory
- Observations: 227,577
- Road-network arcs: 4,061,945
- Median sampling interval: 1.000 s
- Largest GPS gap: 153064.00 s

## Reference-route quality
- Trips with node discontinuities: **11** (00000002, 00000019, 00000033, 00000041, 00000042, 00000069, 00000075, 00000080, 00000086, 00000089, 00000076)
- Trips with repeated route arcs: **10** (00000008, 00000018, 00000069, 00000075, 00000084, 00000086, 00000090, 00000092, 00000095, 00000056)
- Trips with severe GPS-to-route mismatch (coverage < 50%): **2** (00000033, 00000054)

Coverage is the fraction of GPS observations within 50 m of the reference route geometry. Reference routes are labels, not verified ground truth; discontinuous and repeated-arc routes are reported rather than repaired.

## Assumptions
- Distances use a per-trip UTM projection chosen from the trip centroid.
- Reference routes may be non-contiguous; route geometry is evaluated per arc, so jumps are never bridged here.
- Severe-mismatch trips are flagged for low-coverage handling in label alignment and training, not silently treated as clean labels.
