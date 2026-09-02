# Statistical contract

## Image-level losses

The implementation uses independent maximum-IoU many-to-one matching. Multiple
selected predictions may map to one ground truth, which preserves duplicate
multiplicity. DIAL-Out losses are normalized duplicate occupancy and coverable
instance shortfall. DIAL-Prop uses a separate proposal-miss loss.

## Complete-rule family

Every calibration-visible detector checkpoint, adapter/evidence checkpoint,
preprocessing rule, candidate cap/order, graph, active relation set, objective,
solver tolerance, budget, ranking, fusion, fallback, matcher, and risk
definition belongs to one complete rule. A change creates a new rule and changes
the multiplicity term.

## Calibration

`dial_det.method.calibration` implements the one-sided empirical-Bernstein
expression used in the article:

```text
mean + sqrt(2 * unbiased_variance * log(|D||R|/alpha) / n)
     + 7 * log(|D||R|/alpha) / (3*(n-1))
```

`scripts/calibrate_complete_rules.py` requires actual calibration sample IDs in
each loss NPZ and checks them against the split manifest. Utility inputs must
use exactly the utility IDs. Rule IDs and SHA-256 values must agree with the
complete-rule files.

The bound concerns expected bounded risks for fixed rules under its assumptions.
It is not an AP guarantee, a per-image safety statement, or a certificate under
an unregistered distribution shift.
