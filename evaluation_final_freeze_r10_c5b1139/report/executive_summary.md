# Smart-MDT executive evaluation summary

Input: `rust_results_final_freeze_r10_1fbfa30` (7360 rows, 46 datasets).

| Objective | Best method | Result |
| --- | --- | ---: |
| Accuracy | CALS-MDT | 0.864820 |
| Smallest tree | CompactExplain | 11.124 nodes |
| Fastest runtime | Unary | 0.047 s |
| Shortest AXp | CompactExplain | 2.721 |

## Key statistical conclusions


- For CALS-MDT vs CompactExplain on Accuracy, the right-hand method was lower by 0.0047 (95% bootstrap CI [-0.0073, -0.0022], raw p=0.001873, Holm-adjusted p=0.01124, paired Cohen's d=-0.523, medium effect).

- For SmartCertified vs CALS-MDT on Accuracy, the right-hand method was higher by 0.0052 (95% bootstrap CI [0.0014, 0.0092], raw p=0.005784, Holm-adjusted p=0.02892, paired Cohen's d=0.385, small effect).

- For CALS-MDT vs CompactExplain on Fit time (seconds), the right-hand method was lower by 6.0334 (95% bootstrap CI [-7.8811, -4.3309], raw p=2.842e-14, Holm-adjusted p=4.263e-13, paired Cohen's d=-0.965, large effect).

- For SmartCertified vs CALS-MDT on Fit time (seconds), the right-hand method was higher by 8.1640 (95% bootstrap CI [5.6094, 11.0623], raw p=5.446e-10, Holm-adjusted p=7.624e-09, paired Cohen's d=0.845, large effect).

- For SmartCertified vs CALS-MDT on Tree nodes, the right-hand method was lower by 37.9587 (95% bootstrap CI [-45.8178, -30.6804], raw p=3.522e-09, Holm-adjusted p=4.576e-08, paired Cohen's d=-1.411, large effect).

- For SmartCertified vs CompactExplain on Tree nodes, the right-hand method was lower by 39.5652 (95% bootstrap CI [-49.0461, -30.9891], raw p=3.52e-09, Holm-adjusted p=4.576e-08, paired Cohen's d=-1.254, large effect).


## Certification

The audit found 7360 theorem-certified rows, 0 empirical rows, 0 forbidden predicates, 0 feature-label leakage findings, 0 path violations, and 0 empirical fallbacks, 0 AXp evidence violations, and 0 full-row AXp violations.

Repeated runs and depths are averaged within each dataset. Statistical tests
are paired over independent dataset blocks, and confidence intervals use
10000 deterministic dataset-block bootstrap resamples with seed
20260718. Reported effect sizes are the paired Cohen's d, consistent with the
paired test; the full significance table also lists the unpaired Cliff's delta as
a distributional reference, which should not be read as a paired estimator.
