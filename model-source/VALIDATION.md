# Validation of the delivered demo

The final model learns normal 2D planarian identity fields and a normal face-like voltage pattern. **Neither persistent two-headed planaria nor ectopic eyes was demonstrated in the tested electrical interventions.** The damage test also did not establish anatomical regeneration: an erased tissue hole remained visible.

## Training and environment

CPU training used 2,500 normal-pattern iterations followed by 500 persistence fine-tuning iterations, with no damage training. The combined recorded training time was 717.2 seconds (about 12.0 minutes) on this machine. Timing is illustrative and excludes setup and evaluation.

Tested dependencies: Python 3.12.14, PyTorch 2.15.0, NumPy 2.5.4, Pillow 13.0.0. See `requirements-tested.txt`.

The 96-tick loss on one fixed validation planarian decreased from 0.042275 at initialization to 0.000810 at the end of the normal curriculum. The final fine-tuned value was 0.000805. These are a supervised, weighted field loss on one validation specimen, not a benchmark-wide accuracy estimate.

Checkpoints record seed, hyperparameters, damage-training status, and the fine-tuning parent hash. The initial curriculum metadata also records removal of two unused intervention-input columns; normal training outputs were unchanged. Final inference routes intervention inputs only through current and conductance.

## Held-out results

All three planarian controls in each protocol retained one head domain and two eye domains at the end of washout and passed the normal-location competence gate. The same three specimen seeds were used in both protocols; they are not six independent replications. All results are exploratory.

| Protocol | Development / pulse / washout ticks | Competent controls | Two-head candidates after posterior pulse | Ectopic-eye candidates after ectopic pulse |
|---|---|---:|---:|---:|
| evaluation | 96 / 24 / 128 | 3/3 | 0/3 | 0/3 |
| extended_evaluation | 96 / 96 / 512 | 3/3 | 0/3 | 0/3 |

The normal face test produced two eye fields and a face-like electrical map under both horizons. This was learned from supervised normal fields; it is not spontaneous face discovery. Inspect `runs/demo/evaluation/face.png` against its normal target.

In the extended damage test, seed 0 had three thresholded eye components, but it did not pass the full ectopic-eye screen and was not counted as an ectopic-eye candidate. This illustrates why raw component counts alone are inadequate. The damaged specimens retained a tissue gap; no successful regeneration claim is made.

## Verification

- Ten tests passed: parameter budget; bounded long-run dynamics; no boundary wrapping; delayed nonlocal transport; causal voltage effects of edge/preference memory; memory ablations; local damage-history clearing; normal-only data construction; fate-gradient recovery; and independent recurrent runtime parity.
- All 80 saved normal examples passed their construction/component audit. The test suite separately audited 128 procedural samples.
- Binary weight round-trip error: 0.0. Independent NumPy/PyTorch maximum error across eight exported recurrent ticks: 2.38418579e-07. The weight binary is 64544 bytes.
- The independent runtime test also covers multiple model configurations, non-square grids, external currents/conductance, boundary cells, and all ablation switches.
- The comparison sheets and face preview were inspected visually. No actual browser runtime was built or tested.

## Interpretation and next experiment

This establishes a compact executable baseline, separate synthetic data layers, bounded multi-timescale memory, stable normal patterning in a few held-out specimens, and a portable recurrent inference contract. It does not establish a benefit from BDH-inspired memory relative to simpler controls; no trained ablation study was performed.

The next scientific step is to teach richer normal developmental trajectories and isolated physiological perturbations, still excluding abnormal anatomical targets. Those data should help distinguish causal electrical/regulatory coupling from a model that simply preserves normal anatomy. Then freeze new held-out pulse protocols and test whether a persistent alternative attractor appears. A negative result is informative and must not be converted into a successful phenotype by adding a special-case rendering or fate rule.
