# Training and held-out experiments

## Data contract

The supplied NPZ files contain aligned physical, electrical, biochemical, regulatory, and structural layers. Their split RNG namespaces are disjoint. Each saved dataset alternates normal planaria and normal face examples. Default procedural pool training also uses only normal samples but can draw a different domain mix. `--data synthetic_data/train.npz` uses the provided training examples directly. Validation/test examples are never sampled into the pool.

Training supervises channels 0-7 and 12-15. Hidden states and edge memories receive no semantic labels. Sparse eye pixels get extra loss weight. This teaches normal morphology/physiology correspondence; it does not isolate a known causal biological mechanism. The electrical map is synthetic and dimensionless, not derived from image colors. Synthetic pattern construction is an explicit inductive bias, not empirical biological evidence.

The fixed checkpoint is a development demonstration. Treat all current test seeds and pulse settings as exploratory after inspecting them. A later claim needs a new frozen test set and preregistered settings.

## Protocol

The default evaluation runs 96 development ticks, 24 intervention ticks, and 128 washout ticks. Paired scenarios use the same precursor and identical stochastic update schedule:

1. Control: no intervention.
2. Posterior pulse: positive voltage current in one posterior disk and partial local conductance block.
3. Ectopic pulse: negative voltage current in one non-eye disk.
4. Damage: erase a tissue disk and incident edge/delay history, preserve undamaged memory, then allow autonomous recovery.

No intervention sets anatomy, slow regulatory values, or desired outcomes. Pulse sign is an invented hypothesis; it is not claimed to match experimentally established millivolt protocols. Opposite sign, timing, position, strength, and duration should be tested systematically. Increasing pulse intensity until a desired image appears does not constitute confirmatory evidence.

Evaluation writes intermediate/end metrics, predicted-state NPZs, comparison sheets, and GIFs. Face development is evaluated in a separate normal example using the same learned rule. The normal target is shown only for comparison; inference does not read it.

## Gates before claiming emergence

- Normal competence: one correctly located head region and two normal eyes under matched control; check tail identity and tissue integrity visually as well.
- Two-headed candidate: exactly two anterior field components after washout, one near the original head and another posteriorly, with normal eye structures retained.
- Ectopic-eye candidate: more than two eye-field components after washout, an additional component away from normal eye territory, and the original structures retained.
- Persistence: rerun for at least 512-1024 autonomous ticks after the pulse. The included candidate flag only screens the default washout horizon.
- Regeneration: quantify repaired occupancy and identities, and compare with matched control. A colored patch alone is insufficient.
- Replication: report all random seeds and normal failures. Candidate counts should be conditioned on normal competence and should include confidence intervals in a larger study.

`metrics.json` uses head threshold 0.45 (minimum 3 pixels), eye threshold 0.35 (minimum 1 pixel), and four-connected components. Normal-location thresholds are stated in `evaluate.py`. These are transparent initial screening choices, not validated anatomical classifiers. Change threshold and component-size assumptions in sensitivity analyses. A noisy extra pixel can count as an eye component; inspect predicted fields and additional structure.

**The demo has not proven actual two-headed anatomy, ectopic eyes, or biological face emergence.** Failing the normal competence gate makes an anomalous-field interpretation uninterpretable. The code reports that failure rather than counting arbitrary blobs as successful phenotypes.

## Ablations and decisions

Train fresh models for each condition using identical seeds, data, width, pool size, rollout budget, and optimizer settings:

```sh
python -m bdh_nca train --data synthetic_data/train.npz --out runs/no_long --no-long-range
python -m bdh_nca train --data synthetic_data/train.npz --out runs/no_plastic --no-plasticity
python -m bdh_nca train --data synthetic_data/train.npz --out runs/no_consolidation --no-consolidation
python -m bdh_nca train --data synthetic_data/train.npz --out runs/no_interaction --no-interaction
```

This is a mechanism-removal ladder within the same implementation, not an exact reproduction of the canonical NCA/BraiNCA baselines. Removing plasticity retains content-gated messages. A published comparison would need independently validated reference baselines.

Useful next benchmark: give an early transient cue, remove it, then require the same local morphology to preserve its earlier branch through a delay and damage. Use **two normal developmental contexts**, not two-headed/ectopic targets. This forces memory to matter more directly than a single stable anatomy target. No such memory-benefit experiment is claimed in the delivered validation.

Keep edge memory if it improves delayed-cue retention, recovery, or stable held-out responses over the simpler variants. Keep long links only if their coordination benefit exceeds their shortcut bias. Scale grids independently from weights and measure runtime; fixed degree alone does not guarantee a useful speedup on every backend.
