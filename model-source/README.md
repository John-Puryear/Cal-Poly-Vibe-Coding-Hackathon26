# BDH-NCA: a small layered tissue demo

A runnable, **16,116-parameter** PyTorch research prototype. One shared local rule develops simple 2D planarian and craniofacial fields, with sparse communication and bounded connection memory. This is a synthetic hypothesis-testing system, not a calibrated biological simulator.

The deliverable includes source, layered synthetic data, tests, a short trained checkpoint, evaluation images/GIFs, metrics, and portable weights. Start with `synthetic_data/layer_preview.png` and `runs/demo/training_preview.png`. Read `VALIDATION.md` for what the included checkpoint actually achieved.

## Run

From this folder, on the computer where this package was created:

```sh
./demo.sh inspect
./demo.sh evaluate --checkpoint runs/demo/checkpoint.pt --out runs/my_evaluation
```

The launcher uses the existing isolated project environment if available. For a fresh machine, install **Python 3.12 or 3.13** and create an environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m bdh_nca data --out synthetic_data --count 64 --size 32
python -m bdh_nca train --data synthetic_data/train.npz --out runs/new --iterations 2000 --device auto
python -m bdh_nca evaluate --checkpoint runs/new/checkpoint.pt --out runs/new/evaluation
python -m bdh_nca export --checkpoint runs/new/checkpoint.pt --out runs/new/portable
```

`--device auto` selects CUDA, then Apple MPS, then CPU. CPU was used for the delivered validation. MPS support depends on the installed PyTorch build; use `--device cpu` if needed. CPU can be faster for tiny grids with many small operations. No network is needed after dependencies are installed.

## What the model receives

Each cell has 24 dimensionless state values. Data files store separate arrays:

| Layer | NPZ key | Channels | Meaning |
|---|---|---:|---|
| Physical anatomy | `physical` | 4 | body occupancy, head identity, eyes, tail identity |
| Bioelectric | `bioelectric` | 1 | signed voltage proxy; not millivolts |
| Biochemical | `biochemical` | 3 | activator, inhibitor, stress proxies |
| Genomic equivalent | `genomic_equivalent` | 2 | slow regulatory state and axial polarity |
| Cytoskeletal equivalent | `cytoskeletal_equivalent` | 1 | slow structural state |
| Adaptation | `adaptation` | 1 | feedback attenuation |

The additional model channels hold latent fast/slow state, a temporary activity trace, a consolidated preference, and a two-channel tissue-domain identity. `precursor` is the full initial state; `full_target` is the full normal target. Arrays use `[example, channel, height, width]`. Load with `numpy.load(path, allow_pickle=False)`.

The precursor has a coarse tissue silhouette, weak electrical/chemical cues, and a slow axis. **It has no eyes, head label, or tail label.** This first demo studies pattern formation and repair on a supplied tissue scaffold; it does not demonstrate organism growth from a single cell. The fixed shared weights are the developmental rule set. Slow regulatory state is an expression/memory analogue, not DNA sequence mutation.

## Learning and emergence

Only **normal** examples are generated: one anterior territory, one tail for planaria, two normally placed eyes, and normal craniofacial electrical fields. Geometry, scale, rotation, and cue noise vary. The supplied images are visual motivation, not calibrated voltage measurements or training rasters.

Normal anatomical and physiological fields are supervised. There is no two-headed target, ectopic-eye target, abnormal trajectory teacher, head-count reward, or phenotype-specific inference rule. The learner must acquire the mapping from precursor fields to normal fields. A face-like electrical pattern is therefore learned under normal-field supervision; this is **not** evidence of spontaneous discovery of a biological face from unstructured data.

At evaluation, one held-out pulse drives voltage near the posterior, another drives voltage at a non-eye site. They touch only voltage-current and conductance inputs. All anatomical readouts come from the learned transition rule. A successful abnormal pattern would be out-of-distribution generalization relative to this synthetic training distribution. Its biological interpretation would still depend on the invented normal-data assumptions.

Training uses short differentiable rollouts and a pool of continuing states paired with their original normal targets. Damage training is **off by default**. `--damage` explicitly changes the experiment to repair training and is recorded in checkpoints. Persistent pool training itself encourages stability; regeneration is assessed separately.

If normal patterns drift during long evaluation, extend the persistence curriculum before interpreting interventions. `train --init-checkpoint runs/new/checkpoint.pt --steps 48 --pool-chunks 16 --iterations 500 --data synthetic_data/train.npz --out runs/persistent` warm-starts the weights with a fresh optimizer and pool. It is fine-tuning, not exact optimizer-state resumption. The parent checkpoint hash is recorded. Longer rollouts increase training cost and memory without adding parameters.

## Model size and controls

| Hidden width | Trainable parameters | Purpose |
|---:|---:|---|
| 32 | 5,812 | smallest preset |
| 96 | 16,116 | delivered default |
| 256 | 41,876 | larger local experiment |

Use `train --hidden 32` or `--hidden 256` to change size. Grid size and rollout length change runtime state and compute, not parameter count. A 32x32 grid has 24 cell-state floats, 12 edge-memory floats, and 20 delayed-packet floats per cell: approximately 224 KiB per example before activations, gradients, and framework overhead.

The update costs O(HW) for a fixed channel width and rollout step; training through T steps costs O(THW) plus the associated activations. Dense operations still run on sparse-valued activations; no sparse-execution speedup is claimed.

Run independent seeds and keep the data and training budget fixed when comparing `--no-long-range`, `--no-plasticity`, `--no-consolidation`, and `--no-interaction`. These switches retain identical trainable parameter counts, though disabled branches still occupy computation. Edge-memory state is additional runtime memory, not a trainable parameter.

## Files

- `ARCHITECTURE.md`: state, equations, constraints, and design rationale.
- `EXPERIMENTS.md`: normal competence gates, held-out protocols, limitations, and ablations.
- `BROWSER_PORT.md`: portable export and recurrent parity contract; no website is included.
- `SOURCES.md`: supplied source interpretation and primary references.
- `synthetic_data/`: 64 training, 8 validation, and 8 test layered normal examples.
- `runs/demo/`: trained checkpoint and logs; evaluation and export subfolders.

Long-range communication, connection memory, and consolidation are research hypotheses here. Retain their complexity only if controlled comparisons show a benefit.
