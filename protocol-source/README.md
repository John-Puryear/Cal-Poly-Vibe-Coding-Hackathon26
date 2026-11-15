# Compact synthetic protocol source

`bdh_nca/protocol_model.py` is the PyTorch source for the browser's 254-parameter recurrent transition. `checkpoint_step12.pt` is the tested checkpoint. `protocol_export.py` produces the browser's portable weights; the committed `dist/protocol/weights.json` is its direct export.

The model has explicit synthetic organizer, axial-chart, coupling, and consolidation rules plus a small learned correction. The normal target is used for offline evaluation only. The browser runtime reads state, weights, and control fields; it does not read a target during inference.

To rerun the Python protocol from this directory after installing `requirements.txt`:

```sh
python protocol_demo.py --checkpoint checkpoint_step12.pt --device cpu --out validation_run --seed 7000 --seeds 1 --face-seeds 1
```

The browser parity and phenotype tests live under the repository's `tests/protocol-*.mjs`. The fixture in `tests/protocol-fixture.f32` was generated from this PyTorch rule, with its shape metadata in `tests/protocol-fixture.json`.
