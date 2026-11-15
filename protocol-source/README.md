# Compact synthetic protocol source

`bdh_nca/protocol_model.py` is the PyTorch source for the browser's 254-parameter recurrent transition. The v3 temporal experiment starts from the original `checkpoint_step12.pt`, changes explicit dynamics, and uses a separate `checkpoint_temporal_v3.pt` with two manually calibrated scalar weights: `exposure_log_gain` and `eye_lateral_bias`. No gradient optimizer step is claimed for this candidate. The v1 rule and weights are preserved on `bdh-protocol-v1-preserved` and in a separate archive.

The model has explicit synthetic organizer, axial-chart, coupling, and consolidation rules plus a small learned correction. In v3, a sustained middle-band block creates a wound-conditioned electrical response during exposure; prior voltage gates later organizer memory, and posterior organizer activity inhibits tail commitment so a new head can grow without a visible tail intermediate. These are synthetic assumptions, not measured kinetics or a direct voltage pulse. The normal target is used for offline evaluation only. The browser runtime reads state, weights, and control fields; it does not read a target during inference. The inherited body and axis at the cut are not part of the claimed new electrical prepattern.

To rerun the Python protocol from this directory after installing `requirements.txt`:

```sh
python protocol_demo.py --checkpoint checkpoint_temporal_v3.pt --device cpu --out validation_run --seed 7000 --seeds 1 --face-seeds 1
```

The browser parity and phenotype tests live under the repository's `tests/protocol-*.mjs`. Regenerate the PyTorch parity fixture with `python protocol_fixture_v3.py --checkpoint checkpoint_temporal_v3.pt`; its output is `tests/protocol-fixture.f32.gz` and `tests/protocol-fixture.json`.
