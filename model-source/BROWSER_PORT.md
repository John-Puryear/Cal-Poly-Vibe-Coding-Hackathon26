# Future browser port

No website is built in this deliverable. The model is deliberately restricted to operations that can be implemented with WebGPU shaders, WebGL textures, or a tensor library. Export is independent of Python pickle:

```sh
python -m bdh_nca export --checkpoint runs/demo/checkpoint.pt --out runs/demo/portable
```

`manifest.json` specifies channel order, offsets, configuration, tensor shapes, and binary byte offsets. `weights.f32` contains little-endian float32 values in PyTorch tensor order. Convolution weights are `[out,in,1,1]`; treat them as shared per-cell affine matrices. `weights.json` contains the same arrays for prototyping.

The recurrent state is a tuple of cells `[B,24,H,W]`, fast/slow edges `[B,6,H,W]` each, and delayed packets `[B,2,10,H,W]`. The delayed-packet order is eight projected values, source voltage, and source liveness. Edge order is north, south, west, east, west-six, east-six. Zero-pad every shift, never wrap. Use separate read and write buffers: in-place spatial updates change the dynamics.

Implement `model.forward` in the same order. A browser may store channels packed into RGBA textures or use an HWC layout internally, but must transpose weights/state consistently. All learned operations are 1x1 affine maps. Fixed spatial operations are shifts, neighborhood maximum, arithmetic, sums, and clamps. There is no FFT, complex arithmetic, all-pairs graph, autograd, or dynamic topology at inference.

For deterministic verification, supply a binary update mask rather than relying on a browser RNG. `conformance.npz` contains the initial four state tensors, eight masks, eight controls, and all four expected state tensors after each tick. Convert those arrays to typed arrays when writing the browser test harness. Compare **all state channels and edges**, not only the visible output, with max absolute error <= 1e-5 in float32. Keep full precision initially; validate float16 separately before optimizing.

The exporter reloads the binary into a fresh PyTorch model and verifies eight recurrent steps exactly. It also checks all eight ticks against an independently written NumPy-only runtime in `bdh_nca/numpy_runtime.py`, which can be run directly without importing PyTorch:

```sh
python bdh_nca/numpy_runtime.py runs/demo/portable
```

This validates serialization and a second implementation of the transition. **It does not validate a JavaScript/WebGPU implementation**, which remains future work. Test different grid sizes, border cells, damage, delayed-message clearing, and all ablation switches in the actual browser runtime.

Display and damage handling should stay separate from the transition rule. The demo's renderer maps learned anatomy fields to colors and voltage to a diverging palette; it does not affect state. Interactive erasure must clear deleted cells and incident edge/delay history, not just visible pixels.
