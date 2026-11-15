# Morphogenesis Lab

This static site presents a live, synthetic BDH–NCA regeneration protocol. The browser runs a 254-parameter checkpoint in a JavaScript worker; no server, Python process, or GPU is needed to view it.

## Run locally

With Node.js 18 or newer:

```sh
node scripts/serve.mjs
```

On Windows, double-click `run-demo.cmd` to start the server and open the experiment. Otherwise, open `http://127.0.0.1:4173/` after starting the command above. Run through the local server so the module worker and JSON assets can load.

## Use the demo

1. **Normal target + developed:** the target is a visual reference only. Select **Develop normal planarian** to evolve the intact synthetic precursor for 128 ticks.
2. **Trunk fragment:** select **Make trunk fragment** to cut the actual developed model state, keeping columns 10–23 and preserving only surviving cell and connection memory.
3. **Protocol branches:** run untreated control, brief broad block (24 ticks), sustained broad block (96 ticks), or half-dose broad block (96 ticks). Each begins from the same saved fragment and continues until tick 608 after the cut. The block acts on columns 12–23. Select **Treated after clean recut** after the sustained run to cut its washed-out tissue again and observe 512 more untreated ticks.

The three synchronized views show visible body/head/eyes/tail, red/blue synthetic voltage, and a regulatory/axis/structure composite. The third view also allows individual channel inspection. Playback offers 8 ticks/s observation, 40 ticks/s live mode, fast finish, pause, and single-tick stepping. These rates control display only; they do not change the model's tick-based dynamics.

## Verification

```sh
npm run test:protocol
```

The protocol conformance test compares initialization, one update, a cut, and a controlled update against PyTorch for all five state tensors, with maximum absolute error 3.651×10⁻⁷. A held-out, deterministic 128 + 608 + 512 tick test gives one head/two eyes/one tail after untreated, brief, half-dose, and long-only conditions. Sustained broad block gives two heads/four eyes/no tail after washout and clean recut. The site worker integration test verifies the same saved trunk is used across branches and tests pause, single-step, and export. The separate face test verifies voltage prepattern before visible, regulatory, axis, and structural similarity crossings. See [validation details](dist/protocol/validation.json).

These are synthetic results. The spatial chart, organizer threshold, voltage response, and memory rules include explicit modeling choices. Ticks are qualitative and are not calibrated to biological hours. Neither runtime parity nor the observed phenotypes establish biological predictive accuracy. The reference target is never read by the browser transition.

## Source layout

- `dist/app.js`, `dist/protocol.css`, `dist/index.html`: presentation and three-layer rendering.
- `dist/protocol-worker.js`: live protocol scheduling, snapshots, cut, treatment, washout, recut, and export.
- `dist/protocol-runtime.js`: independent JavaScript port of the PyTorch recurrent rule.
- `dist/protocol/`: exported checkpoint weights, synthetic seeds, and validation record.
- `protocol-source/`: compact PyTorch source and checkpoint for this protocol.
- `tests/protocol-*.mjs`: numerical parity, long outcomes, timing, and worker integration.
- `dist/runtime.js`, `dist/worker.js`, `dist/model/`, `model-source/`: retained earlier 16,116-parameter branch and its tests. The current page uses the new protocol runtime.

## Publishing

The tracked `.openai/hosting.json` retains the existing Sites project ID and static `dist` deployment. A GitHub push updates source control but does not by itself update the published Sites URL. Publish the tested `dist/` through the same Sites project while preserving its current audience.
