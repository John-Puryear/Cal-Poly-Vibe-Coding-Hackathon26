# Morphogenesis Lab

This static site presents a live, synthetic BDH–NCA regeneration protocol. The browser runs a 254-parameter checkpoint in a JavaScript worker; no server, Python process, or GPU is needed to view it.

The separate computer research checkpoint has **50,741 learned weights**. The website uses **254 learned values plus explicit synthetic rules** across **1,024 cells** on a 32 × 32 grid. The browser does not load or compress the research checkpoint.

## Run locally

With Node.js 18 or newer:

```sh
node scripts/serve.mjs
```

On Windows, double-click `run-demo.cmd` to start the server and open the experiment. Otherwise, open `http://127.0.0.1:4173/` after starting the command above. Run through the local server so the module worker and JSON assets can load.

## Use the demo

1. **Normal target + developed:** the target is a visual reference only. Select **Develop normal planarian** to evolve the intact synthetic precursor for 128 ticks.
2. **Trunk fragment:** select **Make trunk fragment** to cut the actual developed model state, keeping columns 10–18. This cut removes the old posterior tail while preserving only surviving cell and connection memory.
3. **Protocol branches:** run untreated control, brief broad block (24 ticks), sustained broad block (96 ticks), or half-dose broad block (96 ticks). Each begins from the same saved fragment and continues until tick 608 after the cut. The block acts on columns 12–18. Select **Treated after clean recut** after the sustained run to cut its washed-out tissue again and observe 512 more untreated ticks.

The three synchronized views show visible body/head/eyes/tail, red/blue synthetic voltage, and a regulatory/axis/structure composite. The third view also allows individual channel inspection. Playback offers 8 ticks/s observation, 40 ticks/s live mode, fast finish, pause, and single-tick stepping. These rates control display only; they do not change the model's tick-based dynamics.

## Verification

```sh
npm run test:protocol
```

The v3 protocol conformance test compares initialization, one update, a cut, and a controlled update against PyTorch for all five state tensors. In the sustained-treatment timing assay, the posterior voltage shifts during the block before the measured regulatory difference and visible posterior head. The emerging posterior head does not pass through a visible tail stage. Independent seeds also test normal regeneration, brief and half-dose blocks, the sustained block with washout and clean recut, and the face prepattern. The site worker test checks the saved fragment, pause, stepping, and export. See [validation details](dist/protocol/validation.json) for numerical error, event timing, and phenotype counts.

These are synthetic results. The v3 browser and Python transitions start from the 254-parameter v1 checkpoint and use a separate candidate checkpoint with two scalar weights calibrated manually. This calibration did not use gradient optimizer steps. The explicit spatial chart, organizer threshold, coupling-loss voltage response, and memory rules are modeling choices. The body, polarity axis, and structural scaffold are already present at the cut. The timing result concerns **new treatment-induced posterior signals and anatomy**, not a claim that all biological genes or structures begin after voltage. Ticks are qualitative and are not calibrated to biological hours. Neither runtime parity nor the observed phenotypes establish biological predictive accuracy. The reference target is never read by the browser transition. The original architecture and weights are preserved on the [`bdh-protocol-v1-preserved` branch](https://github.com/John-Puryear/Cal-Poly-Vibe-Coding-Hackathon26/tree/bdh-protocol-v1-preserved) and in a separate local archive.

## Source layout

- `dist/app.js`, `dist/protocol.css`, `dist/index.html`: presentation and three-layer rendering.
- `dist/protocol-worker.js`: live protocol scheduling, snapshots, cut, treatment, washout, recut, and export.
- `dist/protocol-runtime.js`: independent JavaScript port of the PyTorch recurrent rule.
- `dist/protocol/`: exported checkpoint weights, synthetic seeds, and validation record.
- `protocol-source/`: compact PyTorch source and checkpoint for this protocol.
- `tests/protocol-*.mjs`: numerical parity, early timing, long outcomes, face timing, and worker integration.
- `dist/runtime.js`, `dist/worker.js`, `dist/model/`, `model-source/`: retained earlier 16,116-parameter branch and its tests. The current page uses the new protocol runtime.

## Publishing

The tracked `.openai/hosting.json` retains the existing Sites project ID and static `dist` deployment. A GitHub push updates source control but does not by itself update the published Sites URL. Publish the tested `dist/` through the same Sites project while preserving its current audience.
