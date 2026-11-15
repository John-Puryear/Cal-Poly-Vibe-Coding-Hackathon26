# Kaia Biosciences

**Live demo:** https://kaia-biosciences.vercel.app/

Kaia Biosciences builds AI simulations of living tissue so drug teams can test ideas on a computer before running them in the lab. This demo shows the first piece: a model of how cells use electrical signals to decide what to build, and what happens to a regenerating flatworm when a simulated drug blocks those signals.

Built at the Cal Poly AI Hackathon, November 13 to 15, 2026.

## What the demo does

Grow a synthetic planarian, cut a trunk fragment, then apply a simulated treatment that blocks cell-to-cell communication. Watch three synchronized views: anatomy, electrical state, and regulatory/axis/structure fields. A sustained block produces a posterior head that persists after washout and a clean recut. These are synthetic results, not predictions of any real drug's effect.

## How it works

The browser runs a BDH-NCA model: a neural cellular automaton where every cell applies the same small learned rule and only communicates with its neighbors, with memory inspired by the Dragon Hatchling (BDH) architecture. The browser runs a 254-parameter checkpoint in a JavaScript worker; no server, Python process, or GPU is needed to view it.

The separate research checkpoint has 50,741 learned weights. The website uses 254 learned values plus explicit synthetic rules across 1,024 cells on a 32 × 32 grid. The browser does not load or compress the research checkpoint.

## Run locally

With Node.js 18 or newer:

    node scripts/serve.mjs

On Windows, double-click `run-demo.cmd` to start the server and open the experiment. Otherwise, open http://127.0.0.1:4173/ after starting the command above. Run through the local server so the module worker and JSON assets can load.

## Use the demo

- **Normal target + developed:** the target is a visual reference only. Select **Develop normal planarian** to evolve the intact synthetic precursor for 128 ticks.
- **Trunk fragment:** select **Make trunk fragment** to cut the actual developed model state, keeping columns 10 to 18. This cut removes the old posterior tail while preserving only surviving cell and connection memory.
- **Protocol branches:** run untreated control, brief broad block (24 ticks), sustained broad block (96 ticks), or half-dose broad block (96 ticks). Each begins from the same saved fragment and continues until tick 608 after the cut. The block acts on columns 12 to 18. Select **Treated after clean recut** after the sustained run to cut its washed-out tissue again and observe 512 more untreated ticks.
- The three synchronized views show visible body/head/eyes/tail, red/blue synthetic voltage, and a regulatory/axis/structure composite. The third view also allows individual channel inspection. Playback offers 8 ticks/s observation, 40 ticks/s live mode, fast finish, pause, and single-tick stepping. These rates control display only; they do not change the model's tick-based dynamics.

## Verification

    npm run test:protocol

The v3 protocol conformance test compares initialization, one update, a cut, and a controlled update against PyTorch for all five state tensors. In the sustained-treatment timing assay, the posterior voltage shifts during the block before the measured regulatory difference and visible posterior head. The emerging posterior head does not pass through a visible tail stage. Independent seeds also test normal regeneration, brief and half-dose blocks, the sustained block with washout and clean recut, and the face prepattern. The site worker test checks the saved fragment, pause, stepping, and export. See [validation details](dist/protocol/validation.json) for numerical error, event timing, and phenotype counts.

## Limits

These are synthetic results. The v3 browser and Python transitions start from the 254-parameter v1 checkpoint and use a separate candidate checkpoint with two scalar weights calibrated manually. This calibration did not use gradient optimizer steps. The explicit spatial chart, organizer threshold, coupling-loss voltage response, and memory rules are modeling choices. The body, polarity axis, and structural scaffold are already present at the cut. The timing result concerns new treatment-induced posterior signals and anatomy, not a claim that all biological genes or structures begin after voltage. Ticks are qualitative and are not calibrated to biological hours. Neither runtime parity nor the observed phenotypes establish biological predictive accuracy. The reference target is never read by the browser transition. The original architecture and weights are preserved on the `bdh-protocol-v1-preserved` branch and in a separate local archive.

## Product vision and planned data pipeline

The current planarian experiment is a synthetic morphogenesis demo, not a validated drug assay. The long-term product aims to help pharma, biotech, and AI biology teams screen candidate therapies earlier in simulations grounded in and validated against measurements. Planned revenue paths are partner simulation studies, a future privacy-preserving option for proprietary candidates, and licensed data products on morphogenesis and measured intervention responses. No encrypted testing service is implemented in this demo.

The proposed autonomous measurement loop combines embedded bioelectronics with genetically encoded voltage indicators (GEVIs) or voltage-sensitive dyes (VSDs), transcriptomic export from matched samples and time points, and automated alignment for training and held-out validation. This combined pipeline has not been built. See the product and pipeline sections of the live site for primary-study links and limitations.

## Team

- **John Puryear:** biology and computation. Designed and trained the model.
- **Max:** engineering. Got the model running live in the browser.
- **Eliy Miller:** product and business. Designed the site and the go-to-market plan.

## AI tools used

- ChatGPT / Codex - Building and training architecture model
- Claude Code: website layout, styling and copy

## Credits

- Mordvintsev et al., "Growing Neural Cellular Automata," Distill, 2020
- Kosowski et al., "The Dragon Hatchling," 2025 (architecture inspiration)
- Michael Levin lab, bioelectric signaling research

## Source layout

- `dist/app.js`, `dist/protocol.css`, `dist/index.html`: presentation and three-layer rendering.
- `dist/protocol-worker.js`: live protocol scheduling, snapshots, cut, treatment, washout, recut, and export.
- `dist/protocol-runtime.js`: independent JavaScript port of the PyTorch recurrent rule.
- `dist/protocol/`: exported checkpoint weights, synthetic seeds, and validation record.
- `protocol-source/`: compact PyTorch source and checkpoint for this protocol.
- `tests/protocol-*.mjs`: numerical parity, early timing, long outcomes, face timing, and worker integration.
- `dist/runtime.js`, `dist/worker.js`, `dist/model/`, `model-source/`: retained earlier 16,116-parameter branch and its tests. The current page uses the new protocol runtime.
