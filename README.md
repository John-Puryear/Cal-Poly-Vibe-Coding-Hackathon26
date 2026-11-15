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

## Product vision and planned data pipeline

The current planarian experiment is a synthetic morphogenesis demo, not a validated drug assay. The long-term product aims to help pharma, biotech, and AI biology teams screen candidate therapies earlier in simulations grounded and validated against measurements. Planned revenue paths are partner simulation studies, a future privacy-preserving option for proprietary candidates, and licensed data products on morphogenesis and measured intervention responses. No encrypted testing service is implemented in this demo.

The proposed autonomous measurement loop combines embedded bioelectronics with genetically encoded voltage indicators (GEVIs) or voltage-sensitive dyes (VSDs), transcriptomic export from matched samples and time points, and automated alignment for training and held-out validation. This combined pipeline has not been built. See the product and pipeline sections in `dist/index.html` for primary-study links and limitations.

## Source layout

- `dist/app.js`, `dist/protocol.css`, `dist/index.html`: presentation and three-layer rendering.
- `dist/protocol-worker.js`: live protocol scheduling, snapshots, cut, treatment, washout, recut, and export.
- `dist/protocol-runtime.js`: independent JavaScript port of the PyTorch recurrent rule.
- `dist/protocol/`: exported checkpoint weights, synthetic seeds, and validation record.
- `protocol-source/`: compact PyTorch source and checkpoint for this protocol.
- `tests/protocol-*.mjs`: numerical parity, early timing, long outcomes, face timing, and worker integration.
- `dist/runtime.js`, `dist/worker.js`, `dist/model/`, `model-source/`: retained earlier 16,116-parameter branch and its tests. The current page uses the new protocol runtime.

## Update the original public Site

The [current v3 demo](https://bdh-nca-bioelectric-timing-lab.elvenbee20.chatgpt.site/#experiment) is public so the original Site owner can inspect it. This `site-v7-customer-path` branch contains the exact published site files, including the collapsed operating guide, customer-path evidence, and sources. The original Site is [Morphogenesis Lab](https://morphogenesis-lab-bdh-nca.pleasantgerbil1.chatgpt.site/), with project ID `appgprj_6af8b1de0968819184b672671b51dd15`. The `.openai/hosting.json` in this repository already points to that original project and declares `dist/` as the static build. GitHub changes do not automatically publish to ChatGPT Sites.

**For the person who owns the original ChatGPT Site:**

1. Sign in to the ChatGPT account that owns the original Site. Open the public v3 demo above to see the intended result.
2. Start a Codex chat in that account and give it this branch URL: `https://github.com/John-Puryear/Cal-Poly-Vibe-Coding-Hackathon26/tree/site-v7-customer-path`. Paste the instruction below. No ZIP transfer is needed because the complete deployable `dist/` and hosting manifest are on this branch.
3. After Codex deploys, refresh the **original** Site URL and compare it with the public v3 demo. The original URL should show the planarian seed, three live layers, compact treatment controls, and the 50,741 / 254 / 1,024 model comparison.

> Update my **existing** ChatGPT Site `appgprj_6af8b1de0968819184b672671b51dd15` at `https://morphogenesis-lab-bdh-nca.pleasantgerbil1.chatgpt.site/` from the `site-v7-customer-path` branch of `John-Puryear/Cal-Poly-Vibe-Coding-Hackathon26`. Use the repository's `.openai/hosting.json` and the complete `dist/` directory. Preserve the existing project ID, URL, and public audience. Save and deploy a new version of this Site; do not create a replacement Site. Then verify development, trunk cut, brief block, sustained broad block, half-dose block, clean recut, the three synchronized layers, the operating guide, and the sources section.

If Codex cannot obtain a source write credential for the original Site, check that it is signed into the Site owner's ChatGPT account. An owner may also grant **Can edit** to an active member of the same ChatGPT workspace through the Site's **Share → Who has access** settings. GitHub repository ownership alone does not grant ChatGPT Sites publishing rights.

The published v3 demo and this GitHub repository are usable while the original Site owner performs that final deployment.
