# Morphogenesis Lab

Shared website and Python model source for the Cal Poly Vibe Coding Hackathon 2026. See [CONTRIBUTING.md](CONTRIBUTING.md) for the team workflow.

A dark, article-style site with live inference from the supplied BDH-NCA weights. Anatomy and bioelectricity are synchronized views of the same tissue state. Each layer can be hidden independently.

## Start locally

Requires Node.js 18+; no frontend package installation or build step:

```sh
npm start
```

Open http://127.0.0.1:4173. Open the site through a server, not a file URL, so module workers and model files can load.

## Scientific scope

The provided checkpoint learns normal synthetic fields on a supplied scaffold. The site does not insert target anatomy into inference. Manual input modifies only electrical current. The visible body, head, eye, and tail fields are read directly from the model. Blue/amber maps negative/positive normalized voltage proxies; these are not measured millivolts.

Persistent two-headed planaria, ectopic eyes, and anatomical regeneration were not established by the delivered checkpoint. They remain explicitly identified as open experiments.

## Model and changes

- `dist/runtime.js`: independently implemented recurrent transition, using original float32 weights.
- `dist/worker.js`: model loading, paired random-update schedules, interventions and session state.
- `dist/app.js`: presentation, rendering and UI.
- `dist/model/`: original portable manifest/weights, synthetic test precursors, provenance and verification report.

The supplied Python package is included under `model-source/`, with its delivered checkpoint, portable export, and synthetic datasets. The original Downloads folder is unchanged. Generated evaluation galleries and the separate curriculum run are omitted.

To load an updated export, install NumPy and run `python scripts/prepare-model.py model-source`, then `npm test`. This updates the browser model, precursor states, and local test fixture. If the architecture changes, update the JavaScript runtime before claiming parity.

`npm test` requires the generated local `tests/fixture.json`. That fixture is not committed because it is derived from the original package's conformance file. Recreate it with the preparation script. Check `dist/model/browser-validation.json` for the last checked model.

Any change to the transition rule or exported configuration needs a corresponding runtime update and conformance check. The current port supports the delivered 24-channel, 8-message, rank-4 configuration. Training remains in Python.

## Hosting

`dist/` is a static website. No Python server, API key, or GPU service is required for this browser port. The Sites manifest is `.openai/hosting.json`. Use the Sites workflow for this deployment or host `dist/` on a static host. The browser worker uses a single shared model with separate experimental and untreated state buffers.

## Verification

The browser-runtime conformance test checks all four recurrent state tensors across eight exported steps against the supplied PyTorch fixture, with required maximum absolute error ≤ 1e-5. Longer specimen rollouts and manual input integration should also be checked when updating the model. Passing parity validates the port, not biological fidelity or abnormal phenotypes.

For this version, the eight-step maximum error is 1.7881393432617188e-7. Face and planarian 144-step reference runs (96 development / 24 pulse / 24 washout) each have a maximum error of 2.980232238769531e-7. Worker integration checks also pass: model loading, paired controls, input placement without direct anatomy writes, input expiry, weight-identified export, deterministic reset, and specimen selection.

These checks verify runtime parity and worker behavior in Node. The project owner has confirmed the hosted site works. A full signed-in browser review of controls and responsive layout remains part of the team handoff.

Private preview: https://morphogenesis-lab-bdh-nca.pleasantgerbil1.chatgpt.site

## Publishing changes

GitHub stores the shared source. Pushing here does not automatically update the existing private Sites deployment. After reviewing a change, publish the `dist/` directory through the existing Sites project. Preserve its private audience. The `.openai/hosting.json` file identifies that project and contains no credentials.
