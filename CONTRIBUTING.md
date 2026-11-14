# Team workflow

## Run the website

Clone this private repository, install Node.js 18 or newer, and run `npm start` from the repository root. Open http://127.0.0.1:4173. There are no frontend dependencies to install. The website runs the exported model in a browser worker; Python and a GPU are unnecessary for viewing it.

## Divide the work

- Website and presentation: `dist/index.html`, `dist/style.css`, and `dist/app.js`.
- Browser model integration: `dist/runtime.js` and `dist/worker.js`.
- Training and Python architecture: `model-source/bdh_nca/`. See its README and VALIDATION documents before making claims about results.

Use a short branch for each change, push it, and open a pull request into `main`. Pull the latest `main` before starting. Avoid editing the same file concurrently; agree who owns each change. Do not commit credentials, virtual environments, or unrelated personal files.

## Bring a new model to the website

1. Train/evaluate/export in `model-source/` using that package's instructions. Keep model configuration and the checkpoint together.
2. From the repository root, run `python -m pip install numpy`, then `python scripts/prepare-model.py model-source`.
3. Run `npm test` and `npm run test:worker`. For longer rollout parity, run `python scripts/prepare-long-check.py model-source`, then `npm run test:long`.
4. Check anatomy and voltage views, run/pause/step/reset, manual input expiry, specimen changes, export, and narrow-screen layout in a browser.
5. Commit the updated model files and verification reports together. If state channels, update equations, or communication configuration change, the browser runtime must be changed and reverified too.

The two panels show the same experimental tissue state. Manual interventions apply electrical current; they must not paint head/eye/body outputs. Voltage is a dimensionless proxy. Parity with the supplied implementation does not establish biological validity, regeneration, two-headed planaria, or ectopic eyes.

## Publish

Merging a pull request shares source with the team. It does not publish the live site. Publish the reviewed `dist/` through the existing private Sites project as a separate step, keeping viewer access unchanged. Teammates can edit code through GitHub and review the live demo with their invited ChatGPT accounts.
