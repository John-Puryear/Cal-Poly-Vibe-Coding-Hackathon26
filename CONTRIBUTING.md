# Team workflow

## Run the website

Clone this private repository, install Node.js 18 or newer, and run `npm start` from the repository root. Open http://127.0.0.1:4173. There are no frontend dependencies to install. The website runs the exported model in a browser worker; Python and a GPU are unnecessary for viewing it.

## Divide the work

- Website and presentation: `dist/index.html`, `dist/style.css`, `dist/protocol.css`, and `dist/app.js`.
- Active browser protocol: `dist/protocol-runtime.js`, `dist/protocol-worker.js`, and `dist/protocol/`.
- Current Python protocol: `protocol-source/`. The older model remains in `model-source/` for comparison.

Use a short branch for each change, push it, and open a pull request into `main`. Pull the latest `main` before starting. Avoid editing the same file concurrently; agree who owns each change. Do not commit credentials, virtual environments, or unrelated personal files.

## Bring a new model to the website

1. Train, evaluate, and export the compact branch from `protocol-source/`, keeping the checkpoint and transition source together.
2. Update `dist/protocol/weights.json` and the JavaScript runtime for any architectural change.
3. Run `npm run test:protocol` for numerical parity, long protocol outcomes, face timing, and browser worker integration.
4. Check all three layers, live/pause/single-step controls, actual cut, four branch buttons, clean recut, export, and narrow-screen layout in a browser.
5. Commit the checkpoint, exported weights, runtime, and validation record together.

The three panels show one experimental tissue state. The cut removes actual cells and their incident connections; the block changes communication, not anatomy directly. Voltage is a dimensionless proxy. Numerical parity and synthetic phenotype tests do not establish biological predictive validity.

## Publish

Merging a pull request shares source with the team. It does not publish the live site. Publish the reviewed `dist/` through the existing private Sites project as a separate step, keeping viewer access unchanged. Teammates can edit code through GitHub and review the live demo with their invited ChatGPT accounts.
