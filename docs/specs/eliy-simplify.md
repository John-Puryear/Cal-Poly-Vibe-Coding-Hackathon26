# Spec: cut words (branch `eliy-theme`)

Status: APPROVED by Eliy (2026-11-14). Decisions: sub-line is "An early AI model of how cells use electrical signals to shape tissue." (headline unchanged); honesty line stays visible in Business; Q3 table approved, but the step counter keeps the word "Step"; Business headline "Send an experiment. Get a report."; labs slot is a big "[N] labs interviewed" number that hides itself if left empty, leaving the quote cards.

**Goal:** each section reads in about 3 seconds: headline, at most one sentence, one visual. Files: `dist/index.html`, `dist/style.css`, `dist/app.js`. Frozen: Evidence and Limits HTML and text, specimen menu, everything outside my three files.

## Tasks (one commit each)
1. **Hero:** "Test the experiment before the experiment." + "An AI that learns how cells use electricity to build a body." + one blue button ("See it live", jumps to the demo). Strip of 3 big mono numbers, read live: **16,116** learned parameters (from the model), **1,024** cells, each running the same rule (grid width x height), **8 of 8** steps verified against the Python model (read from `model/browser-validation.json`; the stat hides if that file fails to load). Kicker, question box and rule line removed.
2. **Demo, two panels:** "With signal" and "Untreated copy", side by side, both electrical by default. One "Electrical / Anatomy" switch flips both panels together (replaces the two layer toggles and the panel C switch). Visible: Grow tissue, Place at center, Reset. Sliders go into an "Adjust the input" expander. Readout becomes one quiet line: "Difference from untreated copy: electrical 0.018, shape 0.005". Steps shortened: "Grow the tissue." / "Click to add a signal." / "Compare with the untreated copy."
3. **Sections:** one headline under 8 words, one sentence, one visual. Longer text goes into "Technical details".
   - Experiment: "Change one signal. Compare." The figure caption and "Look closely" text go into the demo's Technical details. This keeps "your click only adds current" for the technical judge.
   - Inside the model: keeps headline, the cellular automaton sentence and diagram. Memory paragraph plus both explainers go into Technical details.
   - Who it's for: "Every experiment is a long wait." + "University labs studying development and regeneration wait days to weeks for each result." + quote cards.
   - Business: "Send an experiment. Get a report." (old one is 8 words) + your exact line + diagram. Comparable, pricing and honesty line go into Technical details (see Q2).
   - Data pipeline: "A movie instead of a photo." + label + citation + diagram. Closing paragraph deleted.
4. **Labels:** remove section numbers (nav too), eyebrow, "FIGURE 1", "LOOK CLOSELY", demo caption, "DISPLAY LAYERS", "SYNCHRONIZED", panel type tags, canvas corner tags. Evidence's "03 / EVIDENCE & LIMITS" hidden by CSS only.
5. **Closing line** before Team, large type: "Birth defects. Injuries that don't heal. It starts with understanding the signal."
6. **Wording:** your exact business line; "untreated twin" replaced everywhere, including screen-reader labels and the Reset button label.

Checks after every task: both tests, 0 em dashes, Evidence block and specimen menu unchanged, no sideways scroll at 375, 820 and 1280px, contrast pass.

## Claims check (please read)
- **"Test the experiment before the experiment"** and **"learns how cells use electricity to build a body"** read as things the model does today. It learned made-up normal patterns, not real cells. Safer sub-line: "An AI model of how cells use electricity to build a body."
- Folding the honesty line into Technical details leaves the hero promise with no visible caveat until Evidence.

## Questions
1. Sub-line: yours, or "An AI model of how cells use electricity to build a body"?
2. Honesty line ("Reports help a lab choose which experiments are worth running; they don't replace the experiment."): keep visible as a second line in Business (my recommendation), or hide?
3. Also remove? A/B panel letters (yes), footer tagline (yes), "Research prototype" badge (keep, it's honest), "Skip ahead 48 steps" (keep, it saves pitch time), "Development step" label (keep the number, drop the words), "READ THE FOUNDATIONS" in Evidence (keep, frozen).
4. "[NEEDS ELIY: how many labs we talked to]": turn into a big number in Who it's for ("[N] labs interviewed")?
