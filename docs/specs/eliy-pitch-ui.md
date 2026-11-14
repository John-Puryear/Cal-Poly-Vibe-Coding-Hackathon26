# Spec: make Morphogenesis Lab pitch-ready (branch `eliy-pitch-ui`)

Status: APPROVED by Eliy (2026-11-14). Decisions: test blocker option B; untreated panel opens on the electrical view; page order as proposed with the Evidence "03" label untouched; business honesty line approved; section 3 wording edits to come from Eliy, so copy tasks 4 and 5 run last. Build order: 0, 1, 2, 3, 6, 7, 8, 9, then 4, 5, 10.

v2 changes: John is training the model toward two-headed planaria, ectopic eyes and regeneration. So the Evidence and Limits section is frozen, the specimen list and default specimen are frozen (old change 8 dropped), and all new copy is written so it stays true whatever the model ends up showing.

## Goal

A judge who has never heard of a cellular automaton should, in 4 minutes, be able to: watch tissue grow, change a signal, compare against an untreated copy, and understand who would pay for this and why. The technical judge should still find the honest science and the limits section intact.

## What I found while exploring

1. **The untreated copy already exists in the data.** Every frame the worker sends to the page includes both `cells` (treated tissue) and `baseline` (the untreated twin, run with the exact same random schedule but no signal). Today the page only shows it when you tick "View untreated reference" inside "Advanced observations", and then it *replaces* the treated view, so you can never see both at once. A side-by-side panel can be built with **only `app.js`, `index.html` and `style.css`**. No worker or runtime change is needed.
2. **`npm test` fails today, before any change.** It needs a file called `tests/fixture.json` (the "answer key" used to check the browser model matches the Python model). That file is deliberately not committed to git, and it is created by `scripts/prepare-model.py`, which needs the Python library NumPy, which is not installed on this Mac. `npm run test:worker` passes.
3. **The model's size facts can be read live.** When the model loads, the worker already tells the page its parameter count and channel count, and every frame carries the grid size. Today the intro hard-codes "16,116 / 24 / 32 x 32". If John's retrained model changes size, those numbers would silently go stale. `app.js` can fill them in from the loaded model instead.
4. **Em dashes in current site copy:** the page `<title>` and the two dash placeholders shown before the metrics load (both outside the Evidence section). `BDH-NCA` in the intro eyebrow uses an en dash; I'd switch it to a plain hyphen.
5. Outside the Evidence section, the page mentions heads and eyes only to describe what is drawn (the anatomy legend "Body / head / tail" and "Eye identity", the canvas screen-reader label) and what the model is given at the start ("Eyes and head/tail identity are not present initially"). None of these claim a result. I'll keep their meaning as is.

## Frozen (will not touch)

- **Files:** `dist/runtime.js`, `dist/worker.js`, `dist/model/`, `model-source/`, `scripts/`, `tests/`.
- **Evidence and Limits section:** every character of the `<section id="evidence">` block stays exactly as it is, including its "03 / EVIDENCE & LIMITS" label, results list, note and references. No CSS rule it uses (`.article-section`, `.article-content`, `.section-marker`, `.evidence-section`, `.evidence-list`, `.evidence-tag`, `.result-mark`, `.references`, `.source-heading`, `.evidence-note`) gets edited; new sections get their own class names so the Evidence section looks identical.
- **Specimen menu and default specimen:** the `<select id="specimen">` options and the opening specimen stay as they are. `app.js` keeps sending reset with whatever the menu shows.

## Will change

`dist/index.html`, `dist/style.css`, `dist/app.js`, plus this spec. New CSS goes at the end of `style.css`, reusing existing colors and fonts, with its own narrow-screen rules at 1000px, 700px and 370px like the rest of the file.

## Result-neutral copy rule (applies to every task)

- Never say or imply that two heads, extra eyes or regeneration work. Never say they don't work either, outside the Evidence section. The only place that states results is Evidence and Limits, which John owns.
- Guided steps and compare copy ask the judge to *look*, they never predict what they'll see. No "you'll see a second head", no "nothing changes".
- Where existing copy outside Evidence states a result (the Figure 1 caption ends "This checkpoint learned normal fields; altered anatomical outcomes are exploratory."), replace it with a pointer: "What this model can and cannot do yet is listed in Evidence and limits." That keeps one source of truth.
- No new numbers. Model size facts come from the loaded model (finding 3).
- No em dashes, none of the banned words.

## Blocker before task 1: getting `npm test` green

Your rule says `npm test` must pass after every change, but it fails right now for a reason unrelated to the website. Options:

- **A. Max generates it.** He runs `python scripts/prepare-model.py model-source` on his machine. Downside: that script also rewrites files in `dist/model/` (should be identical, but it is his area) and the file still won't be on your Mac.
- **B. I generate only the test file on your Mac.** Install NumPy into a throwaway environment and run a one-off command that writes just `tests/fixture.json` from `model-source/runs/demo/portable/conformance.npz`, without touching `dist/model/`. The file is ignored by git, so nothing gets committed. (Recommended.)
- **C. Skip it.** Run only `npm run test:worker` and note that `npm test` is blocked. Since I'm not touching the model or worker, the risk is low, but it breaks your rule.

Note: when John's new model lands, the test file has to be regenerated anyway, so whoever brings in the new model should run the full prepare script.

## The changes

### 1. Guided demo path (index.html, style.css, small app.js)

Three numbered steps at the top of the experiment box, above the specimen bar:

1. **Grow the tissue.** Press "Grow tissue" and watch the pattern form.
2. **Change the signal.** Click anywhere on the tissue to add a small electrical current.
3. **Compare with untreated.** Look at the untreated twin. Same tissue, same starting point, no signal. What's different?

`app.js` lightly marks each step done (a check mark and a dimmed style) using information the page already has: step 1 when the step counter passes 0, step 2 when an input has been placed, step 3 once the input has run for a few steps. Reset clears the marks. No new data needed. None of the wording depends on what the model shows.

### 2. Untreated twin, side by side (index.html, style.css, app.js)

- Add a third panel **C: "Untreated twin"** next to A (anatomy) and B (electrical). It draws the `baseline` data the worker already sends.
- A small switch inside panel C flips it between **anatomy** and **electrical** view, so the comparison works whether the signal's effect shows up in shape, in voltage, or both. Default: electrical (open question 2). Clicking on panel C does nothing (you can't treat the untreated copy) and no hover ring is drawn there.
- Under the three panels, two plain-language readouts:
  - "Electrical difference from untreated: 0.052" (from `metrics.voltageDifference`, which the worker already computes).
  - "Shape difference from untreated: 0.000" (new, computed in `app.js` from the same `cells` and `baseline` data: average difference across the body, head, eye and tail channels). This is neutral: it reads near zero if the shape doesn't change, and rises on its own if John's model starts changing shape. No worker change.
  - One line under both: "0 means identical to the untreated twin."
- Remove the "View untreated reference" toggle from Advanced. Keep the grid toggle, export button and the voltage note there.
- Layout: three columns on wide screens. Below 1000px, A and B side by side with C full width underneath. Below 700px, a 2+1 grid or stacked; I'll check on a real narrow screen before committing.
- **Possible with only app.js plus the data the worker sends? Yes.**

### 3. Judge-facing copy in three layers (index.html, app.js)

Pattern: plain line anyone gets, then the real term once, then the translation right after. Draft wording (you approve or edit). Applies to the intro, the experiment section and "Inside the model". Not the Evidence section.

| Where | Now | Proposed |
|---|---|---|
| Page title | Morphogenesis Lab (em dash) Signals into structure | Morphogenesis Lab: signals into structure |
| Hero lead | How do local cellular signals become a larger pattern?... | Cells talk to each other with tiny electrical signals, and those signals help decide what a body grows into. We built an AI model that grows tissue on screen so you can change a signal and see what happens. |
| Model facts / architecture | "neural cellular automaton" (not explained) | It's a neural cellular automaton, meaning every cell runs the same small neural network that only sees its neighbors. |
| Bioelectricity | Bioelectricity (label only) | Bioelectricity, meaning the voltage across cell membranes that cells use to coordinate. |
| Run note | "Precursor ready. Run to develop the pattern." | "Starting outline ready. Press Grow tissue." |
| Figure caption | "...writes only to current and conductance controls... This checkpoint learned normal fields; altered anatomical outcomes are exploratory." | "Your click only adds electrical current (and can partly block how easily current flows). It never paints the tissue directly; whatever changes comes from what the model learned. What this model can and cannot do yet is listed in Evidence and limits." |
| Voltage legend | Hyperpolarized / Depolarized | More negative / More positive (with the real terms "hyperpolarized / depolarized" once in the Advanced note) |
| Buttons | Run development / Advance 48 steps | Grow tissue / Skip ahead 48 steps |
| Status | Live model · browser inference | Live model, running in your browser |
| What is supplied (Inside the model) | "a coarse silhouette, weak signaling cues, and an axis are supplied as the precursor. Eyes and head/tail identity are not present initially." | "We give the model a rough outline of the tissue to start from (the 'precursor'), plus faint signals and a front-to-back direction. The head, tail and eye regions are not in that starting outline; anything that appears is grown by the model." |

The last row keeps the same fact in plainer words and makes no claim about what forms. Claims in "Inside the model" stay the same; only wording gets plainer. I'll ask John to glance at that section before the pitch in case his new training changes how the model is set up.

### 4. New section: Who it's for (index.html, style.css)

- Plain line: "Labs that study how animals develop wait days to weeks for every experiment to play out."
- First customers: university labs running development or regeneration experiments on live animals. (This names the customer's field; it says nothing about what our model does.)
- The pain, in their words: two quote cards, each `[NEEDS INTERVIEW QUOTE]` with `[NEEDS INTERVIEW: name, role, lab]` under it.
- Optional line `[NEEDS ELIY: how many labs we talked to]`, shown only if you have the number.

### 5. New section: How it works as a business (index.html, style.css)

- Three-step flow, built with new classes that copy the look of the existing process figure: **Request** (a lab sends the experiment it wants to try) → **Run** (we run it on our models) → **Report** (they get back what changed compared with an untreated twin).
- Comparable: "Drug companies already pay contract research organizations to run experiments for them. We do the same thing in software, before the animal work."
- Pricing: `[NEEDS ELIY: how we charge, e.g. per experiment or per month]`.
- One neutral honesty line: "Reports help a lab choose which experiments are worth running; they don't replace the experiment. See Evidence and limits for what the current model shows."

### 6. New section: The data pipeline (index.html, style.css)

- Big label at the top: **PLANNED DIRECTION, NOT OUR DATA.** "Builds on published lab work [NEEDS JOHN: embedded bioelectronics paper]."
- Plain line: "Today's AI learned from a photo. We want it to learn from a movie."
- Simple HTML/CSS diagram (no images, no new library): **Record** the same tissue over time (electrical signals plus gene activity) → **Line up** each moment in time → **Train** the model on what came before and after → **Predict** what a new signal will do. Under it: "Watching the same tissue change over time lets the model learn cause and effect, not just what a finished animal looks like."
- On narrow screens the four boxes stack vertically.

### 7. New section: Team (index.html, style.css)

- John: biology and computation. `[NEEDS JOHN: one line]`
- Max: engineering. `[NEEDS MAX: one line]`
- Eliy: product and business. `[NEEDS ELIY: one line]`

### 8. Dropped

Default specimen and specimen list: no change, per John's training work.

## Proposed page order

Because the Evidence section keeps its own "03" label, the cleanest order that doesn't touch it is:

Intro → 01 Experiment → 02 Inside the model → 03 Evidence and limits (unchanged) → 04 Who it's for → 05 How it works as a business → 06 The data pipeline → 07 Team.

Story for the pitch: "here's the model, here's what it does and doesn't do yet, here's who needs it and how we get paid." Top nav shows 4 links max so it fits on a phone (Experiment, Limits, Who it's for, Business). The alternative (Who it's for before the experiment) would make the Evidence label read "03" out of order; see open question 3.

## Tasks (one commit each)

Every task ends with: `npm test` and `npm run test:worker` output shown, a search for em dashes in the three files (must be zero), a check that the Evidence section is byte-for-byte unchanged (`git diff` on that block shows nothing), a check that the specimen menu is unchanged, and a browser check at desktop width and about 390px wide.

0. **Unblock `npm test`** per your choice. No commit (the test file is git-ignored).
1. **Remove em dashes and en dash** from current copy (title, metric placeholders, BDH-NCA eyebrow). Commit: "Remove dashes from site copy."
2. **Untreated twin panel** (HTML panel with anatomy/electrical switch, CSS for 3 columns and narrow screens, app.js drawing `baseline`, both difference readouts, remove old Advanced toggle). Commit: "Show the untreated twin next to the treated tissue."
3. **Guided demo steps** (HTML list, CSS, app.js done marks). Commit: "Add a 3-step guided demo to the experiment."
4. **Plain-language copy, intro and experiment** (hero, buttons, legends, caption pointer, run notes in app.js, model facts read from the loaded model). Commit: "Rewrite the intro and experiment in plain language."
5. **Plain-language copy, Inside the model** (same claims, plainer words). Commit: "Rewrite the model explainer in plain language."
6. **Who it's for section.** Commit: "Add a section on who the product is for."
7. **Business section.** Commit: "Add how the business works."
8. **Data pipeline section.** Commit: "Add the planned data pipeline, labeled as future work."
9. **Team section, nav and footer.** Commit: "Add the team section and update navigation."
10. **Final pass:** all checks, full narrow-screen review, a reread of every new line against the result-neutral rule, and a list of every `[NEEDS ...]` placeholder left so the team can fill them. No commit unless fixes are needed.

## Risks

- **Merge clash with John's update.** If John edits the Evidence section while this branch is open, both branches touch `index.html`. Because I never edit inside that section, Git should merge it cleanly, but whoever merges second should check. Merging this branch early lowers the risk.
- **Retrained model changes shape of data.** If John's new model changes channel layout (which channel is voltage, head, eye), the drawing code and my shape-difference number could read the wrong channels. That's true of today's drawing code too. Max would update `worker.js`/`runtime.js` and tell me which channels moved.
- **Three canvases on a phone** get small. Mitigation: 2+1 layout and a real narrow-screen check each task.
- **Claims drift during rewrite.** Mitigation: the result-neutral rule, and the final pass rereads every new line.
- **Placeholders left live.** If `[NEEDS ...]` text is still there at pitch time it looks unfinished. They are loud on purpose; task 10 lists them all.
- **`style.css` is written in a compressed style** (whole sections on one line). Edits in the middle are easy to break and could affect the frozen Evidence look, so new rules go at the end with new class names.
- **Merging doesn't update the private site.** Publishing `dist/` to the Sites project is a separate step you or Max do.

## Requests for Max (none are required for this spec)

1. **Test file:** if you choose option A, run `python scripts/prepare-model.py model-source` and confirm `npm test` passes and `dist/model/` comes out unchanged.
2. **Nice to have:** include the untreated (`baseline`) state in the exported JSON, so an exported experiment carries both sides of the comparison. Today export saves only the treated state.
3. **When John's model lands:** tell Eliy if the channel layout or the `frame` / `ready` messages change, so the third panel and the live model facts keep working.
4. **Found during task 0:** `npm test` rewrites `dist/model/browser-validation.json` on every run (only the timing number changes). On this branch I restore that file after each run and never commit it. Max may want the test to write its report somewhere outside `dist/`.

## For John (heads-up, no action needed now)

- Evidence and Limits, the specimen menu and the default specimen are untouched on this branch.
- The Figure 1 caption now points to Evidence and limits instead of stating results itself, so your section is the single place results live.
- Please glance at "Inside the model" before the pitch in case your new training changes what the model is given at the start.
- `[NEEDS JOHN: embedded bioelectronics paper]` in the data pipeline section.

## Open questions for Eliy

1. **Blocker:** how do we get `npm test` passing: A (Max), B (I generate the git-ignored test file on your Mac, recommended), or C (skip and note it)?
2. **Third panel default view:** electrical (recommended, the signal goes in there) or anatomy? Either way it has a switch.
3. **Page order:** OK with Evidence staying 03 and the business sections after it? Or should "Who it's for" come first, which means the Evidence label "03" will read out of order (since I can't edit it)?
4. **Business honesty line:** OK with "Reports help a lab choose which experiments are worth running; they don't replace the experiment"?
5. **Copy table:** approve, or mark the lines you want changed.
