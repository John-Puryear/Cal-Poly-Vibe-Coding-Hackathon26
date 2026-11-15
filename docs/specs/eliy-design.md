# Spec: eliy-design (light theme, clearer page order)

Branch: `eliy-design`, from `main` at 3c5a090. Files I change: `dist/index.html`, `dist/style.css` (and `dist/app.js` only if needed, which I don't expect). I don't touch `dist/protocol*`, `dist/model`, `dist/media`, `protocol-source`, `model-source` or `tests`. Any restyle of the demo happens through new rules in `style.css`.

Baseline: `npm run test:protocol` passes on the branch before any change.

## What the page will look like (top to bottom)

| # | Section | Visible | In "Details +" |
|---|---|---|---|
| 1 | Hero | Badge "Product vision · not a drug assay", headline "Test ideas earlier. Learn from biology.", the one-sentence pitch, the honesty note, "See it live" button | Current hero Details text |
| 2 | Why it matters | Headline, the 3 stats three across, each with its source link | NCATS 15-year note (unchanged) |
| 3 | Live demo | John's heading "Grow. Cut. Change communication.", then 4 short guided steps (below), then the full demo card, unchanged | John's "How to run this demo" and his v3 Details, both whole and unchanged |
| 4 | What the demo shows | Headline plus 3 short result lines, each ending "See limits" | John's two result paragraphs and his timing note with the validation link, verbatim |
| 5 | How it works | Headline, John's 3-step diagram (Develop, Cut & block, Wash out & recut), model numbers three across (50,741 / 254 / 1,024) | Architecture text; the model-numbers note; the fate diagnostic (whole block); a nested collapsed "Research notes" holding the development gallery (whole block) |
| 6 | Where this goes | Headline plus 3 short items: request and report product, pharma and biotech customers, planned data pipeline | The 4 venture cards, the private-studies note, the customer evidence cards and sequence, the pipeline text and diagram (all existing wording) |
| 7 | Limits | Headline plus one line | John's "Biological interpretation" paragraph, verbatim |
| 8 | Team | Headline plus 3 cards | none |
| 9 | Sources | Headline only, list collapsed | All source groups, plus John's "Read the foundations" list moved here |

Removed: the "Watch signals become structure" intro (its button is replaced by the hero button and its numbers move to How it works, keeping `id="stat-parameters"` so the live count still updates), the separate "Questions to test" section (the questions go into the guided steps), and the closing line "A signal changes. A pattern follows." (it isn't in the new order). The nav becomes: Demo, Results, How it works, Limits.

## New copy (mine to write, yours to approve)

Headlines (all under 8 words):
- Why it matters: "Why it matters"
- What the demo shows: "What the demo shows"
- How it works: "How the model works"
- Where this goes: "Where this goes"
- Limits: "What it cannot claim" (taken from John's existing headline)
- Team: "Who's building it"
- Sources: "Sources"

Guided demo steps (above the controls):
1. **Grow.** Click Develop normal, then Cut trunk.
2. **Pick a condition.** Each one restarts from the same trunk fragment.
3. **Watch the timing.** "Does posterior voltage shift while the sustained block is active, before a new head appears?" (John's question, verbatim)
4. **Recut.** "After washout and a clean recut, does the treated tissue keep its altered head pattern?" (John's question, verbatim)

Where this goes, three items:
- **Request and report.** Partners request a simulation study and get a written report. *(Planned)*
- **Pharma and biotech teams.** Drug developers first, then AI companies moving into biology.
- **A planned data pipeline.** Measure voltage and gene activity in the lab to train better models. *(Planned)*

Team: John (Biology and computation), Max (Engineering), Eliy (Product and business). Each one-line bio stays a visible TODO until you send the real line.

## Theme and type

- Light page: background #FAF7F1, panels #E7ECF1, text #13191D, muted text #5A6570, lines #E3DED3, links and buttons #1F5FD1. Warm #D9822B only for non-text highlights (the "planned" badge border, status dots), because it is too light to pass contrast as text.
- Demo card: dark #13191D with its own darker panels, and a lighter blue for text inside it so it passes contrast. Canvas pixels, canvas backgrounds and legend chip colors stay exactly as they are.
- Fonts: Inter Tight 300 for every headline (Georgia goes away), Inter 17px for body, JetBrains Mono only for numbers, demo controls and readouts. Decorative mono labels outside the demo switch to Inter or go away (venture card labels, source group labels). Caveat text like "not a drug assay" and "planned" stays.
- Nothing below 12px: the demo's 9 to 11px text gets raised to 12px through overrides in `style.css`.

## Layout

- Text columns about 680px. The demo card uses the full content width (1200px max).
- 96px between sections on desktop, 56px on phones.
- Stats and model numbers: three across on desktop and tablet, stacked on phones (three numbers won't fit side by side at 375px without shrinking below readable).
- One "Details +" style everywhere (thin top line, label left, + or − right), including the demo guide, Research notes and Sources.

## Tasks (one commit each, tests after each)

1. Reorder the page into the 9 sections, move John's blocks whole, add the new headlines, steps and "Where this goes" items, restore the Team section, collapse Sources and Research notes.
2. Light theme colors and the dark demo card.
3. Fonts and the 12px minimum.
4. Spacing, column widths, three-across grids, one Details style.
5. Check: `npm run test:protocol`, no em dashes, contrast spot check, screenshots at 375, 820 and 1280px with the demo run once at each (Develop, Cut, Sustained broad, Clean recut). Then push `eliy-design`.

## Needs John's approval

1. The three short result lines in "What the demo shows" (draft, his full text stays in Details):
   - Normal control: a trunk fragment regrows one head and one tail, untreated or after a brief block.
   - Sustained broad block: a second head forms, and it survives washout and a clean recut.
   - Timing: posterior voltage shifts during the block, before the new head appears.
2. Splitting his "What the demo shows / What it cannot claim" section into two, and using his sentence "These outputs do not validate a drug response or predict real planarian regeneration." as the one visible Limits line.
3. Hiding his decorative labels inside the demo card is not in this plan. "INTERACTIVE PROTOCOL", "PLANARIAN · 32 × 32" and "PLANARIAN PROTOCOL" stay as written, restyled only. Ask him if he wants them gone.
4. Moving his "Read the foundations" list into Sources.
5. Restoring the Team section: his commit 625ccb6 removed it. Ask if that was on purpose.

## Questions for you before I build

1. The hero has two visible sentences (pitch plus honesty note). I'd keep the honesty note visible as a small exception to rule 5. OK?
2. Team bios: send me one line each, or leave them as TODOs?
3. Is dropping the closing line OK?
