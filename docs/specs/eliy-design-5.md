# Spec: eliy-design-5 (skimmable: takeaway first, one line, one Details)

Base: `main` at cee707c (PR #7 merged). Files: `dist/index.html`, `dist/style.css`. Demo logic and protocol files untouched. `npm run test:protocol` after every commit. Preview to `kaia-preview` after the push.

## New wording

| Where | Now | New |
|---|---|---|
| Hero subheading | We're building simulations that let drug teams test ideas on a computer before running them in the lab. Today's demo shows one: ... | We build simulations so drug teams can test ideas before the lab. |
| Drug framing text | In real flatworms, a chemical that blocks ... | In real flatworms, **octanol**, a chemical that **blocks cell-to-cell signals**, made cut pieces grow **a head at both ends**. We apply a simulated version and test whether the model reproduces it. |
| Under "Grow. Cut. Change communication." | Every view reads the same evolving tissue state. | (removed) |
| Results heading | What the demo shows | What the model did |
| How heading | How the model works | How it works |
| Step 1 | Develop / One head and one tail | Develop / One head, one tail |
| Step 2 | Cut & block / Timed middle-band coupling loss | Cut and block / Timed signal block |
| Step 3 | Wash out & recut / Test persistent organizer memory | Wash out and recut / Does it persist? |
| Number labels | research checkpoint weights / browser demo learned values / cells running the same browser rule | Research model weights / Browser model values / Cells, one shared rule |
| Testable question text | The first prospective users are ... (2 sentences) | First users are biotech and pharma research teams, starting with one narrow study we can check against lab results. |
| Card 1 | AI is already in the lab. (2 sentences) | AI is already in the lab. Benchling's 2026 survey of about 100 biotech teams already using AI describes demand for scientific tools, alongside data and validation gaps. |
| Card 2 | Validation defines the use. (2 sentences) | Validation defines the use. NCATS and IQ MPS work shows pharma, researchers and tool builders developing fit-for-purpose lab models together. |
| Card 3 | Qualification takes evidence. (2 sentences) | Qualification takes evidence. The FDA accepted a different AI liver model's letter of intent into ISTAND, the first of three qualification steps. |
| Pipeline headline | Measure voltage. Read gene programs. | Measure voltage and gene activity. |
| Pipeline paragraph | We plan to record voltage ... | (removed; the original long sentence stays first in Details) |
| Limits heading | What it cannot claim | What it can't do yet |
| Team | Name plus role | John, Max, Eliy (names only) on the dark background |

The cut card sentences ("Its sample represents AI adopters...", "They are category precedents...", "...not approval of that tool or this one.") and the original two-sentence intro go into the section's one Details, word for word. Each card keeps its "Source ↗" link.

## One Details per section
Two sections have two today:
- **Demo:** "How to run this demo" and John's v3 "Details". Merged into one "Details": the 4 run steps first, then the v3 text, unchanged.
- **How it works:** "Details" and "Research notes". Merged into one "Details": the current text first, then the Research notes content (heading, text and both images).

## Layout
- **Section dividers** run edge to edge. Lines around Details rows stay content-width.
- **Full-bleed:** the panels and result cards span the full window with a 24px gap at the window edge, so cards don't touch the glass. The Capture to Validate band is full-bleed too.
- **Rhythm, top to bottom:** Hero (contained), Panels (full-bleed), Why it matters (dark band), Demo (contained), Results (full-bleed cards), How it works (contained), Testable question (contained, ends with the sequence strip), Data pipeline (tinted band), Limits (contained box), Team (dark band), Sources (contained).

**Two full-bleed neighbors:**
1. **Panels then Why it matters.** Decided: panels on plain cream (no tint) so the card row reads light and the dark band below clearly breaks from it.
2. **Sequence strip then Data pipeline band.** Already separated last round: the strip is cream between hairlines, the pipeline band is tinted. I'd keep that.

## Header blue (decided: A)
The blue text color (#1F5FD1, used for "CUSTOMERS", "SIMULATION STUDIES" in the panels) on the brand mark and "KAIA", plus a thin blue underline on the nav link for the section you're on.

Footer brand becomes identical to the header brand (the footer mark isn't blue today).

## Tasks (one commit each)
1. Wording changes and the Details merges.
2. Header and footer brand.
3. Full-bleed layout, edge-to-edge dividers, Team band.
4. Checks at 375, 820, 1280 with the demo run once each. Push, preview link, PR.

## Decisions
- Base: updated `main`.
- Header blue: A.
- Panels: plain cream.
- Proposed sequence strip stays visible; only the extra text goes into Details.
