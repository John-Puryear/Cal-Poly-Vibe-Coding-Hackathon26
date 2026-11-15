# Spec: eliy-design-4 (cleanup pass)

Branch `eliy-design-4`, from `main` at c47a9d7. Files: `dist/index.html`, `dist/style.css`. John's demo logic and protocol files stay untouched. `npm run test:protocol` after every commit.

## Changes

1. **Hero.** Smaller, quieter badge (12px, thin border, muted text). More space between the headline and the pitch. Honesty note in Geist, muted gray. Hero "Details" becomes a small text link under the note, with no full-width line.
2. **"Details +" rows.** The text sits exactly centered between the line above and the line below: 24px above, 24px below. I'll measure every row.
3. **Four panels.** A full-bleed band in the light gray-blue (#E7ECF1) with larger cream cards: 4 across on desktop, 2x2 on tablet, stacked on phones.
4. **Drug framing.** Tighter gaps: intro to subheading 24px, subheading to step cards 24px.
5. **What the demo shows.** "See limits" and "Details +" stacked as one group, with one line above and one below.
6. **How it works.** Visible: headline, the 3 steps, the 3 numbers, the image with its caption. One "Details +" holds everything from the old expanders, in order. Research notes stays separate and collapsed. Wording below.
7. **Customer path.** Headline and both sentences in one centered column above the three cards.
8. **Proposed sequence.** Full-bleed band with "Proposed sequence" centered above the three steps.
9. **Data pipeline.** Shorter paragraph (below). The full original goes in Details. Details aligns left and spans the full content width.
10. **Limits.** No line under the box.
11. **Team.** John: "Lead architect". Max: "Engineer and AI expert". Eliy: "Website designer". These replace the current bios.
12. **Sources.** Six shown. The sixth is Durant et al. 2019, "Early planarian voltage signals" (the timing study behind the demo's voltage-before-head result).

## Wording for review

**How it works, removed from view:**
- Subheading "Separate cues for separate fates."
- Sentence "This 50,741-weight research-model diagnostic compares a normal target with a learned state at update 128."

**How it works, the one merged Details, in this order:**
1. The 6 architecture paragraphs ("The tissue state tracks physical matter..." through "...the research checkpoint's 50,741 weights."), unchanged.
2. "This 50,741-weight research-model diagnostic compares a normal target with a learned state at update 128." (moved here, not deleted)
3. "Distinct anatomy, head, eye, tail, and biochemical views gave the model clearer synthetic targets. It is separate from the live 254-value browser protocol."

**Data pipeline, visible paragraph** (was 1 long sentence):
> We plan to record voltage in living tissue and match it with gene activity measured from the same samples.

**Data pipeline Details:** the original sentence ("We plan to pair embedded bioelectronics with genetically encoded voltage indicators (GEVIs)...") first, then the existing paragraph ("Automating this loop...") with its four links.

## Tasks (one commit each)
1. Details rows and hero.
2. Panels band, drug framing spacing, results group.
3. How it works merge.
4. Customer path column and sequence band.
5. Data pipeline, limits line, team, sources.
6. Checks at 375, 820 and 1280px with the demo run once each. Push, open a PR, don't merge.

## Questions before I build
1. **Details spacing vs. last round's rule.** Centering the text means a section ending in "Details +" ends 24px above the divider, while other sections end 40px above it. Last round you asked for those to match. OK to let them differ, since the Details row now has its own centered space?
2. **Band colors.** The page would have four full-bleed bands: panels, Why it matters (dark), sequence and data pipeline. The sequence band sits right before the data pipeline band, so if both are gray-blue they merge into one block. Option: make the sequence band dark (#13191D), like Why it matters. Then the panels band (gray-blue) sits right before the dark Why band, which is fine because they contrast. Go with dark for the sequence band?
