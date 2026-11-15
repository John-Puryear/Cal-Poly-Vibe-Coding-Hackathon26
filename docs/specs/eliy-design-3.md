# Spec: eliy-design-3 (one spacing system, a different layout per section)

Branch `eliy-design-3`, from `eliy-design-2` at 2c5e78c. Files: `dist/index.html`, `dist/style.css`. John's demo logic, canvases and protocol files stay untouched. `npm run test:protocol` after every commit.

## Layout per section

| Section | Layout |
|---|---|
| Hero | Unchanged |
| Four panels | Unchanged |
| Why it matters | Full-width dark band (#13191D), cream text, lighter-blue source links. Three stats in a row: "$2.6B", "1 in 10", "8.5 years", each on one line, with your new labels. Stacked on phones |
| Drug framing | Centered, max 760px. Then "Grow. Cut. Change communication." centered as a subheading, 40px gap, then the four steps as numbered cards (4 across, 2x2 on tablet, stacked on phones) |
| Demo | Unchanged |
| What the demo shows | Unchanged layout, spacing fixed |
| How it works | Split: steps and model numbers left, the protocol comparison image with its caption right. Rest of Research notes stays collapsed |
| Customer path | Headline, text, three cards, then "Proposed sequence" as a full-width timeline strip (a line with three numbered dots and labels) |
| Data pipeline | Full-width tinted band (#E7ECF1). Badge, headline and sentence centered above a large horizontal flow diagram |
| Limits | Bordered callout box (1px, 12px radius, cream). Four limits as a two-column checklist. Details stays inside the box |
| Team | Centered heading, three cards centered |
| Sources | Top 5 in two columns, "More sources" below |

## Spacing system
- Sections: 64px above and below every section divider (40px on phones).
- "Details +" rows: 24px from the content above to the Details line, and 24px from the line to the "Details +" text. The same on every section, including under the demo card and under the result cards. Nothing extra below it except the section's normal 64px.
- Inside sections: 24px between related pieces, 40px between groups (for example, subheading to steps).

## Tasks (one commit each)
1. Spacing system across all sections.
2. Why it matters band.
3. Drug framing and step cards.
4. How it works split with the image.
5. Customer timeline and data pipeline band.
6. Limits callout, Team and Sources.
7. Checks at 375, 820 and 1280px with the demo run once each, contrast, 12px minimum, no em dashes. Push.

## Questions before I build
1. **Data pipeline steps.** You asked for a four-step flow, but the pipeline has three steps today: Capture, Profile, Learn. I can (a) keep three, or (b) split the existing "Learn: Align, train, and validate on held-out interventions" into "Learn: align and train" and "Validate: test on held-out interventions". Option (b) uses only words already on the page. Which one?
2. **The comparison image is very tall** (948 x 2048). At half the page width it runs about 1,000px tall, much taller than the steps beside it. I'd show it whole at about 40% width, because cropping a data figure would hide some of the conditions. Option: show only the top part with an "Open full image" link. Whole or cropped?
3. **Spacing check.** For "gap above Details equals gap below", I read "the divider" as the thin line on the Details row itself: 24px above the line, 24px from the line to "Details +". If you meant 24px from "Details +" down to the next section's divider, say so. That would make sections ending in Details shorter at the bottom than the others.
