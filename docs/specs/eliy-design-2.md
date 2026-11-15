# Spec: eliy-design-2 (wider layout, Geist, drug framing, show more)

Branch `eliy-design-2`, from `eliy-design` at 970dbe3. Files: `dist/index.html`, `dist/style.css`. John's demo logic, canvases and protocol files stay untouched. `npm run test:protocol` after every commit.

## Page order and what is visible

| Section | Visible | Collapsed |
|---|---|---|
| Hero | Badge, headline, your new two-sentence pitch, honesty note, button | Current hero Details |
| Four panels | Customers, Simulation studies, Data products, Growth, with main's original text and labels | "How private studies might work" |
| Why it matters | 3 stats with sources | NCATS note |
| Drug framing + demo | Your headline and text, guided steps, the demo with the new label above the condition buttons | John's "How to run this demo" and v3 Details |
| What the demo shows | 3 cards in one row, each with a "Shown in demo" tag, then one "See limits" link | John's full result paragraphs |
| How it works | 3 steps, model numbers | Architecture text, fate diagnostic, Research notes |
| Customer path | Headline, both sentences, 3 cards in one row, 3-step sequence diagram | none |
| Data pipeline | Badge, headline, sentence, 3-step diagram | Precedent links paragraph |
| Limits | Your 4-item list | John's "Biological interpretation" paragraph |
| Team | 3 cards | none |
| Sources | Top 5 (below) | The other sources |

"Where this goes" is removed.

## Layout and type
- Content width 1120px. Text sections use two columns on desktop: headline left (35%), content right. One column below 820px. The hero stays one column so the big headline keeps its width. The demo card is always full width.
- Section spacing 64px on desktop, 40px on phones.
- Geist everywhere outside the demo: headlines 500, body 400 at 17px, stats and model numbers 500. JetBrains Mono only inside the demo. Inter Tight and Inter are removed.
- Cards: cream background, 1px #E3DED3 border, 12px radius, 22px card headlines. No gray-blue fills on the page. The demo stays dark.

## Copy details
- Drug label: added as a line above the condition buttons inside the demo. John's "3 / CONDITION" label and the button names stay.
- Customer cards: the inline links come out of the sentences. Each card gets one "Source ↗" at the bottom. The wording stays, and each card is already two sentences. Card 2 has two sources: I'd link NCATS and leave IQ MPS in the Sources list.
- Top 5 sources: Tufts CSDD cost (2014) and BIO success rates (2016), both behind stats; Oviedo 2010 (the octanol study behind the drug framing); Durant 2017 (heads that persist after recut); Mordvintsev 2020 (the model idea).

## Tasks (one commit each)
1. Reorder: panels back under the hero, split the customer path and data pipeline into their own sections, remove "Where this goes", change what is visible vs collapsed, add the Limits list and the top 5 sources.
2. New copy: hero pitch, drug framing, drug label, results cards, customer cards and sequence diagram.
3. Geist fonts and card style.
4. 1120px width, two-column sections, 64/40px spacing.
5. Checks at 375, 820 and 1280px with the demo run once each, no em dashes, 12px minimum, contrast. Push.

## Questions before I build
1. **"Predicts" vs Limits.** The drug framing says "see whether the model predicts the same outcome", and the Limits list says "Not a prediction of any real drug's effect." A reader could see that as a contradiction. Option: "see whether the model reproduces the same outcome." Keep "predicts", or switch?
2. **John's wording.** The four Limits lines and the "Shown in demo" tag are limitation and result wording, which the ownership rule gives to John. Last time you approved without waiting. Same here?
3. **"Shown in demo" on the sustained-block card.** John's tag for that result was "Tested protocol", because the claim comes from his Python test. The browser demo does show it (I saw the head survive the recut at all three widths), so the tag is accurate. Flagging it because it replaces his label.
