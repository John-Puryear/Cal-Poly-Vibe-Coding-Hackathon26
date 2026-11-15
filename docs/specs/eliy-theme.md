# Spec: light "research institute" theme (branch `eliy-theme`, from `eliy-pitch-ui`)

Status: APPROVED by Eliy (2026-11-14).

**Goal:** swap the dark teal look for warm cream and institutional blue, with bigger, calmer type. Files: `dist/style.css` and `dist/app.js` (fonts load from the stylesheet, so `index.html` should not need changes). Same layout, same content, same meaning for every canvas color. Frozen: specimen menu, Evidence and Limits HTML and text, `runtime.js`, `worker.js`, model files.

**Found:** 37 hard-coded colors and 78 text sizes of 6 to 9px in the stylesheet.

**Palette:** cream #FAF7F1 page, #E7ECF1 panels, ink #13191D, muted #5A6570, lines #E3DED3, blue #1F5FD1 (links, primary buttons, active states, section numbers), warm #D9822B (sparingly). Contrast: ink 16.6, muted 5.6 / 5.0, blue 5.4, white on blue 5.8. Warm on cream is 2.7, so never used for text. Lines are faint (1.25) by choice.
**Tissue panels:** dark #13191D. Voltage: negative #6E9EF5, positive #F0A25A, zero #2E3740 (approved). Anatomy colors unchanged. Legend bar matches.
**Placeholders:** ink text, orange dashed outline. **Evidence and Limits:** new colors through shared styles only.
**Fonts:** headlines Inter Tight 300 with tight tracking (about -1px on large sizes); body Inter 400; labels, numbers, step counters JetBrains Mono 400. DM Sans, Source Serif 4 and IBM Plex Mono removed.
**Type and spacing:** body 17px / 1.6; nothing below 12px anywhere (mono labels, legends, slider labels, canvas tags included); hero 64 to 80px on desktop, scaling down on phones; section headings 36 to 44px; about 96px between sections on desktop, 56px on phones; paragraphs about 65 to 75 characters wide.

## Tasks (one commit each)
1. **Colors:** tokens, every hard-coded color mapped to one; cream page, panels, dark tissue panels, Evidence via shared styles, placeholders.
2. **Canvas colors:** voltage, hover ring and grid lines in `app.js`; legend bar.
3. **Buttons and switches:** blue primary with white text and 4px corners; 1px ink outline on secondary; blue active toggles, steps, section numbers.
4. **Fonts:** swap the font import and every font family.
5. **Type scale:** sizes above; 12px floor.
6. **Spacing and line length.**
7. **Final check.**

**Checks after every task:** both tests, em dash count 0, Evidence block and specimen menu unchanged, only my three files changed, browser at 375, 820 and 1280px including that "KAIA BIOSCIENCES" fits at the top on a phone.
**Risk:** bigger text crowds the lab toolbar and canvas labels on phones; checked every task.
