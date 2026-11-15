import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ProtocolRuntime, makeRng } from '../dist/protocol-runtime.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = name => JSON.parse(fs.readFileSync(path.join(root, name), 'utf8'));
const seed = read('dist/protocol/face-seed.json');
const runtime = new ProtocolRuntime(read('dist/protocol/weights.json'), seed.width, seed.height);
const n = seed.width * seed.height;
const rng = makeRng(29000 + seed.source_seed);
const ticks = new Set([0, 1, 2, 4, 8, 16, 32, 64, 128]);
const fields = { voltage: 4, visibleHead: 1, visibleEyes: 2, regulatory: 12, genomicAxis: 13, structure: 14 };
const first = Object.fromEntries(Object.keys(fields).map(name => [name, null]));
let state = runtime.initial(new Float32Array(seed.seed));
for (let tick = 0; tick <= 128; tick++) {
  if (ticks.has(tick)) {
    for (const [name, c] of Object.entries(fields)) {
      let numerator = 0, denominator = 0;
      for (let i = 0; i < n; i++) {
        if (seed.target[i] <= 0.35) continue;
        const wanted = seed.target[c * n + i];
        const observed = state.cells[c * n + i];
        numerator += (observed - wanted) ** 2;
        denominator += wanted ** 2;
      }
      const similarity = 1 - numerator / Math.max(denominator, 1e-9);
      if (first[name] === null && similarity >= 0.5) first[name] = tick;
    }
  }
  if (tick === 128) break;
  const fire = new Float32Array(n);
  for (let i = 0; i < n; i++) fire[i] = rng() < 0.5 ? 1 : 0;
  state = runtime.step(state, null, fire);
}
console.log(JSON.stringify(first));
for (const name of ['visibleHead', 'visibleEyes', 'regulatory', 'genomicAxis', 'structure']) {
  assert.ok(first[name] !== null && first.voltage < first[name], `${name} did not follow voltage`);
}
console.log('Protocol face electrical prepattern passed.');
