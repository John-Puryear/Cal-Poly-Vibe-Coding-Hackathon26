import assert from 'node:assert/strict';
import fs from 'node:fs';
import { ProtocolRuntime, cloneProtocolState, makeRng } from '../dist/protocol-runtime.js';

const seed = JSON.parse(fs.readFileSync(new URL('../dist/protocol/planaria-seed.json', import.meta.url)));
const weights = JSON.parse(fs.readFileSync(new URL('../dist/protocol/weights.json', import.meta.url)));
const { width: w, height: h } = seed, n = w * h;
const runtime = new ProtocolRuntime(weights, w, h);
const keep = new Float32Array(n), middle = new Float32Array(n);
for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
  const i = y * w + x;
  keep[i] = x >= 10 && x <= 18 ? 1 : 0;
  middle[i] = x >= 12 && x <= 18 ? 1 : 0;
}
const fire = rng => {
  const mask = new Float32Array(n);
  for (let i = 0; i < n; i++) mask[i] = rng() < .5 ? 1 : 0;
  return mask;
};
const region = state => {
  let count = 0, voltage = 0, regulatory = 0, memory = 0, head = 0, tail = 0;
  for (let i = 0; i < n; i++) {
    const x = i % w;
    if (x < 16 || x > 25 || state.cells[i] <= .35) continue;
    count++;
    voltage += state.cells[4 * n + i];
    regulatory += state.cells[12 * n + i];
    memory += state.induction[i];
    head = Math.max(head, state.cells[n + i]);
    tail = Math.max(tail, state.cells[3 * n + i]);
  }
  return { voltage: voltage / count, regulatory: regulatory / count,
    memory: memory / count, head, tail };
};
let developed = runtime.initial(new Float32Array(seed.seed));
const developRng = makeRng(24000);
for (let tick = 0; tick < 128; tick++) developed = runtime.step(developed, null, fire(developRng));
const fragment = runtime.damage(developed, keep);
let sham = cloneProtocolState(fragment), treated = cloneProtocolState(fragment);
const initial = region(fragment);
assert.ok(initial.tail < .05, 'The trunk cut must remove the old posterior tail');

const branchRng = makeRng(41000), first = { voltage: null, regulatory: null, memory: null, head: null };
let largestTail = initial.tail, at24 = null, at96 = null;
for (let tick = 1; tick <= 96; tick++) {
  const mask = fire(branchRng), control = new Float32Array(4 * n);
  for (let i = 0; i < n; i++) control[n + i] = middle[i];
  sham = runtime.step(sham, null, mask);
  treated = runtime.step(treated, control, mask);
  const a = region(treated), b = region(sham);
  largestTail = Math.max(largestTail, a.tail);
  if (first.voltage === null && a.voltage > .05 && a.voltage - b.voltage > .10) first.voltage = tick;
  if (first.regulatory === null && a.regulatory - b.regulatory > .05) first.regulatory = tick;
  if (first.memory === null && a.memory > .05) first.memory = tick;
  if (first.head === null && a.head > .15) first.head = tick;
  if (tick === 24) at24 = { treated: a, sham: b };
  if (tick === 96) at96 = { treated: a, sham: b };
}
assert.ok(first.voltage !== null && first.voltage <= 8, 'Electrical response must emerge early during treatment');
assert.ok(first.voltage < first.regulatory, 'Regional regulatory change must follow electrical change');
assert.ok(first.voltage < first.memory, 'Persistent head memory must follow the electrical response');
assert.ok(first.regulatory < first.head && first.head <= 96, 'Visible posterior head must form after regulatory change during treatment');
assert.ok(largestTail < .35, 'Treated posterior must not first grow a visible tail');
assert.ok(at24.treated.voltage > .15 && at24.treated.head < .05, 'Voltage must lead visible posterior anatomy');
assert.ok(at96.treated.head > .35 && at96.treated.tail < .05, 'Posterior head must develop without a tail');
console.log(JSON.stringify({ first, at24, at96, largestTail }, null, 2));
console.log('Protocol timing and direct posterior-head growth passed.');
