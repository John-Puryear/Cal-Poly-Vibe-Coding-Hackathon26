import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ProtocolRuntime, cloneProtocolState, makeRng } from '../dist/protocol-runtime.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const json = name => JSON.parse(fs.readFileSync(path.join(root, name), 'utf8'));
const seed = json('dist/protocol/planaria-seed.json');
const weights = json('dist/protocol/weights.json');
const runtime = new ProtocolRuntime(weights, seed.width, seed.height);
const n = seed.width * seed.height, w = seed.width, h = seed.height;

function fire(rng) {
  const mask = new Float32Array(n);
  for (let i = 0; i < n; i++) mask[i] = rng() < 0.5 ? 1 : 0;
  return mask;
}
function channel(cells, c) { return cells.subarray(c * n, (c + 1) * n); }
function components(values, threshold, body = null, minArea = 1) {
  const seen = new Uint8Array(n), result = [];
  for (let start = 0; start < n; start++) {
    if (seen[start] || values[start] <= threshold || (body && body[start] <= 0.35)) continue;
    seen[start] = 1;
    const stack = [start];
    let area = 0, sx = 0, sy = 0;
    while (stack.length) {
      const i = stack.pop(), x = i % w, y = Math.floor(i / w);
      area++; sx += x; sy += y;
      for (const j of [x > 0 ? i - 1 : -1, x + 1 < w ? i + 1 : -1,
        y > 0 ? i - w : -1, y + 1 < h ? i + w : -1]) {
        if (j >= 0 && !seen[j] && values[j] > threshold && (!body || body[j] > 0.35)) {
          seen[j] = 1; stack.push(j);
        }
      }
    }
    if (area >= minArea) result.push({ area, x: sx / area, y: sy / area });
  }
  return result;
}
function eyePair(eyes, head) {
  if (eyes.length !== 2 || eyes.some(eye => eye.area < 2)) return false;
  eyes.sort((a, b) => a.y - b.y);
  const [upper, lower] = eyes;
  if (lower.y - upper.y < 1.5 || Math.abs((upper.y + lower.y) / 2 - head.y) > 2.5) return false;
  return eyes.every(eye => Math.hypot(eye.x - head.x, eye.y - head.y) <= 5);
}
function score(state) {
  const body = channel(state.cells, 0), targetBody = channel(new Float32Array(seed.target), 0);
  const heads = components(channel(state.cells, 1), .45, body, 3);
  const eyes = components(channel(state.cells, 2), .35, body, 1);
  const tails = components(channel(state.cells, 3), .35, body, 1);
  const bodyParts = components(body, .35);
  let intersection = 0, union = 0;
  for (let i = 0; i < n; i++) {
    const got = body[i] > .35, wanted = targetBody[i] > .35;
    if (got && wanted) intersection++;
    if (got || wanted) union++;
  }
  const iou = intersection / union;
  const leftHeads = heads.filter(head => head.x < .42 * (w - 1) && head.area >= 10);
  const rightHeads = heads.filter(head => head.x > .58 * (w - 1) && head.area >= 10);
  const leftEyes = eyes.filter(eye => eye.x < .42 * (w - 1));
  const rightEyes = eyes.filter(eye => eye.x > .58 * (w - 1));
  const leftPair = leftHeads.length === 1 && eyePair(leftEyes, leftHeads[0]);
  const rightPair = rightHeads.length === 1 && eyePair(rightEyes, rightHeads[0]);
  const shape = bodyParts.length === 1 && iou >= .8;
  return {
    iou, heads: heads.length, eyes: eyes.length, tails: tails.length,
    normal: shape && heads.length === 1 && leftHeads.length === 1 && eyes.length === 2 && leftPair && tails.length === 1 && tails[0].area >= 2 && tails[0].x > .58 * (w - 1),
    two: shape && heads.length === 2 && leftHeads.length === 1 && rightHeads.length === 1 && eyes.length === 4 && leftPair && rightPair && tails.length === 0,
  };
}

const trunk = new Float32Array(n), middle = new Float32Array(n);
for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
  const i = y * w + x;
  trunk[i] = x >= 10 && x <= 18 ? 1 : 0;
  middle[i] = x >= 12 && x <= 18 ? 1 : 0;
}
const developRng = makeRng(24000);
let developed = runtime.initial(new Float32Array(seed.seed));
for (let tick = 0; tick < 128; tick++) developed = runtime.step(developed, null, fire(developRng));
assert.equal(score(developed).normal, true, 'Developed planarian is not normal');
const cut = runtime.damage(developed, trunk);
const cases = ['sham', 'brief', 'sustained', 'half', 'longOnly'];
const states = Object.fromEntries(cases.map(name => [name, cloneProtocolState(cut)]));
const branchRng = makeRng(41000);
for (let tick = 0; tick < 608; tick++) {
  const mask = fire(branchRng);
  for (const name of cases) {
    const control = new Float32Array(4 * n);
    const dose = name === 'sustained' && tick < 96 ? 1 : name === 'brief' && tick < 24 ? 1 : name === 'half' && tick < 96 ? .5 : 0;
    if (dose) for (let i = 0; i < n; i++) control[n + i] = dose * middle[i];
    if (name === 'longOnly' && tick < 96) for (let i = 0; i < n; i++) control[2 * n + i] = middle[i];
    states[name] = runtime.step(states[name], control, mask);
  }
}
const washout = Object.fromEntries(cases.map(name => [name, score(states[name])]));
for (const name of cases) states[name] = runtime.damage(states[name], trunk);
const recutRng = makeRng(71000);
for (let tick = 0; tick < 512; tick++) {
  const mask = fire(recutRng);
  for (const name of cases) states[name] = runtime.step(states[name], null, mask);
}
const recut = Object.fromEntries(cases.map(name => [name, score(states[name])]));
console.log(JSON.stringify({washout, recut}, null, 2));
for (const phase of [washout, recut]) {
  for (const name of ['sham', 'brief', 'half', 'longOnly']) assert.equal(phase[name].normal, true, `${name} negative failed`);
  assert.equal(phase.sustained.two, true, 'Sustained broad block did not yield strict two-head phenotype');
}
console.log('Protocol long-horizon outcome passed.');
