import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ProtocolRuntime } from '../dist/protocol-runtime.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const readJson = name => JSON.parse(fs.readFileSync(path.join(root, name), 'utf8'));
const seed = readJson('dist/protocol/planaria-seed.json');
const weights = readJson('dist/protocol/weights.json');
const manifest = readJson('tests/protocol-fixture.json');
const binary = fs.readFileSync(path.join(root, 'tests/protocol-fixture.f32'));
const view = new DataView(binary.buffer, binary.byteOffset, binary.byteLength);
const n = seed.width * seed.height;
const stateLength = manifest.tensor_lengths.reduce((a, b) => a + b, 0);
assert.equal(binary.byteLength, manifest.shape[0] * stateLength * 4);
const runtime = new ProtocolRuntime(weights, seed.width, seed.height);

function flatten(state) {
  const tensors = [state.cells, state.edgeFast, state.edgeSlow, state.delayed, state.induction];
  const all = new Float32Array(stateLength);
  let offset = 0;
  for (const tensor of tensors) { all.set(tensor, offset); offset += tensor.length; }
  return all;
}

function compare(state, index, name) {
  const values = flatten(state);
  let max = 0, mean = 0, worst = 0;
  for (let i = 0; i < stateLength; i++) {
    const expected = view.getFloat32((index * stateLength + i) * 4, true);
    const delta = Math.abs(values[i] - expected);
    if (delta > max) { max = delta; worst = i; }
    mean += delta;
  }
  mean /= stateLength;
  console.log(`${name}: max=${max.toExponential(3)} mean=${mean.toExponential(3)} worst=${worst}`);
  assert.ok(max <= 1e-5, `${name} differs from PyTorch at ${worst}: ${max}`);
}

function fire(spec) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = ((i * spec.a + spec.b) % spec.m) < spec.limit ? 1 : 0;
  return out;
}

let state = runtime.initial(new Float32Array(seed.seed));
compare(state, 0, 'initial');
state = runtime.step(state, null, fire(manifest.fire1));
compare(state, 1, 'one');
const keep = new Float32Array(n);
for (let y = 0; y < seed.height; y++) for (let x = 10; x <= 23; x++) keep[y * seed.width + x] = 1;
state = runtime.damage(state, keep);
compare(state, 2, 'cut');
const control = new Float32Array(4 * n);
for (let y = 0; y < seed.height; y++) for (let x = 12; x <= 23; x++) {
  const i = y * seed.width + x;
  control[i] = 0.4;
  control[n + i] = 0.8;
  control[2 * n + i] = 0.25;
  control[3 * n + i] = 0.7;
}
state = runtime.step(state, control, fire(manifest.fire2));
compare(state, 3, 'treated');
console.log('Protocol runtime parity passed.');
