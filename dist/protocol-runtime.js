// Browser implementation of bdh_nca/protocol_model.py for the compact
// synthetic protocol checkpoint.  All arrays are channel-major Float32Array.
// This is a mechanistic simulation, not measured planarian kinetics.

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const sigmoid = v => 1 / (1 + Math.exp(-v));
const gauss = v => Math.exp(-0.5 * v * v);

function channel(array, index, n) {
  return array.subarray(index * n, (index + 1) * n);
}

function pool(array, width, height, radius, average = false) {
  const out = new Float32Array(width * height);
  const denominator = (2 * radius + 1) ** 2;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let result = average ? 0 : -Infinity;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= width) continue;
          const value = array[yy * width + xx];
          result = average ? result + value : Math.max(result, value);
        }
      }
      out[y * width + x] = average ? result / denominator : result;
    }
  }
  return out;
}

function multiply(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < out.length; i++) out[i] = a[i] * b[i];
  return out;
}

function shifted(array, width, height, dy, dx) {
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const yy = y + dy;
    if (yy < 0 || yy >= height) continue;
    for (let x = 0; x < width; x++) {
      const xx = x + dx;
      if (xx >= 0 && xx < width) out[y * width + x] = array[yy * width + xx];
    }
  }
  return out;
}

export function cloneProtocolState(state) {
  return {
    width: state.width,
    height: state.height,
    cells: new Float32Array(state.cells),
    edgeFast: new Float32Array(state.edgeFast),
    edgeSlow: new Float32Array(state.edgeSlow),
    delayed: new Float32Array(state.delayed),
    induction: new Float32Array(state.induction),
  };
}

// Reproducible browser fire masks. This does not reproduce PyTorch's RNG;
// parity tests supply explicit masks to both implementations.
export function makeRng(seed) {
  let value = seed >>> 0;
  return () => {
    value = (value + 0x6D2B79F5) >>> 0;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class ProtocolRuntime {
  constructor(portable, width = 32, height = 32) {
    if (portable?.format !== 'bdh-protocol-portable-v1') {
      throw new Error('Expected a bdh-protocol-portable-v1 weight artifact');
    }
    if (portable.metadata?.config?.channels !== 24) {
      throw new Error('Protocol checkpoint requires 24 cell channels');
    }
    this.width = width;
    this.height = height;
    this.portable = portable;
    this.n = width * height;
    this.radius = portable.metadata.config.long_radius;
    this.fireRate = portable.metadata.config.fire_rate;
    this.weights = portable.weights;
    const scalar = key => this.weights[key].values[0];
    this.exposureRate = 0.025 * clamp(Math.exp(scalar('exposure_log_gain')), 0.4, 2);
    this.memoryThreshold = 0.38 + 0.10 * Math.tanh(scalar('memory_threshold_delta'));
    this.voltageRate = 0.25 * clamp(Math.exp(scalar('voltage_log_gain')), 0.5, 2);
    this.headRate = 0.075 * clamp(Math.exp(scalar('head_log_gain')), 0.5, 2);
    this.eyeRate = 0.085 * clamp(Math.exp(scalar('eye_log_gain')), 0.5, 2);
    this.tissueRate = 0.28 * clamp(Math.exp(scalar('tissue_log_gain')), 0.5, 2);
    this.eyeAxialFraction = 0.70 + 0.06 * Math.tanh(scalar('eye_axial_bias'));
    this.eyeLateralOffset = 1.7 + 0.4 * Math.tanh(scalar('eye_lateral_bias'));
    this.loopGain = scalar('loop_gain');
    this.loopSupportGain = clamp(Math.tanh(Math.max(this.loopGain, 0)) / 0.46211717, 0, 2);
    this.residual0Weight = this.weights['residual.0.weight'].values;
    this.residual0Bias = this.weights['residual.0.bias'].values;
    this.residual2Weight = this.weights['residual.2.weight'].values;
    this.residual2Bias = this.weights['residual.2.bias'].values;
    this.hidden = portable.metadata.config.hidden;
    this.offsets = [[-1, 0], [1, 0], [0, -1], [0, 1], [0, -this.radius], [0, this.radius]];
  }

  initial(cells) {
    const { width: w, height: h, n } = this;
    if (cells.length !== 24 * n) throw new Error(`Expected ${24 * n} initial cell values`);
    const x = new Float32Array(cells);
    const body = channel(x, 0, n);
    let mass = 0, cx = 0, cy = 0;
    for (let i = 0; i < n; i++) {
      const tissue = clamp(body[i], 0, 1);
      mass += tissue;
      cx += tissue * (i % w);
      cy += tissue * Math.floor(i / w);
    }
    mass = Math.max(mass, 1e-4);
    cx /= mass;
    cy /= mass;
    let vx = 0, vy = 0, cov = 0;
    for (let i = 0; i < n; i++) {
      const tissue = clamp(body[i], 0, 1);
      const dx = i % w - cx, dy = Math.floor(i / w) - cy;
      vx += tissue * dx * dx;
      vy += tissue * dy * dy;
      cov += tissue * dx * dy;
    }
    vx /= mass; vy /= mass; cov /= mass;
    const angle = 0.5 * Math.atan2(2 * cov, vx - vy);
    const discr = Math.sqrt(0.25 * (vx - vy) ** 2 + cov ** 2 + 1e-6);
    const code = [
      2 * cx / Math.max(w - 1, 1) - 1,
      2 * cy / Math.max(h - 1, 1) - 1,
      2 * Math.sqrt(Math.max(0.5 * (vx + vy) + discr, 0.01)) / w,
      2 * Math.sqrt(Math.max(0.5 * (vx + vy) - discr, 0.01)) / h,
      angle / Math.PI,
    ];
    for (let i = 0; i < n; i++) {
      const live = body[i] > 0.1 ? 1 : 0;
      for (let j = 0; j < 3; j++) x[(9 + j) * n + i] = code[j] * live;
      for (let j = 0; j < 2; j++) x[(16 + j) * n + i] = code[3 + j] * live;
    }
    return {
      width: w, height: h, cells: x,
      edgeFast: new Float32Array(6 * n),
      edgeSlow: new Float32Array(6 * n),
      delayed: new Float32Array(20 * n),
      induction: new Float32Array(3 * n),
    };
  }

  damage(state, keep) {
    const { width: w, height: h, n } = this;
    if (keep.length !== n) throw new Error(`Expected a ${n}-cell keep mask`);
    const x = new Float32Array(state.cells.length);
    const ef = new Float32Array(state.edgeFast.length);
    const es = new Float32Array(state.edgeSlow.length);
    const delayed = new Float32Array(state.delayed.length);
    const induction = new Float32Array(state.induction.length);
    const oldBody = channel(state.cells, 0, n);
    const lost = new Float32Array(n);
    for (let i = 0; i < n; i++) lost[i] = (oldBody[i] > 0.1 ? 1 : 0) * (1 - keep[i]);
    const aroundLost = pool(lost, w, h, 1);
    const border = new Float32Array(n);
    for (let i = 0; i < n; i++) border[i] = keep[i] * (oldBody[i] > 0.1 ? 1 : 0) * aroundLost[i];
    const aroundBorder = pool(border, w, h, 3);
    for (let c = 0; c < 24; c++) {
      for (let i = 0; i < n; i++) x[c * n + i] = state.cells[c * n + i] * keep[i];
    }
    for (let i = 0; i < n; i++) {
      x[7 * n + i] = Math.max(x[7 * n + i], 0.8 * border[i]);
      x[8 * n + i] = Math.max(x[8 * n + i], border[i]);
      induction[i] = state.induction[i] * keep[i];
      induction[n + i] = state.induction[n + i] * keep[i];
      induction[2 * n + i] = Math.max(state.induction[2 * n + i] * keep[i], aroundBorder[i] * keep[i]);
    }
    for (let e = 0; e < 6; e++) {
      const [dy, dx] = this.offsets[e];
      const neighborKeep = shifted(keep, w, h, dy, dx);
      for (let i = 0; i < n; i++) {
        const valid = keep[i] * neighborKeep[i];
        ef[e * n + i] = state.edgeFast[e * n + i] * valid;
        es[e * n + i] = state.edgeSlow[e * n + i] * valid;
        if (e >= 4) {
          for (let c = 0; c < 10; c++) delayed[((e - 4) * 10 + c) * n + i] = state.delayed[((e - 4) * 10 + c) * n + i] * valid;
        }
      }
    }
    return { width: w, height: h, cells: x, edgeFast: ef, edgeSlow: es, delayed, induction };
  }

  step(state, control = null, fire = null) {
    const { width: w, height: h, n } = this;
    if (control && control.length !== 4 * n) throw new Error(`Expected ${4 * n} control values`);
    if (fire && fire.length !== n) throw new Error(`Expected ${n} fire values`);
    const x = Array.from({ length: 24 }, (_, c) => channel(state.cells, c, n));
    const ef = Array.from({ length: 6 }, (_, c) => channel(state.edgeFast, c, n));
    const es = Array.from({ length: 6 }, (_, c) => channel(state.edgeSlow, c, n));
    const ind = Array.from({ length: 3 }, (_, c) => channel(state.induction, c, n));
    const newCells = new Float32Array(24 * n);
    const dst = Array.from({ length: 24 }, (_, c) => channel(newCells, c, n));
    const newInduction = new Float32Array(3 * n);
    const newEdgeFast = new Float32Array(6 * n);
    const newEdgeSlow = new Float32Array(6 * n);
    const newDelayed = new Float32Array(20 * n);
    const f = new Float32Array(n);
    const block = new Float32Array(n), longBlock = new Float32Array(n);
    const current = new Float32Array(n), purine = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      f[i] = fire ? fire[i] : (Math.random() < this.fireRate ? 1 : 0);
      if (control) {
        current[i] = clamp(control[i], -1, 1);
        block[i] = clamp(control[n + i], 0, 1);
        longBlock[i] = clamp(control[2 * n + i], 0, 1);
        purine[i] = clamp(control[3 * n + i], 0, 1);
      }
    }

    // The stored five-number chart comes only from surviving original tissue.
    const alive = new Float32Array(n), face = new Float32Array(n);
    const polarity = new Float32Array(n), physical = new Float32Array(n);
    const code = new Float32Array(5);
    let liveCount = 0;
    for (let i = 0; i < n; i++) {
      alive[i] = x[0][i] > 0.1 ? 1 : 0;
      liveCount += alive[i];
      face[i] = clamp(x[23][i], 0, 1);
      polarity[i] = x[13][i];
      physical[i] = x[7][i];
      code[0] += x[9][i] * alive[i];
      code[1] += x[10][i] * alive[i];
      code[2] += x[11][i] * alive[i];
      code[3] += x[16][i] * alive[i];
      code[4] += x[17][i] * alive[i];
    }
    for (let k = 0; k < 5; k++) code[k] /= Math.max(liveCount, 1);
    const cx = (code[0] + 1) * Math.max(w - 1, 1) / 2;
    const cy = (code[1] + 1) * Math.max(h - 1, 1) / 2;
    const major = Math.max(code[2] * w, 1);
    const minor = Math.max(code[3] * h, 1);
    const angle = code[4] * Math.PI;
    const cs = Math.cos(angle), sn = Math.sin(angle);
    const axial = new Float32Array(n), lateral = new Float32Array(n), allowed = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const dx = i % w - cx, dy = Math.floor(i / w) - cy;
      axial[i] = dx * cs + dy * sn;
      lateral[i] = -dx * sn + dy * cs;
      allowed[i] = (axial[i] / major) ** 2 + (lateral[i] / minor) ** 2 <= 1 ? 1 : 0;
    }

    const nearby = pool(x[0], w, h, 1);
    const nearBudget = pool(x[8], w, h, 1);
    const newBody = dst[0], newBudget = dst[8];
    const newborn = new Float32Array(n), live = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const growthGate = clamp((nearBudget[i] - 0.04) * 9, 0, 1);
      const growth = this.tissueRate * growthGate * Math.max(nearby[i] - x[0][i], 0) * allowed[i];
      newBody[i] = clamp(x[0][i] + f[i] * (growth + 0.035 * alive[i] * (1 - x[0][i])), 0, 1);
      newborn[i] = x[0][i] <= 0.1 && newBody[i] > 0.1 ? 1 : 0;
      newBudget[i] = clamp(Math.max(x[8][i] * 0.97, 0.85 * nearBudget[i] * newborn[i]), 0, 1);
      live[i] = newBody[i] > 0.1 ? 1 : 0;
    }
    const support = pool(alive, w, h, 1, true);
    const inheritedPolarity = pool(multiply(polarity, alive), w, h, 1, true);
    const inheritedPhysical = pool(multiply(physical, alive), w, h, 1, true);
    const inheritedFace = pool(face, w, h, 1);
    for (let i = 0; i < n; i++) {
      polarity[i] = polarity[i] * (1 - newborn[i]) + inheritedPolarity[i] / Math.max(support[i], 1e-4) * newborn[i];
      physical[i] = physical[i] * (1 - newborn[i]) + inheritedPhysical[i] / Math.max(support[i], 1e-4) * newborn[i];
      face[i] = face[i] * (1 - newborn[i]) + inheritedFace[i] * newborn[i];
      for (let k = 0; k < 3; k++) dst[9 + k][i] = code[k] * live[i];
      dst[16][i] = code[3] * live[i];
      dst[17][i] = code[4] * live[i];
    }

    // Reversible coupling loss exposes a local wound-conditioned electrical
    // tendency. The spatial spread is an explicit synthetic assumption; the
    // block itself still acts on conductance, not as an injected voltage pulse.
    const posterior = new Float32Array(n), headMemory = channel(newInduction, 0, n);
    const eyeMemory = channel(newInduction, 1, n), injury = channel(newInduction, 2, n);
    const eyeWindow = new Float32Array(n), blockedInjury = new Float32Array(n);
    const couplingSignal = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      posterior[i] = sigmoid(12 * (-polarity[i] - 0.12)) * (1 - face[i]);
      blockedInjury[i] = block[i] ** 3 * ind[2][i];
    }
    // A local middle-band block disconnects a wound field. Its electrical
    // influence can reach the newly forming posterior even when the residual
    // injury and posterior fate gradients do not coincide cell for cell.
    const broadBlockedInjury = pool(blockedInjury, w, h, 6);
    for (let i = 0; i < n; i++) {
      couplingSignal[i] = broadBlockedInjury[i] * Math.sqrt(posterior[i]);
      // The previous voltage must enter the induction window before blocked
      // wound tissue can consolidate a persistent posterior organizer.
      const voltageWindow = sigmoid(24 * (x[4][i] + 0.02));
      const exposure = this.exposureRate * couplingSignal[i] * voltageWindow * (1 - ind[0][i]);
      const bistable = 0.025 * ind[0][i] * (1 - ind[0][i]) * (ind[0][i] - this.memoryThreshold);
      headMemory[i] = clamp(ind[0][i] + f[i] * (exposure + bistable), 0, 1) * live[i];
      eyeWindow[i] = Math.exp(-(((x[4][i] + 0.35) / 0.18) ** 2));
      eyeMemory[i] = clamp(ind[1][i] + f[i] * (
        0.012 * purine[i] * eyeWindow[i] * (1 - ind[1][i]) +
        0.012 * ind[1][i] * (1 - ind[1][i]) * (ind[1][i] - 0.30)
      ), 0, 1) * live[i];
      injury[i] = clamp(ind[2][i] * 0.995, 0, 1) * live[i];
    }
    const nearbyHeadMemory = pool(headMemory, w, h, 4);
    const posteriorProgram = new Float32Array(n);
    for (let i = 0; i < n; i++) posteriorProgram[i] = sigmoid(28 * (nearbyHeadMemory[i] - this.memoryThreshold - 0.06)) * Math.sqrt(posterior[i]);

    // Bioelectric update reads prior voltage plus explicit coupling and packets.
    const liveVoltage = multiply(x[4], live);
    const localVNumerator = pool(liveVoltage, w, h, 1, true);
    const localVDenominator = pool(live, w, h, 1, true);
    const voltage = dst[4];
    const rowCenter = new Float32Array(h);
    for (let y = 0; y < h; y++) {
      let rowMass = 0, moment = 0;
      for (let xx = 0; xx < w; xx++) {
        const value = newBody[y * w + xx];
        rowMass += value; moment += value * xx;
      }
      rowCenter[y] = moment / Math.max(rowMass, 1e-4);
    }
    for (let i = 0; i < n; i++) {
      const xrel = i % w - rowCenter[Math.floor(i / w)];
      const faceBilateral = gauss((Math.abs(xrel) - 3.5) / 0.9);
      const broadFace = gauss((physical[i] - 0.065) / 0.095) * gauss(xrel / 7);
      const electricEyePair = faceBilateral * gauss((physical[i] - 0.07) / 0.024);
      const electricMouth = gauss((physical[i] + 0.10) / 0.03) * gauss(xrel / 4);
      const faceVoltage = 0.58 * broadFace - 0.75 * electricEyePair - 0.45 * electricMouth;
      const planarianVoltage = 0.62 * polarity[i] + 0.78 * posteriorProgram[i] + 0.85 * couplingSignal[i] - 0.50 * x[2][i];
      const target = (1 - face[i]) * planarianVoltage + face[i] * faceVoltage + 0.15 * current[i];
      const diffusion = 0.07 * (1 - block[i]) * (localVNumerator[i] / Math.max(localVDenominator[i], 1e-4) - x[4][i]);
      const packetAlive = state.delayed[9 * n + i] + state.delayed[19 * n + i];
      const longVoltage = (state.delayed[8 * n + i] + state.delayed[18 * n + i]) / Math.max(packetAlive, 1);
      const longDiff = 0.025 * (1 - block[i]) * (1 - longBlock[i]) * (packetAlive > 0 ? 1 : 0) * (longVoltage - x[4][i]);
      const slowMean = (es[0][i] + es[1][i] + es[2][i] + es[3][i]) / 4;
      const edgeRecall = 0.02 * Math.tanh(this.loopGain) * slowMean * headMemory[i];
      voltage[i] = clamp(x[4][i] + f[i] * (this.voltageRate * (target - x[4][i]) + diffusion + longDiff + edgeRecall), -1, 1) * live[i];
    }

    // Fate readout uses *previous* voltage, preserving electrical prepattern.
    const previousAbsV = new Float32Array(n);
    const oldVoltageLive = multiply(x[4], live);
    for (let i = 0; i < n; i++) previousAbsV[i] = Math.abs(x[4][i]);
    const faceReady = pool(previousAbsV, w, h, 2, true);
    const organizerV = pool(oldVoltageLive, w, h, 2, true);
    const organizerSupport = pool(live, w, h, 2, true);
    const slowMeanField = new Float32Array(n);
    for (let i = 0; i < n; i++) slowMeanField[i] = (es[0][i] + es[1][i] + es[2][i] + es[3][i]) / 4;
    const nearbySlow = pool(slowMeanField, w, h, 4);
    const headGate = new Float32Array(n), tailGate = new Float32Array(n);
    const loopSupport = new Float32Array(n), ready = new Float32Array(n);
    const anteriorSource = new Float32Array(n);
    let posteriorDrive = 0;
    for (let i = 0; i < n; i++) {
      ready[i] = (1 - face[i]) + face[i] * clamp((faceReady[i] - 0.08) * 9, 0, 1);
      const v = organizerV[i] / Math.max(organizerSupport[i], 1e-4);
      headGate[i] = sigmoid(18 * (v - 0.03));
      tailGate[i] = sigmoid(18 * (-v - 0.03));
      loopSupport[i] = clamp(0.12 + 0.88 * this.loopSupportGain * clamp(nearbySlow[i] / 0.20, 0, 1), 0, 1);
      const leftLive = i % w > 0 ? live[i - 1] : 0;
      const leftBoundary = live[i] * (1 - leftLive);
      anteriorSource[i] = leftBoundary * sigmoid(16 * (polarity[i] - 0.08)) * headGate[i];
      if (axial[i] > 0.45 * major) {
        posteriorDrive = Math.max(posteriorDrive, posteriorProgram[i] * headGate[i] * loopSupport[i] * live[i]);
      }
    }
    const averagedAnterior = pool(anteriorSource, w, h, 2, true);
    const headTarget = new Float32Array(n), eyeTarget = new Float32Array(n), tailTarget = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const posteriorHeadField = Math.exp(-0.5 * ((axial[i] - 0.75 * major) / 2.2) ** 2 - 0.5 * (lateral[i] / 2.2) ** 2);
      const posteriorHead = clamp(1.1 * posteriorDrive * posteriorHeadField, 0, 1);
      const planHead = clamp(clamp(4.8 * averagedAnterior[i], 0, 1) + posteriorHead, 0, 1) * live[i];
      const tailField = Math.exp(-0.5 * ((axial[i] - 0.75 * major) / 1.6) ** 2 - 0.5 * (lateral[i] / 1.9) ** 2);
      const tailFate = (1 - posteriorProgram[i]) * sigmoid(16 * (-polarity[i] - 0.08)) * tailGate[i]
        * (1 - clamp(1.2 * couplingSignal[i], 0, 1))
        * (1 - clamp(1.4 * posteriorDrive, 0, 1));
      const planTail = clamp(1.25 * tailField * tailFate, 0, 1) * live[i];
      const eyeAxial = gauss((Math.abs(axial[i]) - this.eyeAxialFraction * major) / 1.0);
      const eyeLateral = gauss((Math.abs(lateral[i]) - this.eyeLateralOffset) / 0.65);
      const leftProgram = sigmoid(-2 * axial[i]) * sigmoid(16 * (polarity[i] - 0.08));
      const rightProgram = sigmoid(2 * axial[i]) * posteriorDrive;
      const eyeOrganizer = clamp(leftProgram * headGate[i] + rightProgram, 0, 1);
      const planEye = clamp(1.3 * eyeAxial * eyeLateral * eyeOrganizer + 0.7 * eyeMemory[i], 0, 1) * live[i];
      const faceHead = sigmoid(15 * (voltage[i] - 0.09)) * live[i];
      const faceEyeAxial = gauss((physical[i] - 0.07) / 0.023);
      const faceEyeLateral = gauss((Math.abs(lateral[i]) - 0.38 * minor) / 0.95);
      const faceEye = clamp(1.25 * faceEyeAxial * faceEyeLateral * live[i], 0, 1);
      headTarget[i] = (1 - face[i]) * planHead + face[i] * faceHead;
      eyeTarget[i] = (1 - face[i]) * planEye + face[i] * faceEye;
      tailTarget[i] = (1 - face[i]) * planTail;
    }

    // Learned 9→16→5 pointwise correction, deliberately capped at ±0.025.
    const residual = new Float32Array(5 * n);
    const hiddenValues = new Float32Array(this.hidden);
    for (let i = 0; i < n; i++) {
      const input = [newBody[i], voltage[i], polarity[i], headMemory[i], eyeMemory[i],
        x[5][i], x[6][i], x[12][i], x[14][i]];
      for (let j = 0; j < this.hidden; j++) {
        let v = this.residual0Bias[j];
        const offset = j * 9;
        for (let k = 0; k < 9; k++) v += this.residual0Weight[offset + k] * input[k];
        hiddenValues[j] = Math.tanh(v);
      }
      for (let j = 0; j < 5; j++) {
        let v = this.residual2Bias[j];
        const offset = j * this.hidden;
        for (let k = 0; k < this.hidden; k++) v += this.residual2Weight[offset + k] * hiddenValues[k];
        residual[j * n + i] = 0.025 * Math.tanh(v);
      }
    }
    for (let i = 0; i < n; i++) {
      dst[1][i] = clamp(x[1][i] + f[i] * ready[i] * (this.headRate * (headTarget[i] - x[1][i]) + residual[i]), 0, 1) * live[i];
      dst[2][i] = clamp(x[2][i] + f[i] * ready[i] * (this.eyeRate * (eyeTarget[i] - x[2][i]) + residual[n + i]), 0, 1) * live[i];
      dst[3][i] = clamp(x[3][i] + f[i] * 0.075 * ((tailTarget[i] - x[3][i]) + residual[2 * n + i]), 0, 1) * live[i];
      dst[5][i] = clamp(x[5][i] + f[i] * 0.045 * (dst[1][i] - 0.6 * dst[2][i] - x[5][i]), 0, 1) * live[i];
      dst[6][i] = clamp(x[6][i] + f[i] * 0.035 * (0.25 * newBody[i] + 0.65 * dst[2][i] + 0.4 * dst[3][i] - x[6][i]), 0, 1) * live[i];
      dst[12][i] = clamp(x[12][i] + f[i] * ready[i] * (0.05 * (Math.tanh(2 * x[4][i]) - x[12][i]) + residual[3 * n + i]), -1, 1) * live[i];
      dst[14][i] = clamp(x[14][i] + f[i] * ready[i] * (0.025 * (0.75 * newBody[i] - x[14][i]) + residual[4 * n + i]), 0, 1) * live[i];
      dst[13][i] = clamp(polarity[i] + f[i] * face[i] * 0.012 * (Math.tanh(3 * physical[i]) - polarity[i]), -1, 1) * live[i];
      dst[20][i] = clamp(x[20][i] + f[i] * 0.04 * (voltage[i] - x[20][i]), -1, 1) * live[i];
      dst[21][i] = clamp(x[21][i] + f[i] * 0.003 * (dst[20][i] * dst[12][i] - x[21][i]), -1, 1) * live[i];
      dst[7][i] = physical[i];
      dst[15][i] = x[15][i] * live[i];
      dst[18][i] = x[18][i] * live[i];
      dst[19][i] = x[19][i] * live[i];
      dst[22][i] = x[22][i] * live[i];
      dst[23][i] = face[i] * live[i];
    }

    // Reinforced directed links and one-tick delayed axial packets.
    for (let e = 0; e < 6; e++) {
      const [dy, dx] = this.offsets[e];
      const nv = shifted(voltage, w, h, dy, dx);
      const nl = shifted(live, w, h, dy, dx);
      const nh = shifted(headMemory, w, h, dy, dx);
      const neighborOpen = shifted(block, w, h, dy, dx);
      const neighborLongBlock = e >= 4 ? shifted(longBlock, w, h, dy, dx) : null;
      // The zero exterior is important: shift(1-block) is zero outside, not one.
      for (let i = 0; i < n; i++) {
        const yy = Math.floor(i / w) + dy, xx = i % w + dx;
        const hasNeighbor = yy >= 0 && yy < h && xx >= 0 && xx < w ? 1 : 0;
        const localOpen = (1 - block[i]) * (hasNeighbor - neighborOpen[i]);
        const longOpen = e >= 4 ? (1 - longBlock[i]) * (hasNeighbor - neighborLongBlock[i]) : 1;
        const conductance = e >= 4 ? localOpen * longOpen : localOpen;
        const eligible = clamp(1 - 0.5 * Math.abs(nv[i] - voltage[i]), 0, 1) * live[i] * nl[i] * conductance;
        const memoryLoop = 0.30 * Math.tanh(Math.max(this.loopGain, 0)) * headMemory[i] * nh[i] * live[i] * nl[i] * conductance;
        const evalue = clamp(eligible + memoryLoop, 0, 1);
        const fast = clamp(ef[e][i] + f[i] * (0.035 * evalue * Math.max(1 - ef[e][i] - es[e][i], 0) - 0.015 * ef[e][i]), 0, 1);
        const slow = clamp(es[e][i] + f[i] * (0.002 * fast - 0.00035 * es[e][i]), 0, 1);
        const valid = live[i] * nl[i];
        newEdgeFast[e * n + i] = fast * valid;
        newEdgeSlow[e * n + i] = slow * valid;
        if (e >= 4) {
          const src = yy * w + xx;
          const d = e - 4;
          const packet = [voltage, dst[13], dst[12], headMemory, eyeMemory,
            dst[5], dst[6], newBody, voltage, live];
          for (let k = 0; k < 10; k++) {
            newDelayed[(d * 10 + k) * n + i] = hasNeighbor ? packet[k][src] * valid : 0;
          }
        }
      }
    }
    return {
      width: w, height: h, cells: newCells, edgeFast: newEdgeFast,
      edgeSlow: newEdgeSlow, delayed: newDelayed, induction: newInduction,
    };
  }
}
