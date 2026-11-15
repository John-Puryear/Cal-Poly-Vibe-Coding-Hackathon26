"""Held-out executable demo for the synthetic protocol branch.

The trained model sees physical perturbations and its own recurrent state.
Normal targets appear only in evaluation metrics and comparison images.
"""
import argparse
import json
from pathlib import Path

import torch

from bdh_nca.data import make_example
from bdh_nca.protocol_assays import score_paired_protocol, score_prepattern
from bdh_nca.protocol_model import ProtocolNCA, ProtocolState
from bdh_nca.render import sheet, fate_diagnostic


def repeat_state(state, count):
    return ProtocolState(*(v.repeat(count, *([1] * (v.ndim - 1))) for v in state.tensors()))


def load_model(checkpoint, device):
    model = ProtocolNCA().to(device)
    info = None
    if checkpoint:
        info = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(info["model"])
    model.eval()
    return model, info


@torch.no_grad()
def planaria_case(model, seed, device, out=None):
    torch.manual_seed(17000 + seed)
    ex = make_example(32, seed, "planaria", "test")
    state = model.initial(ex.seed.to(device))
    for _ in range(128):
        state = model(state)
    developed = state.cells[:1].cpu()
    xx = torch.arange(32, device=device)[None, None, None, :]
    trunk = ((xx >= 10) & (xx <= 23)).expand(1, 1, 32, 32)
    middle = ((xx >= 12) & (xx <= 23)).expand(1, 1, 32, 32)
    fragment = model.damage(state, trunk)
    state = repeat_state(fragment, 6)
    intact = model.initial(ex.seed.to(device))
    for _ in range(128):
        intact = model(intact)
    for dst, src in zip(state.tensors(), intact.tensors()):
        dst[4:5] = src
    initial_fragment = state.cells[:1].cpu()
    exposure_voltage = None
    for tick in range(608):
        control = state.cells.new_zeros(6, 4, 32, 32)
        if tick < 24:
            control[1, 1:2] = middle[0]
        if tick < 96:
            control[2, 1:2] = middle[0]
            control[3, 2:3] = middle[0]
            control[4, 1:2] = middle[0]
            control[5, 1:2] = .5 * middle[0]
        fire = (torch.rand(1, 1, 32, 32, device=device) < .5).to(state.cells.dtype).expand(6, -1, -1, -1)
        state = model(state, control, fire)
        if tick == 95:
            exposure_voltage = state.cells[:, 4:5].cpu().clone()
    washout = state.cells.cpu().clone()
    washout_memory = state.induction.cpu().clone() if state.induction is not None else None
    state = model.damage(state, trunk.expand(6, -1, -1, -1))
    for _ in range(512):
        fire = (torch.rand(1, 1, 32, 32, device=device) < .5).to(state.cells.dtype).expand(6, -1, -1, -1)
        state = model(state, fire=fire)
    recut = state.cells.cpu().clone()
    recut_memory = state.induction.cpu().clone() if state.induction is not None else None
    body = ex.target[:, :1]
    screen = score_paired_protocol(washout[:1], washout[2:3], recut[:1], recut[2:3], body,
                                   {"brief": (washout[1:2], recut[1:2]),
                                    "long_only": (washout[3:4], recut[3:4]),
                                    "intact_broad": (washout[4:5], recut[4:5]),
                                    "half_dose": (washout[5:6], recut[5:6])})[0]
    posterior = (body[0, 0] > .35) & (torch.arange(32)[None] >= 20) & (torch.arange(32)[None] <= 23)
    memory_summary = None
    if washout_memory is not None:
        memory_summary = {
            "washout_posterior_head": [float(washout_memory[j, 0][posterior].mean()) for j in range(6)],
            "recut_posterior_head": [float(recut_memory[j, 0][posterior].mean()) for j in range(6)],
        }
    if out is not None:
        rows = [("Normal target", ex.target), ("Developed", developed),
                ("Trunk fragment", initial_fragment), ("Sham after washout", washout[:1]),
                ("Brief block", washout[1:2]), ("Sustained broad block", washout[2:3]),
                ("Long-only block", washout[3:4]), ("Intact, broad block", washout[4:5]),
                ("Half-dose broad block", washout[5:6]),
                ("Sham after clean recut", recut[:1]),
                ("Treated after clean recut", recut[2:3])]
        sheet(rows, out / f"planaria_{seed}.png", title="Protocol branch | held-out planarian anatomy and fields")
        fate_diagnostic(rows, out / f"planaria_fates_{seed}.png")
    return {"seed": seed, "screen": screen, "posterior_memory": memory_summary,
            "exposure_voltage_right": None if exposure_voltage is None else
            [float(exposure_voltage[j, 0, :, 20:24].mean()) for j in range(6)]}


@torch.no_grad()
def face_case(model, seed, device, out=None):
    torch.manual_seed(29000 + seed)
    ex = make_example(32, seed, "face", "test")
    start = ex.seed.to(device).clone()
    start[:, 7:8] = .35 * start[:, 13:14]
    start[:, 4:5] = 0
    start[:, 12:15] = 0
    start[:, 1:4] = 0
    baseline = model.initial(start)
    clamped = model.initial(start.clone())
    ticks = (0, 1, 2, 4, 8, 16, 32, 64, 128)
    timeline = {}
    clamp_timeline = {}
    for tick in range(max(ticks) + 1):
        if tick in ticks:
            timeline[tick] = baseline.cells.cpu().clone()
            clamp_timeline[tick] = clamped.cells.cpu().clone()
        if tick == max(ticks):
            break
        fire = (torch.rand(1, 1, 32, 32, device=device) < .5).to(start.dtype)
        baseline = model(baseline, fire=fire)
        clamped = model(clamped, fire=fire)
        if tick < 24:
            cells = clamped.cells.clone()
            cells[:, 4:5] = 0
            clamped = ProtocolState(cells, clamped.edge_fast, clamped.edge_slow,
                                    clamped.delayed, clamped.induction)
    screen = score_prepattern(timeline, ex.target)[0]
    clamp_screen = score_prepattern(clamp_timeline, ex.target)[0]
    if out is not None:
        rows = [("Normal face target", ex.target), ("Physical cue only", timeline[0]),
                ("Tick 4", timeline[4]), ("Tick 8", timeline[8]),
                ("Tick 32", timeline[32]),
                ("Tick 64", timeline[64]), ("Tick 128", timeline[128])]
        sheet(rows, out / f"face_{seed}.png", title="Protocol branch | bioelectric prepattern timing")
    return {"seed": seed, "screen": screen,
            "early_voltage_clamp_first_crossing": clamp_screen["first_crossing"],
            "early_voltage_clamp_scores": clamp_screen["scores"]}


def run(args):
    torch.set_num_threads(3)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(args.seed)
    model, ck = load_model(args.checkpoint, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for i in range(args.seeds):
        row = planaria_case(model, args.seed + i, device, out if i < args.visual_count else None)
        cases.append(row)
        print(json.dumps({"planaria_seed": row["seed"],
                          "retained": row["screen"]["retained_after_recut"],
                          "sham": row["screen"]["matched_sham_pass"],
                          "negative_controls": row["screen"]["negative_controls_pass"]}), flush=True)
    faces = []
    for i in range(args.face_seeds):
        row = face_case(model, args.seed + 1000 + i, device, out if i < args.visual_count else None)
        faces.append(row)
        print(json.dumps({"face_seed": row["seed"],
                          "prepattern": row["screen"]["full_face_prepattern_pass"],
                          "first_crossing": row["screen"]["first_crossing"]}), flush=True)
    summary = {
        "checkpoint": args.checkpoint, "training": None if ck is None else ck.get("training"),
        "parameter_count": model.parameter_count, "planaria_seeds": args.seeds,
        "strict_retained_count": sum(row["screen"]["retained_after_recut"] for row in cases),
        "matched_sham_count": sum(row["screen"]["matched_sham_pass"] for row in cases),
        "negative_control_count": sum(row["screen"]["negative_controls_pass"] for row in cases),
        "face_seeds": args.face_seeds,
        "full_face_prepattern_count": sum(row["screen"]["full_face_prepattern_pass"] for row in faces),
        "clock_calibrated": False, "target_used_at_inference": False,
        "synthetic_rule_note": "Explicit organizer, coupling, timing, and eye rules are modeling choices, not measured biology.",
        "planaria": cases, "faces": faces,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ("parameter_count", "strict_retained_count",
                                              "matched_sham_count", "negative_control_count",
                                              "full_face_prepattern_count")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--face-seeds", type=int, default=3)
    p.add_argument("--visual-count", type=int, default=1)
    run(p.parse_args())
