"""Regenerate the deterministic PyTorch/browser parity fixture for temporal v3.

The starting seed, compact checkpoint, fire masks, cut, and control values are
identical to those used by tests/protocol-conformance.mjs. The transition code
comes from bdh_nca.protocol_model; no target anatomy enters this sequence.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from bdh_nca.protocol_model import ProtocolConfig, ProtocolNCA


ROOT = Path(__file__).resolve().parents[1]
FIRE1 = {"a": 17, "b": 13, "m": 29, "limit": 15}
FIRE2 = {"a": 19, "b": 7, "m": 31, "limit": 16}
CONTROL = {"current": .4, "broad": .8, "long": .25, "purine": .7,
           "middle_columns": [12, 18]}


def fire_mask(spec, height, width):
    index = torch.arange(height * width).reshape(1, 1, height, width)
    return (((index * spec["a"] + spec["b"]) % spec["m"]) < spec["limit"]).float()


def flatten(state):
    arrays = [tensor.detach().cpu().numpy().astype("<f4", copy=False).ravel()
              for tensor in state.tensors()]
    return arrays, np.concatenate(arrays)


@torch.no_grad()
def generate(checkpoint, seed_path, output_binary, output_manifest):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ProtocolConfig(**saved["metadata"]["config"])
    model = ProtocolNCA(config).eval()
    model.load_state_dict(saved["model"], strict=True)
    seed = json.loads(seed_path.read_text())
    height, width = seed["height"], seed["width"]
    cells = torch.tensor(seed["seed"], dtype=torch.float32).reshape(1, 24, height, width)

    state = model.initial(cells)
    states = [state]
    state = model(state, fire=fire_mask(FIRE1, height, width))
    states.append(state)

    xx = torch.arange(width).reshape(1, 1, 1, width)
    keep = ((xx >= 10) & (xx <= 18)).float().expand(1, 1, height, width)
    state = model.damage(state, keep)
    states.append(state)

    middle = ((xx >= CONTROL["middle_columns"][0])
              & (xx <= CONTROL["middle_columns"][1])).float().expand(1, 1, height, width)
    control = torch.zeros(1, 4, height, width)
    for channel, key in enumerate(("current", "broad", "long", "purine")):
        control[:, channel:channel+1] = CONTROL[key] * middle
    state = model(state, control=control, fire=fire_mask(FIRE2, height, width))
    states.append(state)

    rows = []
    lengths = None
    for item in states:
        arrays, row = flatten(item)
        if lengths is None:
            lengths = [len(array) for array in arrays]
        elif lengths != [len(array) for array in arrays]:
            raise RuntimeError("Protocol state tensor lengths changed during fixture")
        rows.append(row)
    result = np.stack(rows).astype("<f4", copy=False)
    output_binary.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_binary.write_bytes(gzip.compress(result.tobytes(), compresslevel=9, mtime=0))
    manifest = {
        "format": "bdh-protocol-parity-v3",
        "source_seed": seed.get("source_seed"),
        "shape": list(result.shape),
        "state_order": ["initial", "one", "cut", "treated"],
        "tensor_lengths": lengths,
        "fire1": FIRE1,
        "fire2": FIRE2,
        "cut_columns": [10, 18],
        "control": CONTROL,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"fixture": str(output_binary), "manifest": str(output_manifest),
                      "states": result.shape[0], "values_per_state": result.shape[1]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "protocol-source/checkpoint_temporal_v3.pt")
    parser.add_argument("--seed", type=Path, default=ROOT / "dist/protocol/planaria-seed.json")
    parser.add_argument("--binary", type=Path, default=ROOT / "tests/protocol-fixture.f32.gz")
    parser.add_argument("--manifest", type=Path, default=ROOT / "tests/protocol-fixture.json")
    args = parser.parse_args()
    generate(args.checkpoint, args.seed, args.binary, args.manifest)
