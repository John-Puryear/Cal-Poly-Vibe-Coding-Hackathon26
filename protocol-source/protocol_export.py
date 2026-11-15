"""Export the compact protocol branch's learned values for a future JS runtime.

The JSON holds weights and an operation inventory, not an executable browser
implementation. All analytic transition rules remain in protocol_model.py.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from bdh_nca.protocol_model import ProtocolConfig, ProtocolNCA


def digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def export(checkpoint, output):
    source = Path(__file__).parent / "bdh_nca" / "protocol_model.py"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ProtocolConfig(**saved["metadata"]["config"])
    model = ProtocolNCA(config)
    model.load_state_dict(saved["model"], strict=True)
    weights = {
        name: {"shape": list(value.shape), "values": value.detach().cpu().reshape(-1).tolist()}
        for name, value in model.state_dict().items()
    }
    artifact = {
        "format": "bdh-protocol-portable-v1",
        "checkpoint_sha256": digest(checkpoint),
        "transition_source_sha256": digest(source),
        "metadata": model.metadata(),
        "training": saved.get("training"),
        "operations": [
            "pointwise arithmetic/clamp/sigmoid/tanh/exp/sqrt/atan2/sin/cos",
            "1x1 convolution", "3x3/5x5/7x7/9x9 pooling",
            "integer grid shifts", "per-grid reductions and moments",
        ],
        "state_channels_per_cell": 24 + 6 + 6 + 2 * 10 + 3,
        "weights": weights,
        "browser_runtime_included": False,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2))

    # JSON's float text must round-trip exactly to the original float32 values.
    decoded = json.loads(output.read_text())
    reconstructed = {
        name: torch.tensor(item["values"], dtype=torch.float32).reshape(item["shape"])
        for name, item in decoded["weights"].items()
    }
    if any(not torch.equal(reconstructed[name], tensor) for name, tensor in model.state_dict().items()):
        raise RuntimeError("Portable weight round-trip changed a trained value")
    print(json.dumps({"output": str(output), "parameters": model.parameter_count,
                      "float32_round_trip": True, "bytes": output.stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.out)
