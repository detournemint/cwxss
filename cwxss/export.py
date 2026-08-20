"""Export the trained decoder to ONNX.

torch is a large dependency and a training tool. A radio in a park does not need
it: the model is 180 KB of weights and the arithmetic to run it is ordinary. ONNX
Runtime is a fraction of the size, starts instantly, and is the same numbers.

    python3 cwxss/export.py models/cw.pt models/cw.onnx
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M     # noqa: E402


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "models/cw.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "models/cw.onnx"
    state = torch.load(src, map_location="cpu", weights_only=False)["state"]
    net = M.CWNet(hidden=M.hidden_of(state))
    net.load_state_dict(state)
    net.eval()
    # A dynamic time axis: transmissions are not a fixed length, and padding
    # every clip to a maximum would waste most of the computation.
    dummy = torch.zeros(1, 800)
    torch.onnx.export(
        net, dummy, dst,
        input_names=["envelope"], output_names=["logprobs"],
        dynamic_axes={"envelope": {0: "batch", 1: "frames"},
                      "logprobs": {0: "batch", 1: "frames"}},
        opset_version=17)
    size = Path(dst).stat().st_size / 1024
    print(f"  wrote {dst}  ({size:.0f} KB)")


if __name__ == "__main__":
    main()
