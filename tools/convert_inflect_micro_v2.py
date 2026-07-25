#!/usr/bin/env python3
"""Export Inflect Micro v2 inference weights to a plain safetensors file.

The public checkpoint stores several weight-normalized modules in PyTorch
checkpoint form. The Python inference path calls remove_weight_norm() before
generation, so the C++ runtime expects these already-folded inference tensors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    sys.path.insert(0, str(model_dir))
    sys.path.insert(0, str(model_dir / "runtime"))

    from inference import InflectTTS  # noqa: WPS433

    tts = InflectTTS(model_dir, device="cpu")
    state = {}
    for name, tensor in tts.model.state_dict().items():
        if not torch.is_tensor(tensor):
            continue
        state[name] = tensor.detach().cpu().contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(args.output))
    print(f"wrote {len(state)} tensors to {args.output}")


if __name__ == "__main__":
    main()
