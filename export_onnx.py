#!/usr/bin/env python
"""Export a trained SigLIP2 attribute checkpoint (.pt) to ONNX.

This is the bridge step for every non-PyTorch deployment: TensorRT int8/fp16
engines, ONNX Runtime CPU/GPU inference, TPU-MLIR bmodel — all start from an
.onnx graph, so nothing downstream works without this file being correct.

The exported graph covers the neural net only (vision tower + per-group
heads). Image decode/preprocessing stays in Python; the model outputs one
logits tensor per attribute group, named after the group (age, gender, ...).

Usage (run inside the `closure` conda env):

    conda run -n closure python export_onnx.py \
        --checkpoint /path/to/run/best.pt \
        --output /path/to/model.onnx \
        --verify-image /path/to/some.jpg   # optional parity check vs torch
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import siglip2_infer


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="best.pt/last.pt file or a dir containing one (fp32 only — "
             "export from the unquantized checkpoint, not fp16/int8 outputs)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--verify-image", type=Path, default=None,
        help="If given, runs this image through torch and ONNX Runtime and "
             "compares per-head predictions",
    )
    args = parser.parse_args()

    ckpt_path = siglip2_infer.find_checkpoint_file(args.checkpoint)
    if ckpt_path is None:
        raise SystemExit(f"No checkpoint found at {args.checkpoint}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if ckpt.get("quantization") not in (None, "fp32"):
        raise SystemExit(
            f"Checkpoint is {ckpt['quantization']} — export from the original "
            "float32 best.pt, not a quantized checkpoint"
        )

    predictor = siglip2_infer.load_predictor(args.checkpoint)
    model, processor, groups, _ = predictor
    model = model.float().cpu().eval()

    head_names = [g["name"].lower() for g in groups]
    dummy = torch.randn(1, 3, 224, 224)

    class _OnnxExportable(torch.nn.Module):
        """Flatten the dict output into positional tensors for ONNX."""
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            logits = self.inner(pixel_values)
            return tuple(logits[k] for k in head_names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _OnnxExportable(model),
        dummy,
        args.output,
        input_names=["pixel_values"],
        output_names=head_names,
        dynamic_axes={
            "pixel_values": {0: "batch"},
            **{name: {0: "batch"} for name in head_names},
        },
        opset_version=args.opset,
    )

    # The dynamo exporter writes weights to a separate .onnx.data file —
    # consolidate into a single self-contained .onnx so deployment is one file.
    import onnx

    model_proto = onnx.load(str(args.output))
    onnx.save(model_proto, str(args.output), save_as_external_data=False)
    (args.output.parent / f"{args.output.name}.data").unlink(missing_ok=True)

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Exported {args.output} ({size_mb:.1f} MB), outputs: {head_names}")

    if args.verify_image:
        import onnxruntime as ort
        import numpy as np
        from PIL import Image

        pv = processor(
            images=Image.open(args.verify_image).convert("RGB"),
            return_tensors="pt",
        )["pixel_values"]
        with torch.no_grad():
            ref = model(pv)
        sess = ort.InferenceSession(str(args.output))
        onnx_out = sess.run(None, {"pixel_values": pv.numpy()})

        print(f"\nParity check on {args.verify_image.name}:")
        ok = True
        for i, name in enumerate(head_names):
            ref_logits = ref[name].numpy()
            diff = np.abs(ref_logits - onnx_out[i]).max()
            match = ref_logits.argmax() == onnx_out[i].argmax()
            ok &= bool(match)
            print(f"  {name:<10} argmax match: {match}   max |diff|: {diff:.6f}")
        print("PASS" if ok else "FAIL — predictions diverge, do not deploy this .onnx")


if __name__ == "__main__":
    main()
