#!/usr/bin/env python
"""Post-training convert a checkpoint from train_siglip2_attributes.py to a
smaller/faster dtype, and measure exactly what it costs in accuracy.

Two independent modes:

  --mode fp16
      Cast weights to float16. Not quantization — still floats, just lower
      precision. ~2x smaller on disk, fast on any GPU with fp16 support
      (this machine's RTX 3080 included). Recommended default when serving
      stays on this GPU — negligible accuracy cost.

  --mode int8-dynamic
      Dynamically quantize the model's nn.Linear layers to int8 weights
      (PyTorch `quantize_dynamic`). Activations stay float — this is the
      least aggressive form of int8 quantization (no calibration data
      needed), but PyTorch's dynamic-quantized kernels are CPU-only, so this
      mode always evaluates/runs on CPU regardless of GPU availability. Only
      useful if inference is actually moving off the GPU.

Always evaluates the ORIGINAL float32 checkpoint and the converted one on the
same test split (when --export-dir/--config are given) so the accuracy
tradeoff is measured, not assumed, before you decide whether to ship it.

Usage (run inside the `closure` conda env):

    conda run -n closure python quantize_siglip2.py \\
        --checkpoint /path/to/best.pt \\
        --export-dir /path/to/dataset/export \\
        --config /path/to/dataset-collector/server/template/siglip2/siglip2_person_attribute/config.yaml \\
        --output-dir /path/to/runs/quantized \\
        --mode fp16
"""
from __future__ import annotations

import argparse
import copy
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from train_siglip2_attributes import (
    IGNORE_INDEX,
    AttributeGroup,
    AttributeJsonlDataset,
    MultiAttributeSiglip2,
    collate,
)


def load_checkpoint(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    base_checkpoint = ckpt.get("checkpoint")
    raw_groups = ckpt.get("groups") or []
    if not base_checkpoint or not raw_groups:
        raise ValueError(f"{ckpt_path} is missing 'checkpoint'/'groups' metadata")

    groups = [
        AttributeGroup(name=g["name"], alias=g.get("alias", g["name"]), options=g["options"])
        for g in raw_groups
    ]
    model = MultiAttributeSiglip2(base_checkpoint, groups)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, base_checkpoint, groups


@torch.no_grad()
def evaluate(model, loader, groups, device):
    model_dtype = next(model.parameters()).dtype
    totals = {g.key: 0 for g in groups}
    correct = {g.key: 0 for g in groups}
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device=device, dtype=model_dtype)
        labels = {k: v.to(device) for k, v in labels.items()}
        logits = model(pixel_values)
        for group in groups:
            target = labels[group.key]
            mask = target != IGNORE_INDEX
            if not mask.any():
                continue
            # argmax in float32 regardless of model dtype — fp16/int8-dequant
            # logits are fine to compare in higher precision.
            preds = logits[group.key].float().argmax(dim=-1)
            correct[group.key] += int((preds[mask] == target[mask]).sum())
            totals[group.key] += int(mask.sum())
    per_group_acc = {
        g.key: (correct[g.key] / totals[g.key] if totals[g.key] else None) for g in groups
    }
    valid = [a for a in per_group_acc.values() if a is not None]
    macro = sum(valid) / len(valid) if valid else 0.0
    return per_group_acc, macro


def print_accuracy(label: str, groups, acc, macro):
    print(f"\n{label} macro_acc={macro:.4f}")
    for g in groups:
        a = acc[g.key]
        print(f"  {g.alias:<16} {'n/a' if a is None else f'{a:.3f}'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained best.pt/last.pt from train_siglip2_attributes.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["fp16", "int8-dynamic"], required=True)
    parser.add_argument("--export-dir", type=Path, default=None, help="Dataset export dir, for before/after eval on --test-dir")
    parser.add_argument("--config", type=Path, default=None, help="Template config.yaml, for before/after eval")
    parser.add_argument("--test-dir", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if (args.export_dir is None) != (args.config is None):
        parser.error("--export-dir and --config must be given together")

    model, base_checkpoint, groups = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = None
    if args.export_dir is not None:
        processor = AutoImageProcessor.from_pretrained(base_checkpoint)
        test_dir = args.export_dir / args.test_dir
        test_ds = AttributeJsonlDataset(test_dir, groups, processor)
        loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=lambda b: collate(b, groups),
        )
        print(f"Evaluation set: {len(test_ds)} images from {test_dir}")

    baseline_acc = baseline_macro = None
    if loader is not None:
        model.to(device)
        baseline_acc, baseline_macro = evaluate(model, loader, groups, device)
        print_accuracy("Baseline (float32)", groups, baseline_acc, baseline_macro)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "fp16":
        quant_model = copy.deepcopy(model).half()
        eval_device = device
        quant_model.to(eval_device)
    else:  # int8-dynamic
        # torch.quantization.quantize_dynamic is deprecated in favor of
        # torchao as of this torch version, but still functional and avoids
        # an extra dependency for one call — revisit if it's actually
        # removed in a future torch upgrade.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cpu_model = copy.deepcopy(model).to("cpu").float()
            quant_model = torch.quantization.quantize_dynamic(
                cpu_model, {nn.Linear}, dtype=torch.qint8,
            )
        eval_device = torch.device("cpu")

    out_ckpt = {
        "model_state": quant_model.state_dict(),
        "checkpoint": base_checkpoint,
        "groups": [{"name": g.name, "alias": g.alias, "options": g.options} for g in groups],
        "quantization": args.mode,
    }
    out_path = args.output_dir / f"{args.mode}.pt"
    torch.save(out_ckpt, out_path)

    if loader is not None:
        quant_acc, quant_macro = evaluate(quant_model, loader, groups, eval_device)
        print_accuracy(args.mode, groups, quant_acc, quant_macro)
        print(f"\n  delta vs baseline: {quant_macro - baseline_macro:+.4f} macro_acc")

    orig_mb = args.checkpoint.stat().st_size / (1024 * 1024)
    new_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSize: {orig_mb:.1f}MB -> {new_mb:.1f}MB ({orig_mb / new_mb:.2f}x smaller)")
    print(f"Saved {args.mode} checkpoint to {out_path}")


if __name__ == "__main__":
    main()
