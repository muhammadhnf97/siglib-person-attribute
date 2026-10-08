#!/usr/bin/env python
"""Run a TensorRT .engine over a dataset split and score it against
metadata.jsonl ground truth.

Prints per-head accuracy and a table of wrong predictions. Works for both
fp16 and int8 engines — output tensors are whatever dtype the engine
declares.

Usage (run inside the `closure` conda env):

    conda run -n closure python infer_engine.py \\
        --engine siglip2-runs/person-attribute-2/model_int8.engine \\
        --data-dir person-attribute-2/test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from transformers import AutoImageProcessor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LOG = trt.Logger(trt.Logger.WARNING)

LABELS = {
    "age": ["kid", "adult"],
    "gender": ["male", "female", "unknown"],
    "head": ["none", "hijab", "hat", "helm", "other"],
}

TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int8: torch.int8,
    trt.int32: torch.int32,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="split dir holding images + metadata.jsonl")
    ap.add_argument("--num-images", type=int, default=None,
                    help="process N images cycling through the dir (no "
                         "per-image output; prints elapsed time at "
                         "1/10/100/1000/... milestones)")
    ap.add_argument("--base-checkpoint", default="google/siglip2-base-patch16-224")
    ap.add_argument("--show-all", action="store_true",
                    help="print every prediction, not just errors")
    args = ap.parse_args()

    meta = {}
    meta_path = args.data_dir / "metadata.jsonl"
    for line in meta_path.read_text().splitlines():
        rec = json.loads(line)
        meta[rec["file_name"]] = rec

    images = sorted(
        p for p in args.data_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS and p.name in meta
    )
    print(f"{len(images)} labeled images in {args.data_dir}")

    runtime = trt.Runtime(LOG)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    ctx = engine.create_execution_context()

    io = [engine[i] for i in range(engine.num_io_tensors)]
    in_name = next(n for n in io if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out_names = [n for n in io if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    inp = torch.zeros(tuple(engine.get_tensor_shape(in_name)),
                      dtype=TRT_TO_TORCH[engine.get_tensor_dtype(in_name)],
                      device="cuda")
    outs = {n: torch.empty(tuple(engine.get_tensor_shape(n)),
                           dtype=TRT_TO_TORCH[engine.get_tensor_dtype(n)],
                           device="cuda") for n in out_names}
    ctx.set_tensor_address(in_name, inp.data_ptr())
    for n, t in outs.items():
        ctx.set_tensor_address(n, t.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream

    processor = AutoImageProcessor.from_pretrained(args.base_checkpoint)

    correct = {k: 0 for k in LABELS}
    confusions = {}  # (head, expected, predicted) -> count
    total = 0
    milestones = {10**i for i in range(8)}  # 1, 10, 100, ..., 10M
    t_start = time.perf_counter()
    n_iter = args.num_images if args.num_images else len(images)
    for i in range(n_iter):
        p = images[i % len(images)]
        img = Image.open(p).convert("RGB")
        pv = processor(images=img, return_tensors="np")["pixel_values"][0]
        inp.copy_(torch.from_numpy(pv).cuda())
        ctx.execute_async_v3(stream)
        torch.cuda.synchronize()
        total += 1

        if args.num_images:
            if total in milestones:
                el = time.perf_counter() - t_start
                print(f"  {total:>7} imgs in {el:8.2f}s  ({el / total * 1000:6.2f} ms/img)")
            continue

        truth = meta[p.name]
        preds = {}
        mistakes = []
        for n in out_names:
            logits = outs[n].float().cpu().numpy().reshape(-1)
            base = next((k for k in LABELS if n.startswith(k)), n)
            if base not in LABELS:
                continue
            pred = LABELS[base][int(logits.argmax())]
            preds[base] = pred
            if truth.get(base) == pred:
                correct[base] += 1
            else:
                mistakes.append(base)
                key = (base, truth.get(base), pred)
                confusions[key] = confusions.get(key, 0) + 1
        line = f"{p.name}   " + ", ".join(
            f"{k}: {preds[k]}" for k in ("age", "gender", "head") if k in preds
        )
        if mistakes:
            print(line + "   # wrong: " + ", ".join(mistakes))
        elif args.show_all:
            print(line)
    elapsed = time.perf_counter() - t_start

    if args.num_images:
        print(f"\n=== {args.engine.name}: {total:,} imgs in {elapsed:.1f}s "
              f"| {total / elapsed:.1f} img/s ===")
        return

    print(f"\n=== {args.engine.name} on {args.data_dir.name} ===")
    for k, c in correct.items():
        n_err = total - c
        print(f"  {k:7} {c:>2}/{total} = {c / total:.3f}   ({n_err} wrong)")
    print(f"  {elapsed:.1f}s | {total / elapsed:.1f} img/s")

    if confusions:
        print("\nconfusion (expected -> predicted):")
        for (head, exp, pred), cnt in sorted(confusions.items(), key=lambda kv: -kv[1]):
            print(f"  {head:7} {exp:>7} -> {pred:<7} x{cnt}")


if __name__ == "__main__":
    main()
