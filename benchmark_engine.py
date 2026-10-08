#!/usr/bin/env python
"""Benchmark a TensorRT .engine: per-image latency + stress test.

Sibling of benchmark_siglip2.py — same shape (warmup, chunked throughput,
p50/p95/p99, VRAM) but the inference path is TRT runtime instead of
torch. Preprocessing (HF image processor) is identical so numbers are
comparable to the .pt benchmarks.

Usage (run inside the `closure` conda env):

    conda run -n closure python benchmark_engine.py \\
        --engine siglip2-runs/person-attribute-2/model_fp16.engine \\
        --images-dir person-attribute-2/test --num-images 300
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from transformers import AutoImageProcessor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LOG = trt.Logger(trt.Logger.WARNING)


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--num-images", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--project", type=int, default=10000)
    ap.add_argument("--base-checkpoint", default="google/siglip2-base-patch16-224")
    args = ap.parse_args()

    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images in {args.images_dir}")
    processor = AutoImageProcessor.from_pretrained(args.base_checkpoint)

    runtime = trt.Runtime(LOG)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    ctx = engine.create_execution_context()

    in_name = next(engine[i] for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine[i]) == trt.TensorIOMode.INPUT)
    out_names = [engine[i] for i in range(engine.num_io_tensors)
                 if engine.get_tensor_mode(engine[i]) == trt.TensorIOMode.OUTPUT]
    in_shape = tuple(engine.get_tensor_shape(in_name))
    print(f"engine: {args.engine.name} | input {in_name} {in_shape} | outputs {out_names}")

    trt_to_torch = {trt.float32: torch.float32, trt.float16: torch.float16,
                    trt.int8: torch.int8, trt.int32: torch.int32,
                    trt.bfloat16 if hasattr(trt, "bfloat16") else trt.float32: torch.float32}
    in_dtype = trt_to_torch.get(engine.get_tensor_dtype(in_name), torch.float32)
    inp = torch.zeros(in_shape, dtype=in_dtype, device="cuda")
    outs = {
        n: torch.empty(
            tuple(engine.get_tensor_shape(n)), device="cuda",
            dtype=trt_to_torch.get(engine.get_tensor_dtype(n), torch.float32),
        )
        for n in out_names
    }
    ctx.set_tensor_address(in_name, inp.data_ptr())  # TRT>=10: unified setter
    for n, t in outs.items():
        ctx.set_tensor_address(n, t.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream

    def run_one(i: int):
        img = Image.open(images[i % len(images)]).convert("RGB")
        pv = processor(images=img, return_tensors="np")["pixel_values"][0]
        inp.copy_(torch.from_numpy(pv).cuda())
        ctx.execute_async_v3(stream)
        torch.cuda.synchronize()

    for i in range(args.warmup):
        run_one(i)
    torch.cuda.reset_peak_memory_stats()

    latencies, t_start = [], time.perf_counter()
    for i in range(args.num_images):
        t0 = time.perf_counter()
        run_one(i)
        latencies.append(time.perf_counter() - t0)
        done = i + 1
        if done % args.chunk == 0 or done == args.num_images:
            chunk_ms = statistics.mean(latencies[-args.chunk:]) * 1000
            overall = done / (time.perf_counter() - t_start)
            print(f"  [{done:>5}/{args.num_images}] chunk avg {chunk_ms:7.1f} ms/img | overall {overall:6.2f} img/s")

    total = time.perf_counter() - t_start
    s = sorted(latencies)
    per_img = total / args.num_images
    print(f"  mean {statistics.mean(latencies) * 1000:.1f} ms | p50 {statistics.median(s) * 1000:.1f}"
          f" | p95 {s[int(len(s) * .95)] * 1000:.1f} | p99 {s[int(len(s) * .99)] * 1000:.1f}"
          f" | min {s[0] * 1000:.1f} | max {s[-1] * 1000:.1f} ms")
    print(f"  {1 / per_img:.2f} img/s | {args.project:,} imgs -> {fmt_duration(per_img * args.project)}")
    print(f"  peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")
    print("\nlast-image predictions:")
    labels = {"age": ["Kid", "Adult"], "gender": ["Male", "Female", "Unknown"],
              "head": ["None", "Hijab", "Hat", "Helm", "Other"]}
    for n in out_names:
        logits = outs[n].cpu().numpy().reshape(-1)
        base = next((k for k in labels if n.startswith(k)), n)
        idx = int(logits.argmax())
        lbl = labels.get(base, list(map(str, range(len(logits)))))[idx]
        print(f"  {base:7} -> {lbl:8} logits={np.round(logits, 3)}")


if __name__ == "__main__":
    main()
