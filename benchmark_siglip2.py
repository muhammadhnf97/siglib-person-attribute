#!/usr/bin/env python
"""Benchmark SigLIP2 attribute inference: per-image latency + stress test.

Times exactly what the collector's /prelabel path does — image open, HF
preprocess, forward pass, argmax — by reusing siglip2_infer.py's
load_predictor/predict_values (works for float32, fp16, and int8-dynamic
checkpoints). Runs N sequential inferences in chunks so you can see whether
throughput degrades under sustained load, then projects wall-clock time for
a larger image count.

Usage (run inside the `closure` conda env):

    conda run -n closure python benchmark_siglip2.py \\
        --model-dir /path/to/template/model/<name> \\
        --images-dir /path/to/some/image/dir \\
        --num-images 300 --warmup 10
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Reuse the collector's real inference path so numbers match production.
sys.path.insert(
    0, "/home/me/disk-256/00_RND_MODELS/paddlepaddle/dataset-collector/server"
)
import torch

import siglip2_infer

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-dir", required=True, type=Path,
        help="Directory containing best.pt/last.pt (as under template/.../model/)",
    )
    parser.add_argument(
        "--images-dir", required=True, type=Path,
        help="Directory of images to feed the model (cycles if fewer than --num-images)",
    )
    parser.add_argument("--num-images", type=int, default=300,
                        help="Inference calls to time (default 300)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Untimed warmup inferences (default 10)")
    parser.add_argument("--chunk", type=int, default=50,
                        help="Print per-chunk throughput every N images (default 50)")
    parser.add_argument("--project", type=int, default=10000,
                        help="Image count to project total wall-clock for (default 10000)")
    args = parser.parse_args()

    images = sorted(
        p for p in args.images_dir.iterdir() if p.suffix.lower() in IMG_EXTS
    )
    if not images:
        raise SystemExit(f"No images found in {args.images_dir}")
    print(f"{len(images)} unique images available; cycling as needed")

    predictor = siglip2_infer.load_predictor(args.model_dir)
    model, processor, groups, device = predictor
    dtype = next(model.parameters()).dtype
    print(f"device={device} dtype={dtype} heads={[g['name'] for g in groups]}")

    # predict_values maps checkpoint group outputs into a template's index
    # layout — rebuild an equivalent layout so benchmark calls the real path.
    attr_groups = []
    offset = 0
    for g in groups:
        n = len(g["options"])
        attr_groups.append(
            {"name": g["name"], "options": g["options"],
             "indices": list(range(offset, offset + n))}
        )
        offset += n

    def run_one(i: int):
        return siglip2_infer.predict_values(
            images[i % len(images)], predictor, attr_groups
        )

    for i in range(args.warmup):
        run_one(i)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"warmup done ({args.warmup} imgs); timing {args.num_images} inferences\n")

    latencies = []
    t_start = time.perf_counter()
    for i in range(args.num_images):
        t0 = time.perf_counter()
        run_one(i)
        latencies.append(time.perf_counter() - t0)
        done = i + 1
        if done % args.chunk == 0 or done == args.num_images:
            chunk_ms = statistics.mean(latencies[-args.chunk:]) * 1000
            overall = done / (time.perf_counter() - t_start)
            print(
                f"  [{done:>5}/{args.num_images}] chunk avg {chunk_ms:7.1f} ms/img"
                f" | overall {overall:6.2f} img/s"
            )

    total = time.perf_counter() - t_start
    mean_ms = statistics.mean(latencies) * 1000
    sorted_l = sorted(latencies)
    p50 = statistics.median(sorted_l) * 1000
    p95 = sorted_l[int(len(sorted_l) * 0.95)] * 1000
    p99 = sorted_l[int(len(sorted_l) * 0.99)] * 1000
    per_img = total / args.num_images

    print("\n=== per-image latency (sequential, batch=1) ===")
    print(f"  mean {mean_ms:.1f} ms | p50 {p50:.1f} | p95 {p95:.1f} | p99 {p99:.1f}")
    print(f"  min  {sorted_l[0] * 1000:.1f} ms | max {sorted_l[-1] * 1000:.1f} ms")
    print(f"\n=== throughput ===")
    print(f"  {args.num_images} images in {fmt_duration(total)} ({1 / per_img:.2f} img/s)")
    print(f"  projection for {args.project:,} images: {fmt_duration(per_img * args.project)}")
    if device.type == "cuda":
        print(
            f"  peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB"
            f" allocated, {torch.cuda.max_memory_reserved() / 1024**2:.0f} MiB reserved"
        )
    print("\nThroughput hint: chunk avgs above flag degradation — if later chunks")
    print("get steadily slower, sustained load is throttling the model.")


if __name__ == "__main__":
    main()
