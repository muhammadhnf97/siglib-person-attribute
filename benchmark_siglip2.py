#!/usr/bin/env python
"""Benchmark SigLIP2 attribute inference: per-image latency + stress test.

Times exactly what the collector's /prelabel path does — image open, HF
preprocess, forward pass, argmax — by reusing siglip2_infer.py's
load_predictor/predict_values (works for float32, fp16, and int8-dynamic
checkpoints). Runs N sequential inferences in chunks so you can see whether
throughput degrades under sustained load, then projects wall-clock time for
a larger image count.

`--model-dir` accepts:
  - a single .pt file,                          -> benchmarks that checkpoint
  - a dir holding a checkpoint (best.pt etc.),  -> benchmarks it
  - a dir holding model subdirs (e.g. template/ -> benchmarks every model in
    <template>/model/),                            it and prints a comparison

Usage (run inside the `closure` conda env):

    conda run -n closure python benchmark_siglip2.py \\
        --model-dir /path/to/template/siglip2_person_attribute/model \\
        --images-dir /path/to/some/image/dir \\
        --num-images 300 --warmup 10
"""
from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path

import torch

import siglip2_infer

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def find_models(model_dir: Path) -> list[Path]:
    """Resolve --model-dir into a list of checkpoint paths/dirs to benchmark.

    A dir holding several checkpoint files (e.g. a quantize output with both
    fp16.pt and int8-dynamic.pt) benchmarks each file — they differ in speed.
    last.pt is skipped when best.pt exists since same-arch weights benchmark
    identically for latency.
    """
    if model_dir.is_file():
        return [model_dir]
    ckpt_files = [
        model_dir / n
        for n in ("best.pt", "fp16.pt", "int8-dynamic.pt", "int8-torchao.pt")
        if (model_dir / n).is_file()
    ]
    if not ckpt_files and (model_dir / "last.pt").is_file():
        ckpt_files = [model_dir / "last.pt"]
    if ckpt_files:
        return ckpt_files
    return sorted(
        d for d in model_dir.iterdir()
        if d.is_dir() and siglip2_infer.find_checkpoint_file(d)
    )


def run_benchmark(name, model_path, images, args):
    predictor = siglip2_infer.load_predictor(model_path)
    model, processor, groups, device = predictor
    dtype = next(model.parameters()).dtype
    print(f"\n=== {name} ===")
    print(f"checkpoint: {siglip2_infer.find_checkpoint_file(model_path)}")
    print(f"device={device} dtype={dtype} heads={[g['name'] for g in groups]}")

    # predict_values maps checkpoint group outputs into an index layout —
    # rebuild an equivalent one so the benchmark calls the real path.
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
    sorted_l = sorted(latencies)
    per_img = total / args.num_images
    peak_mib = (
        torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
    )

    print(
        f"  mean {statistics.mean(latencies) * 1000:.1f} ms"
        f" | p50 {statistics.median(sorted_l) * 1000:.1f}"
        f" | p95 {sorted_l[int(len(sorted_l) * 0.95)] * 1000:.1f}"
        f" | p99 {sorted_l[int(len(sorted_l) * 0.99)] * 1000:.1f}"
        f" | min {sorted_l[0] * 1000:.1f} | max {sorted_l[-1] * 1000:.1f} ms"
    )
    print(
        f"  {1 / per_img:.2f} img/s | {args.project:,} imgs ->"
        f" {fmt_duration(per_img * args.project)}"
    )
    if peak_mib is not None:
        print(f"  peak VRAM: {peak_mib:.0f} MiB")

    del model, predictor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": name,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "mean_ms": statistics.mean(latencies) * 1000,
        "p95_ms": sorted_l[int(len(sorted_l) * 0.95)] * 1000,
        "img_s": 1 / per_img,
        "proj": per_img * args.project,
        "peak_mib": peak_mib,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-dir", required=True, type=Path,
        help="Checkpoint file, model dir, or parent dir of model dirs (runs all)",
    )
    parser.add_argument(
        "--images-dir", required=True, type=Path,
        help="Directory of images to feed the model (cycles if fewer than --num-images)",
    )
    parser.add_argument("--num-images", type=int, default=300,
                        help="Inference calls to time per model (default 300)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Untimed warmup inferences per model (default 10)")
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

    models = find_models(args.model_dir)
    if not models:
        raise SystemExit(f"No checkpoints found under {args.model_dir}")
    print(f"{len(images)} unique images; benchmarking {len(models)} model(s): "
          + ", ".join(m.name for m in models))

    results = []
    for m in models:
        label = m.stem if m.is_file() else m.name
        if m.is_file() and m.stem in ("best", "last"):
            label = f"{m.parent.name}/{m.name}"
        results.append(run_benchmark(label, m, images, args))

    if len(results) > 1:
        print("\n=== comparison (sequential, batch=1) ===")
        print(f"{'model':<32} {'device':<5} {'dtype':<8} {'mean':>8} {'p95':>8}"
              f" {'img/s':>7} {f'{args.project // 1000}k imgs':>9} {'VRAM':>7}")
        for r in sorted(results, key=lambda r: -r["img_s"]):
            vram = f"{r['peak_mib']:.0f}MiB" if r["peak_mib"] else "cpu"
            print(
                f"{r['name']:<32} {r['device']:<5} {r['dtype']:<8}"
                f" {r['mean_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms"
                f" {r['img_s']:>7.2f} {fmt_duration(r['proj']):>9} {vram:>7}"
            )
        print("\nThroughput hint: watch the per-chunk avgs above — steadily")
        print("rising chunks mean the model throttles under sustained load.")


if __name__ == "__main__":
    main()
