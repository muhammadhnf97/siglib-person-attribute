#!/usr/bin/env python
"""Predict attributes for images with any exported model format.

Backends are picked from the --model file extension:

  .pt      PyTorch checkpoint (fp32/fp16/int8-dynamic/int8-torchao) via
           siglip2_infer — also accepts a directory holding a checkpoint.
  .onnx    ONNX Runtime — CUDA EP if available, else CPU.
  .engine  TensorRT engine — NVIDIA GPU only, batch=1 builds.

--data-dir points at a split dir holding images; if it has metadata.jsonl,
per-head accuracy and a confusion summary are printed. --num-images cycles
through the images for pure timing runs without scoring.

Usage (run inside the `closure` conda env):

    python predict.py --model siglip2-runs/person-attribute-2/model_int8.engine \
        --data-dir person-attribute-2/test

    python predict.py --model siglip2-runs/person-attribute-2/best.pt \
        --data-dir person-attribute-2/test --show-all
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------- backends
class PtBackend:
    def __init__(self, path: Path):
        import torch
        import siglip2_infer

        self.torch = torch
        self.predictor = siglip2_infer.load_predictor(path)
        model, self.processor, groups, self.device = self.predictor
        self.labels = {g["name"].lower(): [str(o).lower() for o in g["options"]]
                       for g in groups}
        print(f"backend=torch device={self.device} dtype={next(model.parameters()).dtype}")

    def predict(self, img: Image.Image) -> dict[str, int]:
        model, processor, _, device = self.predictor
        pv = processor(images=img, return_tensors="pt")["pixel_values"].to(
            device=device, dtype=next(model.parameters()).dtype
        )
        with self.torch.no_grad():
            logits = model(pv)
        return {k: int(v.argmax(-1).item()) for k, v in logits.items()}


class OnnxBackend:
    def __init__(self, path: Path):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        print(f"backend=onnxruntime providers={self.sess.get_providers()}")

    def predict(self, img: Image.Image) -> dict[str, int]:
        pv = self.processor(images=img, return_tensors="np")["pixel_values"]
        outs = self.sess.run(None, {self.input_name: pv})
        names = [o.name for o in self.sess.get_outputs()]
        return {n: int(np.asarray(o).argmax(-1).item()) for n, o in zip(names, outs)}


class EngineBackend:
    def __init__(self, path: Path):
        import tensorrt as trt
        import torch

        self.torch = torch
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        self.ctx = engine.create_execution_context()
        io = [engine[i] for i in range(engine.num_io_tensors)]
        in_name = next(n for n in io if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        out_names = [n for n in io if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        to_torch = {trt.float32: torch.float32, trt.float16: torch.float16,
                    trt.int8: torch.int8, trt.int32: torch.int32}
        self.inp = torch.zeros(tuple(engine.get_tensor_shape(in_name)),
                               dtype=to_torch.get(engine.get_tensor_dtype(in_name), torch.float32),
                               device="cuda")
        self.outs = {n: torch.empty(tuple(engine.get_tensor_shape(n)),
                                    dtype=to_torch.get(engine.get_tensor_dtype(n), torch.float32),
                                    device="cuda") for n in out_names}
        self.ctx.set_tensor_address(in_name, self.inp.data_ptr())
        for n, t in self.outs.items():
            self.ctx.set_tensor_address(n, t.data_ptr())
        self.stream = torch.cuda.current_stream().cuda_stream
        print(f"backend=tensorrt input={in_name} outputs={out_names}")

    def predict(self, img: Image.Image) -> dict[str, int]:
        pv = self.processor(images=img, return_tensors="np")["pixel_values"][0]
        self.inp.copy_(self.torch.from_numpy(pv).cuda())
        self.ctx.execute_async_v3(self.stream)
        self.torch.cuda.synchronize()
        return {n: int(t.float().cpu().numpy().reshape(-1).argmax())
                for n, t in self.outs.items()}


# ------------------------------------------------------------------ driver
def load_labels(model_path: Path) -> dict[str, list[str]]:
    """Group name -> lowercase option list. From labels.json next to the
    model (written by train.py), else a --labels path. Empty if absent."""
    candidates = [model_path.parent / "labels.json"] if model_path.is_file() else []
    for c in candidates:
        if c.is_file():
            data = json.loads(c.read_text())
            return {k.lower(): [str(o).lower() for o in v["options"]]
                    for k, v in data.items()}
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path,
                    help=".pt / .onnx / .engine file, or a dir holding a checkpoint")
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="dir of images; metadata.jsonl there enables scoring")
    ap.add_argument("--labels", type=Path, default=None,
                    help="labels.json path (auto-detected next to the model otherwise)")
    ap.add_argument("--num-images", type=int, default=None,
                    help="cycle through images N times for timing, no scoring")
    ap.add_argument("--show-all", action="store_true",
                    help="print every prediction, not just errors")
    ap.add_argument("--base-checkpoint", default="google/siglip2-base-patch16-224",
                    help="HF checkpoint for preprocessing (onnx/engine only)")
    args = ap.parse_args()

    images = sorted(p for p in args.data_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"no images in {args.data_dir}")

    suffix = args.model.suffix.lower() if args.model.is_file() else ".pt"
    backend = {".onnx": OnnxBackend, ".engine": EngineBackend}.get(suffix, PtBackend)(args.model)

    labels = getattr(backend, "labels", None) or load_labels(args.labels or args.model)
    if not isinstance(backend, PtBackend):
        from transformers import AutoImageProcessor
        backend.processor = AutoImageProcessor.from_pretrained(args.base_checkpoint)
        if not labels:
            print("warning: no labels.json found — printing class indices")

    meta = {}
    meta_path = args.data_dir / "metadata.jsonl"
    if meta_path.is_file():
        for line in meta_path.read_text().splitlines():
            rec = json.loads(line)
            meta[rec["file_name"]] = rec
        meta = {k: v for k, v in meta.items() if (args.data_dir / k).is_file()}
        if meta:
            print(f"{len(meta)} labeled images in {args.data_dir}")

    correct = {k: 0 for k in labels}
    confusions = {}
    total = 0
    milestones = {10**i for i in range(8)}
    t_start = time.perf_counter()
    n_iter = args.num_images if args.num_images else len(images)
    for i in range(n_iter):
        p = images[i % len(images)]
        preds_idx = backend.predict(Image.open(p).convert("RGB"))
        total += 1

        if args.num_images:
            if total in milestones:
                el = time.perf_counter() - t_start
                print(f"  {total:>7} imgs in {el:8.2f}s  ({el / total * 1000:6.2f} ms/img)")
            continue

        preds = {k: (labels.get(k) or [str(v)])[v] for k, v in preds_idx.items()}
        truth = meta.get(p.name, {})
        mistakes = []
        for k, pred in preds.items():
            if truth.get(k) == pred:
                correct[k] += 1
            else:
                mistakes.append(k)
                key = (k, truth.get(k), pred)
                confusions[key] = confusions.get(key, 0) + 1
        line = f"{p.name}   " + ", ".join(f"{k}: {v}" for k, v in preds.items())
        if mistakes:
            print(line + "   # wrong: " + ", ".join(mistakes))
        elif args.show_all or not meta:
            print(line)
    elapsed = time.perf_counter() - t_start

    if args.num_images:
        print(f"\n=== {args.model.name}: {total:,} imgs in {elapsed:.1f}s "
              f"| {total / elapsed:.1f} img/s ===")
        return

    print(f"\n=== {args.model.name} on {args.data_dir.name} ===")
    if meta and labels:
        for k, c in correct.items():
            n_err = total - c
            print(f"  {k:7} {c:>2}/{total} = {c / total:.3f}   ({n_err} wrong)")
    print(f"  {elapsed:.1f}s | {total / elapsed:.1f} img/s")
    if confusions:
        print("\nconfusion (expected -> predicted):")
        for (head, exp, pred), cnt in sorted(confusions.items(), key=lambda kv: -kv[1]):
            print(f"  {head:7} {str(exp):>7} -> {pred:<7} x{cnt}")


if __name__ == "__main__":
    main()
