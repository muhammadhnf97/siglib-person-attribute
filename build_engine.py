#!/usr/bin/env python
"""Compile model.onnx into a TensorRT engine for this machine's GPU.

  --precision fp16   fp16 Tensor Core kernels, no calibration needed.
                     Good first build — verifies the whole pipeline works.

  --precision int8   int8 Tensor Core kernels. TRT 11 dropped the old
                     IInt8Calibrator API, so this is two stages:
                       1. ONNX Runtime static-quantizes model.onnx with
                          real calibration images (--calib-dir), baking
                          QuantizeLinear/DequantizeLinear nodes + scales
                          into a QDQ graph (saved next to --output).
                       2. TRT builds from that graph — it reads the QDQ
                          scales directly, no runtime calibrator.

Engines are hardware-specific — rebuild if the GPU/driver/TRT version
changes.

Usage (run inside the `closure` conda env):

    conda run -n closure python build_engine.py \
        --onnx siglip2-runs/person-attribute-2/model.onnx \
        --output siglip2-runs/person-attribute-2/model_fp16.engine \
        --precision fp16

    conda run -n closure python build_engine.py \
        --onnx siglip2-runs/person-attribute-2/model.onnx \
        --output siglip2-runs/person-attribute-2/model_int8.engine \
        --precision int8 \
        --calib-dir person-attribute-2/test --calib-num 64
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorrt as trt
from PIL import Image
from transformers import AutoImageProcessor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LOG = trt.Logger(trt.Logger.INFO)


def quantize_onnx_int8(onnx_path: Path, qdq_path: Path, calib_dir: Path,
                       num_images: int, base_checkpoint: str):
    """Produce a QDQ int8 ONNX via ONNX Runtime static quantization."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantType,
        quantize_static,
    )

    processor = AutoImageProcessor.from_pretrained(base_checkpoint)
    paths = sorted(p for p in calib_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise SystemExit(f"no images in {calib_dir}")
    paths = paths[:num_images]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(paths)

        def get_next(self):
            p = next(self.it, None)
            if p is None:
                return None
            img = Image.open(p).convert("RGB")
            pv = processor(images=img, return_tensors="np")["pixel_values"]
            return {"pixel_values": pv.astype(np.float32)}

    print(f"int8 calibration: {len(paths)} images from {calib_dir}")
    quantize_static(
        model_input=str(onnx_path),
        model_output=str(qdq_path),
        calibration_data_reader=Reader(),
        quant_format=__import__(
            "onnxruntime.quantization", fromlist=["QuantFormat"]
        ).QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )
    print(f"QDQ model saved: {qdq_path}")


def build(onnx_path: Path, output: Path, fp16: bool, int8: bool):
    builder = trt.Builder(LOG)
    network = builder.create_network()  # TRT >= 10: explicit batch is implicit
    parser = trt.OnnxParser(network, LOG)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("ONNX parse failed")
    print(f"parsed {onnx_path}: {network.num_inputs} in / {network.num_outputs} out")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # fallback for non-quantized ops

    print("building engine (this takes a few minutes)...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise SystemExit("engine build failed")
    output.write_bytes(engine)
    print(f"saved {output} ({output.stat().st_size / 1024**2:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--calib-dir", type=Path, help="image dir for int8 calibration")
    ap.add_argument("--calib-num", type=int, default=64)
    ap.add_argument("--base-checkpoint", default="google/siglip2-base-patch16-224",
                    help="HF checkpoint for the image processor (int8 only)")
    args = ap.parse_args()

    if args.precision == "int8":
        if args.calib_dir is None:
            raise SystemExit("--precision int8 requires --calib-dir")
        qdq_path = args.output.with_suffix(".qdq.onnx")
        quantize_onnx_int8(
            args.onnx, qdq_path, args.calib_dir, args.calib_num,
            args.base_checkpoint,
        )
        build(qdq_path, args.output, fp16=True, int8=True)
    else:
        build(args.onnx, args.output, fp16=True, int8=False)


if __name__ == "__main__":
    main()
