"""Inference helper for SigLIP2 multi-attribute checkpoints produced by
`train.py` / `quantize.py` in this directory.

Self-contained copy of the equivalent helper in the dataset-collector server
— this directory must not depend on dataset-collector code. Only needs
`torch` + `transformers` + `Pillow` (the `closure` conda env has all three).

Understands checkpoint naming from both pipelines:
  - model dirs:    best.pt, last.pt
  - quantize dirs: fp16.pt, int8-dynamic.pt
and accepts a direct path to a .pt file as well as a directory.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

CHECKPOINT_NAMES = ("best.pt", "last.pt", "fp16.pt", "int8-dynamic.pt",
                    "int8-torchao.pt")


class Siglip2AttributeModel(nn.Module):
    """SigLIP2 vision tower + one linear head per attribute group.

    `groups` is the list of {"name", "options"} dicts saved inside the
    checkpoint by the training script — defines the heads and the order of
    classes each head's logits correspond to.
    """

    def __init__(self, checkpoint: str, groups: list[dict]):
        super().__init__()
        backbone = AutoModel.from_pretrained(checkpoint).vision_model
        hidden_size = backbone.config.hidden_size
        self.backbone = backbone
        self.heads = nn.ModuleDict(
            {g["name"].lower(): nn.Linear(hidden_size, len(g["options"])) for g in groups}
        )

    def forward(self, pixel_values):
        pooled = self.backbone(pixel_values=pixel_values).pooler_output
        return {key: head(pooled) for key, head in self.heads.items()}


def find_checkpoint_file(model_path: Path) -> Path | None:
    if model_path.is_file():
        return model_path if model_path.suffix == ".pt" else None
    for name in CHECKPOINT_NAMES:
        candidate = model_path / name
        if candidate.is_file():
            return candidate
    return None


def load_predictor(model_path: Path):
    """Loads a trained checkpoint and returns (model, processor, groups, device).

    `model_path` may be a directory holding a checkpoint file or a direct
    path to a .pt file.

    Handles quantization metadata written by quantize.py: "fp16" casts
    the model before loading, "int8-dynamic" rebuilds the quantized module
    structure (packed Linear params) and runs on CPU since those kernels have
    no GPU backend. Checkpoints without a "quantization" field are float32.
    """
    ckpt_path = find_checkpoint_file(model_path)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint ({'/'.join(CHECKPOINT_NAMES)}) found in {model_path}"
        )

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        # torchao checkpoints serialize tensor subclasses that the default
        # weights_only=True unpickler rejects — retry with it off.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_checkpoint = ckpt.get("checkpoint")
    groups = ckpt.get("groups") or []
    if not base_checkpoint or not groups:
        raise ValueError(f"Checkpoint {ckpt_path} is missing 'checkpoint'/'groups' metadata")
    quantization = ckpt.get("quantization")

    model = Siglip2AttributeModel(base_checkpoint, groups)
    if quantization == "int8-dynamic":
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        device = torch.device("cpu")
    elif quantization == "int8-torchao":
        # Rebuild the quantized module structure (int8 Linear subclasses)
        # before loading weights; runs real int8 matmul kernels on CUDA.
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig,
            quantize_,
        )
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if quantization == "fp16":
            model = model.half()
        elif quantization not in (None, "fp32"):
            raise ValueError(f"Unknown quantization mode '{quantization}' in {ckpt_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(base_checkpoint)
    return model, processor, groups, device


def predict_values(image_path: Path, predictor, attr_groups: list[dict]) -> list[int]:
    """Runs the model and returns a one-hot vector matching the template's
    current attribute layout (`attr_groups` — name/options/indices dicts).

    Predictions are mapped back from the checkpoint's own group option order
    into the requested option order/indices by option name, so minor
    reordering (not removal/renaming) of options after training still works.
    Groups absent from the checkpoint's heads are left unset.
    """
    model, processor, ckpt_groups, device = predictor

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(
        device=device, dtype=next(model.parameters()).dtype
    )

    with torch.no_grad():
        logits = model(pixel_values)

    vector_length = max((max(g.get("indices", [0]) or [0]) for g in attr_groups), default=-1) + 1
    values = [0] * max(vector_length, 1)

    ckpt_groups_by_key = {g["name"].lower(): g for g in ckpt_groups}

    for group in attr_groups:
        key = group["name"].lower()
        if key not in logits:
            continue
        ckpt_group = ckpt_groups_by_key.get(key)
        if not ckpt_group:
            continue

        pred_idx = int(logits[key].argmax(dim=-1).item())
        if pred_idx >= len(ckpt_group["options"]):
            continue
        option_name = str(ckpt_group["options"][pred_idx]).lower()

        config_options = [str(o).lower() for o in group.get("options", [])]
        if option_name not in config_options:
            continue
        local_idx = config_options.index(option_name)

        indices = group.get("indices") or []
        if local_idx < len(indices):
            values[indices[local_idx]] = 1

    return values
