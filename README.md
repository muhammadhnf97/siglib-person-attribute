# SigLIP2 multi-attribute training

Fine-tunes a SigLIP2 vision backbone with one linear head per attribute
group, for any `siglip2/*` template in the collector (e.g.
`siglip2_person_attribute`). See
`train_siglip2_attributes.py` and `quantize_siglip2.py` for full `--help`.

Run everything inside the `closure` conda env. Checkpoints are ~350MB
(float32) — point `--output-dir` at `$HOME`, not `disk-256`, which runs out
of space easily.

## Train

```bash
conda run -n closure python /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/train_siglip2_attributes.py \
  --export-dir /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/person-attribute-2 \
  --config /home/me/disk-256/00_RND_MODELS/paddlepaddle/dataset-collector/server/template/siglip2/siglip2_person_attribute/config.yaml \
  --output-dir "$HOME/siglip2-runs/person-attribute-2" \
  --checkpoint google/siglip2-base-patch16-224 \
  --epochs 15 --batch-size 32 --balanced-loss
```

Defaults are tuned to avoid collapsing the shared backbone on small/imbalanced
datasets: `--lr 2e-5`, `--freeze-backbone-epochs 2` (head warm-up before the
backbone unfreezes), and `--balanced-loss` caps any single class's weight at
`--max-class-weight` (default `10x`) so a near-absent class (e.g. 4 "kid"
images out of 546) can't destabilize training. If `best_macro_acc` lands
suspiciously close to the dataset's majority-class ratio, the model likely
collapsed — check per-head accuracy, not just the macro average.

Outputs `best.pt`, `last.pt`, and `labels.json` in `--output-dir`.

## Quantize (optional)

Always measures baseline vs. converted accuracy on the same test split before
you decide whether to ship it — never trust a quantized model without that
diff.

```bash
conda run -n closure python /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/quantize_siglip2.py \
  --checkpoint "$HOME/siglip2-runs/person-attribute-2/best.pt" \
  --export-dir /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/person-attribute-2 \
  --config /home/me/disk-256/00_RND_MODELS/paddlepaddle/dataset-collector/server/template/siglip2/siglip2_person_attribute/config.yaml \
  --output-dir "$HOME/siglip2-runs/person-attribute-2-quant" \
  --mode fp16
```

- `--mode fp16`: 2x smaller, same GPU, effectively no accuracy cost. Default
  choice when serving stays on this machine's GPU.
- `--mode int8-dynamic`: ~3.5x smaller, CPU-only (PyTorch's dynamic-quant
  kernels have no GPU backend), small accuracy cost — only worth it if
  inference is moving off the GPU.

## Deploying to the collector for prelabeling

1. Copy `best.pt` (or the quantized `fp16.pt`/`int8-dynamic.pt`, renamed to
   `best.pt`) into
   `dataset-collector/server/template/siglip2/<template>/model/<name>/`.
2. Point that template's `config.yaml` `pre-label-model` at `model/<name>`,
   or pass `"model": "model/<name>"` in the `/datasets/{name}/prelabel`
   request body.
3. `disk-256` fills up fast — `df -h /home/me/disk-256` before copying a new
   checkpoint there, and `md5sum` the source/destination after copying to
   catch a truncated copy (silent on a full disk) before it causes a 500 at
   prelabel time.
