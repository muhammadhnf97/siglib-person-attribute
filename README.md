# SigLIP2 multi-attribute training

Fine-tunes a SigLIP2 vision backbone with one linear head per attribute
group, for any `siglip2/*` template in the collector (e.g.
`siglip2_person_attribute`). See
`train_siglip2_attributes.py` and `quantize_siglip2.py` for full `--help`.

Run everything inside the `closure` conda env. Checkpoints are ~350MB
(float32) — point `--output-dir` at `$HOME`, not `disk-256`, which runs out
of space easily.

`--config` tells the script which attribute groups to build heads for (names,
option lists, how many classes each head has) — it's required, both scripts
fail fast without it. The collector's `/datasets/{name}/export` now bundles
the exact `config.yaml` that produced the export's `metadata.jsonl` labels
right inside the export dir, so point `--config` at `<export-dir>/config.yaml`
rather than back into `dataset-collector/server/template/...` — that way
training always matches the config an export was actually made with, even if
the live template gets edited later.

## Train

```bash
conda run -n closure python /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/train_siglip2_attributes.py \
  --export-dir /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/person-attribute-2 \
  --config /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/person-attribute-2/config.yaml \
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
  --checkpoint /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/siglip2-runs/person-attribute-2/best.pt \
  --export-dir /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/person-attribute-2 \
  --config /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/person-attribute-2/config.yaml \
  --output-dir /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/siglip2-runs/person-attribute-2-quant \
  --mode fp16
```

- `--mode fp16`: 2x smaller, same GPU, effectively no accuracy cost. Default
  choice when serving stays on this machine's GPU.
- `--mode int8-dynamic`: ~3.5x smaller, CPU-only (PyTorch's dynamic-quant
  kernels have no GPU backend), small accuracy cost — only worth it if
  inference is moving off the GPU.

## Benchmark (latency / stress test)

Times the same code path `/prelabel` runs (load, preprocess, forward, argmax)
— works on float32, fp16, and int8-dynamic checkpoints. `--images-dir` is just
a source of real images to feed the model (labels are ignored; it cycles if
the folder is smaller than `--num-images` — point it at a real batch for a
representative run). Prints per-image latency stats, per-chunk throughput so
you can spot degradation under sustained load, peak VRAM, and a wall-clock
projection for `--project` images (default 10k).

```bash
conda run -n closure python /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/benchmark_siglip2.py \
  --model-dir /home/me/disk-256/00_RND_MODELS/paddlepaddle/dataset-collector/server/template/siglip2/siglip2_person_attribute/model/person-attribute-2-fp16 \
  --images-dir /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/person-attribute-2/test \
  --num-images 300
```

Measured on this machine (RTX 3080): fp16 ~11ms/img ≈ 89 img/s (~2min per
10k images); int8-dynamic ~110ms/img ≈ 9 img/s (~18min per 10k) since it
runs on CPU.

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
