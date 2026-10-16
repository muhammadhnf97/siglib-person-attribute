# cluster_people.py — distinct-person estimate for a crop folder

Answers the question a crop count can't: **how many different people are
actually in this folder?** Supplementary analysis — not part of the
train/deploy artifact path.

```bash
conda run -n closure python cluster_people.py clustering-img \
    --checkpoint siglip2-runs/person-attribute-5/best.pt
```

## How it works — three phases

```
images ──► [trained SigLIP2 backbone] ──► L2-normalized 768-d vectors
       ──► cosine distance matrix ──► clustering ──► reports
```

1. **Embed** — loads the full checkpoint via `siglip2_infer`
   (`best.pt`/`last.pt` or a run dir). `backbone(...).pooler_output` gives a
   768-d embedding per crop; the heads *also* run to produce per-image
   predicted attributes used in the reports. One pass feeds both.
   Garment-tuned features are the point — clothing is the dominant identity
   signal in same-session footage.
2. **Cluster** — default `agglo`: sklearn `AgglomerativeClustering`,
   `linkage="average"`, `metric="precomputed"`, `distance_threshold=t` —
   merges bottom-up until nothing is closer than `t`, so **k is
   estimated, not given**.
3. **Report** — JSON + text + montages + PDF (see below).

## Arguments

| Flag | Default | Effect |
|---|---|---|
| `dir` | required | folder of crops, searched **recursively** |
| `--checkpoint` | newest `siglip2-runs/*/best.pt` | run dir or `.pt` file |
| `--method` | `agglo` | `agglo` or `kmeans` |
| `--threshold` | auto | pin the cosine distance threshold |
| `--k-max` | 200 | kmeans sweep upper bound |
| `--batch-size` | 32 | embedding batch size |
| `--out` | `<dir>/clusters.json` | JSON output path |
| `--link-dir` | off | symlink crops into `cluster-00001/...` dirs for eyeball review |
| `--montage-dir` | `<dir>/clusters_montage` | one grid JPEG per cluster |
| `--no-montage` | off | skip montages |
| `--pdf` | `<dir>/clusters_report.pdf` | PDF path |
| `--no-pdf` | off | skip PDF |

## Auto threshold picking

Without `--threshold`, it sweeps 0.05→0.55 and prints the
count-vs-threshold curve, then picks the **curve knee** (point furthest
below the chord joining the ends). If the knee sits inside a plateau it
snaps to the plateau's smallest `t`. The collapsing tail (everything
merging into blobs) counts as a "plateau" but is explicitly not trusted.

`--method kmeans` sweeps k=2..k-max scoring `silhouette_score` — kept for
comparison; k-means assumes even spherical clusters, which people-counts
aren't. Prefer agglo.

## Outputs

| Artifact | Contents |
|---|---|
| `clusters.json` | `{filename: cluster_id}` + per-cluster summary (size, coherence, attribute distribution, zones) |
| `clusters_report.txt` | per-cluster dominant profile + `mixed:` lines where members disagree — disagreement on an attribute usually means two people merged |
| `clusters_montage/` | one JPEG per cluster — filename carries `n` (size) + `q` (coherence, mean intra-cluster similarity) + dominant attributes. **Review lowest-q first** — they're the likely merged-stranger clusters |
| `clusters_report.pdf` | styled report: cover stats, zone table, cluster cards with thumbnails |
| `.people_embeddings.npz` | derived cache (`x`, `files`, `ckpt`, `preds`) — reruns are instant; invalidated when files/checkpoint change. Gitignored; safe to delete to force re-embed |

`--link-dir` writes symlinks to full-res originals when montage thumbnails
are too small to judge.

## Zones — dummy data, not real metadata

Each image gets 1–2 of `entrance`/`food_court`/`playground`/`shopping`
assigned **deterministically by filename hash** (~75% get one zone, ~25%
get a second). Zones appear in the JSON, PDF zone table, and per-cluster
cards — but they are **placeholders** for real camera/location metadata.
Replace `assign_zones()` when real metadata exists.

## Honest limitations

- **Identity ≈ clothing.** Same person in different clothes splits into
  multiple clusters; different people in identical uniforms merge. Visual
  review of montages is not optional.
- Back-view/occluded crops of the same person may split.
- Single-session footage only — not a re-ID model. If montages show merged
  strangers, the embedding source is the lever (e.g. a dedicated re-ID
  model), not the threshold.
- Interpretation guide: big coherent clusters = recurring people;
  `mixed:` attribute lines = suspect merges; many singletons = either truly
  many one-off people *or* threshold too strict.
