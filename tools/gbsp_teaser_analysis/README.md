# GBSP CAMO real-data teaser

This stage uses no KNN and no background dictionary on the left. It computes
all unique BG--BG and all FG--BG pairwise DINO cosine affinities, averages
unit-sum per-image histograms, and juxtaposes them with the frozen formal
Full-BC GBSP-r8 residual distribution.

Smoke test:

```bash
python -m tools.gbsp_teaser_analysis.run_camo \
  --config configs/gbsp_teaser_real.yaml \
  --out_dir ../workdir/27-gbsp_teaser_real/smoke8 \
  --max_samples 8 \
  --device cuda
```

Formal CAMO-Test stage (run only when full evaluation is explicitly allowed):

```bash
python -m tools.gbsp_teaser_analysis.run_camo \
  --config configs/gbsp_teaser_real.yaml \
  --out_dir ../workdir/27-gbsp_teaser_real/camo250 \
  --max_samples -1 \
  --device cuda
```
