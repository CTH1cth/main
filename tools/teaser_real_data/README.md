# GBSP real-data teaser

This pipeline imports the frozen production KNN8 scorer and reuses the frozen
GBSP-r8 core cache. It never extracts DINO features, changes a model, scans a
hyperparameter, or trains a network.

Small end-to-end smoke test:

```bash
python -m tools.teaser_real_data.run_teaser \
  --config configs/gbsp_real_teaser.yaml \
  --out_dir ../workdir/26-gbsp_real_teaser/smoke24 \
  --max_samples_per_dataset 8 \
  --device cpu
```

Formal held-out run (execute only when full evaluation is explicitly allowed):

```bash
python -m tools.teaser_real_data.run_teaser \
  --config configs/gbsp_real_teaser.yaml \
  --out_dir ../workdir/26-gbsp_real_teaser/formal \
  --max_samples_per_dataset -1 \
  --device cuda
```
