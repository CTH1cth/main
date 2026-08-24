# Per-query maximum valid-background similarity diagnostic

This package implements the held-out CAMO-Test diagnostic defined in
`configs/gbsp_teaser_max_similarity.yaml`. It reuses frozen formal DINO and
GBSP-r8 caches; GT is used only to create diagnostic labels and an ideal valid
background reference set.

It does not implement KNN, train a model, change GBSP, or evaluate any dataset
other than CAMO-Test.
