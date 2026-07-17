# EGSA-R1 消融计划

## 主比较

- B1：当前固定 ECST clean Long45。
- R1：固定 ECST + EGSA residual arbitration。
- N0：错误负对照，允许 segmentation loss 更新 gate。
- P0：仅 global prior，不学习 residual。

B1/R1 至少使用三个随机种子，主比较固定 epoch41，同时完整报告
CAMO/COD10K/NC4K 的 S-measure、weighted F-measure、E-measure 和 MAE。

## 单种子机制消融

依次移除 DINO margin、temporal statistics、student uncertainty、
disagreement modulation、cross-view consistency、bounded residual、
prior shrinkage、mass regularization、smoothness，以及用训练 teacher 替代
独立 utility evaluator。

## 机制验收

- utility-validation AUC 相对 global prior 至少提升 0.03；
- balanced accuracy > 0.55；
- utility valid pixel ratio >= 1%；
- residual 非全零且 saturation <= 80%；
- disagreement pixel 同时出现 teacher gate 上调与下调；
- shuffled utility target 明显弱于真实 target。

## 性能验收

三个种子的平均 `Q_avg` 至少提升 0.002，至少两个数据集提升，任一数据集平均
下降不超过 0.002。若失败，固定 ECST 继续作为正式主线，EGSA 保留为研究分支。
