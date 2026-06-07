# MY-baseline Clean Cached-DINO EMA Baseline

## 改动了什么

本目录实现了一个干净、直线型的 cached-DINO EMA UCOD baseline。代码只放在：

```text
/home/dell01/CTH/MY-baseline/main/
```

运行数据入口和产物目录放在：

```text
/home/dell01/CTH/MY-baseline/datasets/
/home/dell01/CTH/MY-baseline/workdir/
```

结构如下：

```text
main/
  configs/
    dinov1_s8.py
    dinov1_b8.py
    dinov2_b14.py
  train.py
  eval.py
  model.py
  common/
    cache_pseudo.py
    cache_features.py
    dataset.py
    metrics.py
    utils.py
  README.md
```

`datasets/COD` 是软链接，指向本仓库已有数据集：

```text
/home/dell01/CTH/dataset/COD
```

## 保存位置怎么改

三个配置文件都显式写了路径：

```python
DATA_ROOT = "/home/dell01/CTH/MY-baseline/datasets/COD"
CACHE_ROOT = "/home/dell01/CTH/MY-baseline/datasets/cache"
WORK_ROOT = "/home/dell01/CTH/MY-baseline/workdir"
EXP_NAME = "dinov1_s8_clean"
```

调整规则：

- 改数据集入口：改 `DATA_ROOT`。
- 改特征缓存和伪标签缓存位置：改 `CACHE_ROOT`。
- 改训练/eval 输出根目录：改 `WORK_ROOT`。
- 改单次实验输出子目录：改 `EXP_NAME`。
- 选择 DINO 版本：运行对应 config。

## Cache 目录

伪标签缓存：

```text
{CACHE_ROOT}/pseudo_label_cache/{backbone_key}/manifest_train.jsonl
{CACHE_ROOT}/pseudo_label_cache/{backbone_key}/TR-CAMO/*.pt
{CACHE_ROOT}/pseudo_label_cache/{backbone_key}/TR-COD10K/*.pt
```

特征缓存：

```text
{CACHE_ROOT}/features_cache/{backbone_key}/manifest_train.jsonl
{CACHE_ROOT}/features_cache/{backbone_key}/manifest_val.jsonl
{CACHE_ROOT}/features_cache/{backbone_key}/manifest_test.jsonl
{CACHE_ROOT}/features_cache/{backbone_key}/{split}/{dataset}/*.pt
```

每个 `.pt` 保存 dict，不保存裸 tensor。

缓存有两种生成方式：

- 手动生成：进入 `main/common/` 后直接运行 `cache_pseudo.py` 和 `cache_features.py`。
- 自动生成：启动 `train.py` 或 `eval.py` 时，程序会先检查当前 config 对应的缓存；缺 manifest、缺样本或缺 `.pt` 文件时会自动生成当前 backbone/split 的缓存。

如果已有 manifest 中出现不属于当前数据集的 `dataset + stem`，或 manifest 中 shape 字段非法，程序会直接报错，避免混用错误缓存。

## 命令

所有命令建议先进入 `cth` 环境：

```bash
source /home/dell01/anaconda3/etc/profile.d/conda.sh && conda activate cth
cd /home/dell01/CTH/MY-baseline/main
```

生成 fixed pseudo cache：

```bash
cd /home/dell01/CTH/MY-baseline/main/common
python cache_pseudo.py --config ../configs/dinov1_s8.py --overwrite
python cache_pseudo.py --config ../configs/dinov1_b8.py --overwrite
python cache_pseudo.py --config ../configs/dinov2_b14.py --overwrite
```

生成 feature cache：

```bash
cd /home/dell01/CTH/MY-baseline/main/common
python cache_features.py --config ../configs/dinov1_s8.py --split train --overwrite
python cache_features.py --config ../configs/dinov1_s8.py --split val --overwrite
python cache_features.py --config ../configs/dinov1_s8.py --split test --overwrite

python cache_features.py --config ../configs/dinov1_b8.py --split train --overwrite
python cache_features.py --config ../configs/dinov1_b8.py --split val --overwrite
python cache_features.py --config ../configs/dinov1_b8.py --split test --overwrite

python cache_features.py --config ../configs/dinov2_b14.py --split train --overwrite
python cache_features.py --config ../configs/dinov2_b14.py --split val --overwrite
python cache_features.py --config ../configs/dinov2_b14.py --split test --overwrite
```

训练：

```bash
python train.py --config configs/dinov1_s8.py
python train.py --config configs/dinov1_b8.py
python train.py --config configs/dinov2_b14.py
```

测试 best checkpoint：

```bash
python eval.py \
  --config configs/dinov1_s8.py \
  --ckpt /home/dell01/CTH/MY-baseline/workdir/dinov1_s8_clean/train/ckpt/best.pth
```

## 输出目录

训练输出：

```text
{WORK_ROOT}/{EXP_NAME}/train/
  train.log
  config.yaml
  ckpt/
    epoch_005.pth
    epoch_010.pth
    epoch_015.pth
    epoch_020.pth
    epoch_025.pth
    best.pth
```

eval 输出：

```text
{WORK_ROOT}/{EXP_NAME}/eval/
  eval.log
  config.yaml
  pred/
    TE-CAMO/*.png
    TE-COD10K/*.png
    CHAMELEON/*.png
    NC4K/*.png
```

预测图格式为 binary `0/255 png`。

`config.yaml` 是运行时解析后的配置快照，作用等价于之前的 `resolved_config.json`，只是采用更常见的 yaml 命名和格式。它不是新的配置入口，而是记录当次运行实际使用的参数，便于复现。

eval 指标只写入 `eval.log`，训练阶段 best 验证指标只写入 `train.log`，不额外生成 `metrics.json`、`metrics.txt` 或 `best_metrics.json`。

## 关键实现约束

- DINO 只在 `common/cache_pseudo.py` 和 `common/cache_features.py` 中加载。
- `train.py` 的正式训练循环不加载 DINO；若启动前发现缓存缺失，会先调用 cache 脚本生成缓存。
- 训练集不读取 GT。
- 验证和测试只用 student，不使用 EMA teacher。
- loss 固定在 `68x68`。
- fixed pseudo 插值到 `68x68` 使用 bilinear，插值后不二值化。
- teacher pseudo 使用 `sigmoid > 0.5` 二值化。
- scheduler 使用 `StepLR(step_size=25, gamma=0.95)`，每 iteration 调用。
- 第 21 轮第一个 batch forward 前会按 UCOD-DPL 源码重建 optimizer/scheduler 并重置 `global_step=0`，不重置 teacher，避免最后 5 轮 teacher-only / finetune 阶段学习率衰减到接近 0。
- 不引入 APM、判别器、Look-Twice、SAM、复杂 decoder、runner、factory、registry。

## 最终测试结果

本次只写代码，没有生成 cache、没有启动训练、没有启动测试。

## 与已有基线或论文结果的对比结论

当前还没有运行实验结果，因此不能给出数值对比。代码结构上只保留任务书指定的 Clean Cached-DINO EMA baseline 逻辑，不复用旧实验产物。
