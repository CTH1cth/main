# dinov1_s8_despl_dagp_uncgate_ndr_lrfloor_2e5 模型说明

本文档说明当前最优配置：

```text
configs/dinov1_s8_despl_dagp_uncgate_ndr_lrfloor_2e5.py
```

对应实验名：

```text
dinov1_s8_despl_dagp_uncgate_ndr_lrfloor_2e5
```

这份说明面向人工理解，重点解释在原始 DINO cached baseline 之外新增或改变的设计：DESPL-only pseudo、DAGP-Safe graph propagation、uncertainty output gate、NDR native detail residual branch、辅助损失、teacher-student 训练、LR floor 和 reset 策略。原始 baseline 中已经存在的 dataset、metric、cache 框架只做必要背景说明。

注意：当前代码中还存在后续实验分支，例如 TADR、gate025075、beta005、multi-view proto 等，但本配置没有启用这些分支。本文只描述该配置实际走到的模型和训练路径。

## 1. 总览

这个版本可以理解为：

```text
Cached DINOv1-s8 feature [B,384,37,37]
    -> DAGP-Safe coarse head, still in 37x37 feature space
    -> coarse logits [B,1,37,37]
    -> upsample to 68x68
    -> NDR detail residual branch, using RGB_68 + Sobel_68 + coarse probability
    -> final logits [B,1,68,68]
    -> BCE loss against DESPL/teacher mixed target [B,1,68,68]
```

核心原则是分辨率分工：

- DAGP-Safe 在 raw cached DINO feature 上工作，即 `37x37`。
- NDR 在训练 loss 空间工作，即 `68x68`。
- 训练损失、pseudo、teacher target 都对齐到 `LOSS_SIZE=68`。
- 评估时输出 logits 再按原有 eval 逻辑 resize 到 GT 尺寸计算指标。

相对原 baseline，主要新增了四层保护：

1. DAGP graph branch 前 6 个 epoch 关闭，epoch 7 到 15 线性 ramp。
2. DAGP graph residual 被 base uncertainty gate 限制，只在 base 预测不确定处更容易起作用。
3. NDR residual branch 前 6 个 epoch 关闭，epoch 7 到 15 线性 ramp。
4. NDR residual 只作为 68x68 residual refinement，不直接替代 coarse logits。

## 2. 配置继承关系

配置继承链为：

```text
dinov1_s8.py
    -> dinov1_s8_despl_only_teacher_cache.py
        -> dinov1_s8_despl_only_lr_floor_2e5.py
            -> dinov1_s8_despl_only_dagp_safe_lrfloor_2e5.py
                -> dinov1_s8_despl_only_dagp_safe_uncgate_lrfloor_2e5.py
                    -> dinov1_s8_despl_dagp_uncgate_ndr_lrfloor_2e5.py
```

最终生效的关键字段如下：

```text
BACKBONE_KEY = dinov1-s8
HEAD_TYPE = dagp_safe
LOSS_SIZE = 68
MAX_EPOCH = 25
FINETUNE_RESET_EPOCH = 21
LR_POLICY = step_floor
USE_LR_FLOOR = True
LR_FLOOR = 2e-5
EMA_WEIGHT = 0.99
THRESHOLD = 0.5

P_INIT_MODE = despl_only
USE_DESPL_PSEUDO = True
USE_DESPL_LIGHT_CACHE = True
P_INIT_DESPL_WEIGHT = 1.0
P_INIT_FIXED_WEIGHT = 0.0

USE_DAGP_SAFE_HEAD = True
USE_NDR_BRANCH = True
GKD_MODE = off
```

训练和评测数据集沿用原 baseline：

```text
TRAIN_DATASETS = ["TR-CAMO", "TR-COD10K"]
VAL_DATASETS = ["TE-CAMO"]
TEST_DATASETS = ["CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K"]
```

## 3. 原 baseline 保留的部分

本版本没有修改这些基础部分：

- 数据集组织方式。
- cached DINO feature 读取方式。
- train/val/test split。
- validation metric 和 eval metric。
- prediction resize 到 GT 后计算指标。
- checkpoint 保存的基本字段。
- teacher-student 框架。
- EMA teacher 更新方式。

原 baseline 的主干逻辑可以抽象为：

```text
cached DINO feature
    -> segmentation head
    -> logits
    -> pseudo/teacher target
    -> BCE loss
```

本版本主要是在 segmentation head 和训练策略上增强，而不是重新设计数据集、metric 或 DINO cache。

## 4. 数据流

### 4.1 DINO cached feature

使用 `BACKBONE_KEY="dinov1-s8"`。基础配置中 DINO feature input size 为 `296`，patch size 为 `8`，因此 cached feature 空间尺寸是：

```text
296 / 8 = 37
```

训练 batch 中的 feature 形状为：

```text
feature: [B,384,37,37]
```

这个版本最重要的点是：`HEAD_TYPE="dagp_safe"` 会让 `train.py` 的 `make_model_input()` 直接返回 raw cached feature，不会在 head 之前把 feature 插值到 `68x68`。

也就是说：

```text
DAGP 输入 = [B,384,37,37]
DAGP 输出 coarse logits = [B,1,37,37]
NDR 和 loss 之前才进入 68x68 空间
```

### 4.2 train sample 返回字段

当前配置下，`CachedTrainDataset` 每个 sample 至少包含：

```text
feature
pseudo
dataset
stem
image_path
image_68
pseudo_fixed
pseudo_despl
p_init_area
p_fixed_area
p_despl_area
use_fixed_in_pseudo
fixed_used_for_training
```

其中：

- `dataset` 和 `stem` 可以组成稳定 sample id：`(dataset, stem)`。
- `image_path` 是原图路径。
- `image_68` 是 NDR 需要的 RGB 输入。
- `pseudo` 是训练用初始 pseudo。
- `pseudo_despl` 是 DESPL pseudo。
- `pseudo_fixed` 是原 fixed pseudo，当前不参与训练 target，只用于日志或质量对比。

`image_68` 的生成方式：

```text
PIL RGB image
    -> resize to [68,68] with bilinear
    -> float32
    -> CHW layout
    -> range [0,1]
```

没有随机增强，也没有 ImageNet mean/std normalization。这样做是为了让 `image_68` 与 cached feature、pseudo、loss size 保持确定对齐。

### 4.3 pseudo 读取

当前配置为：

```text
USE_DESPL_LIGHT_CACHE = True
P_INIT_MODE = despl_only
P_INIT_DESPL_WEIGHT = 1.0
P_INIT_FIXED_WEIGHT = 0.0
```

因此训练初始 pseudo 使用 DESPL light cache 里的 `p_despl_68`。`p_fixed_68` 会被读取，但不会进入训练 pseudo 混合。

最终训练 pseudo 形状：

```text
pseudo: [B,1,68,68]
```

### 4.4 DESPL pseudo 是怎么生成的

当前训练并不是在线生成 DESPL pseudo，而是读取已经离线生成好的 cache。要理解训练用的 `pseudo`，需要区分三层东西：

```text
fixed pseudo cache
    -> nper_pseudo_bank 里的 p_despl / p_fixed / p_gcm / p_init
        -> despl_blend_pseudo_cache 里的 p_despl_68 / p_fixed_68 / tensor
            -> 当前训练实际使用 p_despl_68
```

当前配置中：

```text
USE_DESPL_PSEUDO = True
USE_DESPL_LIGHT_CACHE = True
DESPL_LIGHT_CACHE_ROOT = ../datasets/cache/despl_blend_pseudo_cache
DESPL_PSEUDO_SOURCE = nper_pseudo_bank
P_INIT_MODE = despl_only
P_INIT_DESPL_WEIGHT = 1.0
P_INIT_FIXED_WEIGHT = 0.0
```

所以训练时真实使用的是：

```text
pseudo = p_despl_68
```

不是 `0.8*p_despl + 0.2*p_fixed`，也不是 source bank 里的 `p_init` quality fusion。即使某些历史 light cache manifest 中的 `formula` 字段显示旧公式，`CachedTrainDataset` 在 `P_INIT_MODE="despl_only"` 时会强制把 `pseudo` 设为 `pseudo_despl`。这就是日志里：

```text
p_init_formula = p_despl
fixed_used_for_training = False
use_fixed_in_pseudo = False
```

的含义。

#### 4.4.1 第一层：fixed pseudo cache

DESPL 需要一个 coarse foreground 参考，主要用于解决 spectral mask 的前景/背景符号翻转问题。这个参考来自 fixed pseudo cache。

fixed pseudo 由 `common/cache_pseudo.py` 生成，不使用训练 GT。它的流程是：

```text
原图
    -> resize 到 DINO pseudo_input_size=224
    -> ImageNet normalize
    -> DINO forward，取最后一层 attention
    -> hook 最后一层 attention key projection
    -> 根据 class-token attention 找低注意力参考 patch
    -> 用 key feature cosine similarity 扩展背景区域
    -> foreground pseudo = 1 - background mask
    -> 小连通域后处理
    -> 保存 fixed pseudo
```

更具体地说，`compute_img_bkg_seg()` 复用 UCOD-DPL 风格的背景种子思想：

1. 读取 DINO 最后一层 attention，取 class token 到 patch token 的 attention。
2. 找到整体 attention 最低的 patch，作为背景参考 patch。
3. 取 DINO key feature，计算参考 patch 与所有 patch 的 cosine similarity。
4. similarity 高于 `bkg_th` 的 patch 被认为是背景。
5. fixed foreground mask 是背景 mask 的反：

```text
p_fixed = 1 - bkg_mask
```

对当前 `dinov1-s8` 基础配置：

```text
pseudo_input_size = 224
patch_size = 8
fixed pseudo 初始 patch grid = 28x28
```

fixed pseudo 的作用不是当前训练的最终监督，而是给 DESPL 生成过程提供一个方向参考。

#### 4.4.2 第二层：nper_pseudo_bank 中的 DESPL

真正的 DESPL mask 由 `nper/pseudo_bank.py` 中的 `make_despl()` 和 `spectral_mask()` 生成。入口一般是 `common/cache_nper_pseudo_bank.py`。

生成时每个训练样本会读取：

```text
image_path
fixed pseudo
cached DINO feature
```

这里同样不读取训练 GT。

DESPL 的核心思想是：在 DINO feature + RGB color 构成的图上做 spectral partition，把图切成前景/背景两部分，再用 fixed pseudo 解决正负方向。

当前 NPER/DESPL 生成配置中的关键参数来自 `configs/nper_ucod_v1.py`：

```text
DESPL_GRID = 28
DESPL_LAMBDA_COLOR = 0.1
DESPL_COLOR_SIGMA = 0.1
DESPL_EIG_BINS = 50
DESPL_LOWCONF_ALPHA = 0.1
DESPL_AUGS = ["identity", "hflip", "vflip", "rot180"]
```

DESPL 生成分成以下步骤。

第一步，把 DINO feature、RGB image、fixed pseudo 对齐到同一个 spectral grid：

```text
cached DINO feature [384,37,37]
    -> bilinear resize to [384,28,28]

RGB image
    -> bicubic resize to [3,28,28]

fixed pseudo
    -> bilinear resize to [1,28,28]
```

第二步，把每个 grid cell 当成图节点：

```text
N = 28 * 28 = 784
feature node: [N,384]
RGB node:     [N,3]
```

DINO feature 先做 L2 normalize，然后计算语义 affinity：

```text
affinity_sem(i,j) = (cosine(feature_i, feature_j) + 1) / 2
```

RGB color affinity 用高斯核：

```text
dist2(i,j) = ||rgb_i - rgb_j||^2
affinity_color(i,j) = exp(-dist2(i,j) / (2 * sigma^2))
```

最终图 affinity：

```text
affinity = affinity_sem + lambda_color * affinity_color
```

当前：

```text
lambda_color = 0.1
sigma = 0.1
```

对角线会被置 0，避免节点自己连接自己；最后 affinity 会做对称化并 clamp 到非负。

第三步，构造 normalized graph Laplacian：

```text
degree_i = sum_j affinity(i,j)
L = I - D^(-1/2) * affinity * D^(-1/2)
```

然后做特征分解：

```text
eigvals, eigvecs = torch.linalg.eigh(L)
```

代码取前两个 eigenvector 作为候选分割向量。每个向量先 min-max normalize 到 `[0,1]`，再用 histogram entropy 评估“分割是否更清晰”：

```text
hist = histogram(vec, bins=50, range=[0,1])
entropy = -sum(p * log(p))
```

entropy 更低的向量被选为 main vector，另一个作为 auxiliary vector：

```text
main_vec = lower_entropy_eigenvector
aux_vec = the_other_eigenvector
```

第四步，对 main vector 做 Otsu threshold：

```text
threshold = otsu(main_vec)
mask = main_vec > threshold
```

这一步给出初始二值 spectral partition。

第五步，处理 threshold 附近的低置信区域。低置信区域定义为：

```text
low_conf = abs(main_vec - threshold)
           <= DESPL_LOWCONF_ALPHA * (main_vec.max - main_vec.min)
```

当前：

```text
DESPL_LOWCONF_ALPHA = 0.1
```

这些 threshold 附近的点不完全相信 main vector，而用 auxiliary vector 改写：

```text
mask[low_conf] = aux_vec[low_conf] > 0.5
```

这一步的目的，是减少 Otsu threshold 边缘附近的随机抖动。

第六步，用 fixed pseudo 解决 spectral mask 的前景/背景符号问题。spectral partition 本身不知道哪一侧是前景，可能得到前景，也可能得到背景的反。代码比较：

```text
IoU(mask, fixed_grid)
IoU(~mask, fixed_grid)
```

如果反相 mask 与 fixed pseudo 更一致，就翻转：

```text
if IoU(~mask, fixed_grid) > IoU(mask, fixed_grid):
    mask = ~mask
```

所以 fixed pseudo 在这里主要是方向锚点，不是最终训练监督。

第七步，做多视角增强投票。当前 DESPL 增强包括：

```text
identity
hflip
vflip
rot180
```

每个增强视角都独立执行 spectral partition，然后把 mask 反变换回原方向，最后平均：

```text
p_despl_grid = mean(mask_identity, mask_hflip, mask_vflip, mask_rot180)
p_despl_grid = clamp(p_despl_grid, 0, 1)
```

因此 `p_despl_grid` 是一个 soft mask，不一定只有 0 和 1。它的空间尺寸是：

```text
p_despl_grid: [1,28,28]
```

第八步，`nper_pseudo_bank` 会把 `p_despl_grid` resize 到 NPER pseudo bank 的 loss size。当前已生成的 source bank 中，样本 payload 形状是：

```text
p_fixed: [1,352,352]
p_despl: [1,352,352]
p_gcm:   [1,352,352]
p_init:  [1,352,352]
```

其中：

- `p_fixed` 来自 fixed pseudo。
- `p_despl` 来自上面的 DESPL spectral mask。
- `p_gcm` 是另一个基于 DINO attention 的候选 mask。
- `p_init` 是 NPER bank 自己的 quality fusion 结果。
- `anchor_fg`、`anchor_bg`、`pixel_weight` 是 NPER 训练相关的可靠性辅助字段。

当前 NDR 主实验不会直接使用 `p_gcm`、`p_init`、anchor 或 pixel weight。

#### 4.4.3 第三层：DESPL light cache

训练不会每次都打开完整 `nper_pseudo_bank` source payload，而是通过 `common/cache_despl_blend_pseudo.py` 预先生成轻量 cache：

```text
datasets/cache/despl_blend_pseudo_cache/dinov1-s8/<dataset>/<stem>.pt
```

light cache 读取 source bank 的：

```text
p_fixed
p_despl
```

并保存：

```text
tensor
p_init
p_fixed_68
p_despl_68
shape
source_shape
source_cache_path
formula
p_init_area
p_fixed_area
p_despl_area
```

其中 `p_fixed_68` 和 `p_despl_68` 都被 resize 到当前训练的：

```text
LOSS_SIZE = 68
```

当前训练最关键的字段是：

```text
p_despl_68: [1,68,68]
```

`CachedTrainDataset` 的实际读取逻辑是：

```text
light_payload = load_despl_light_pseudo(...)
pseudo_fixed = light_payload["pseudo_fixed"]   # p_fixed_68
pseudo_despl = light_payload["pseudo_despl"]   # p_despl_68

if P_INIT_MODE == "despl_only":
    pseudo = pseudo_despl
    use_fixed_in_pseudo = False
else:
    pseudo = P_INIT_DESPL_WEIGHT * pseudo_despl
             + P_INIT_FIXED_WEIGHT * pseudo_fixed
```

因此在当前配置中：

```text
训练初始 pseudo = p_despl_68
```

`p_fixed_68` 仍然会随 batch 返回，主要用于日志、诊断和后续可能的质量分析；它不参与当前 loss target。

#### 4.4.4 DESPL 与 teacher target 的关系

DESPL 只定义训练早期的 fixed target 来源。进入训练 loop 后，当前 batch 的 target 还会与 EMA teacher 融合：

```text
epoch < 21:
    mixed_target = fixed_weight * p_despl_68
                   + teacher_weight * teacher_binary

epoch >= 21:
    mixed_target = teacher_binary
```

所以完整监督链路是：

```text
offline DESPL spectral pseudo
    -> p_despl_68
    -> epoch1 作为 100% target
    -> epoch2-20 逐步让位给 EMA teacher binary
    -> epoch21-25 teacher-only
```

这里 DESPL 不参与 teacher 的 forward，也不直接参与 NDR/DAGP 结构；它只通过训练 target 影响 student 参数。

#### 4.4.5 DESPL 生成过程中的安全边界

这个 DESPL 生成链路有几个重要约束：

- 不读取训练 GT。
- 只在训练集样本上生成 cache。
- 所有 payload 都校验 `dataset`、`stem`、`backbone_key`，避免错配。
- DESPL spectral affinity 使用原始 cached DINO feature，不用当前 student/teacher。
- fixed pseudo 只用于符号翻转，不是当前训练目标。
- 当前训练实际目标使用 `p_despl_68`，不会使用 light cache 中可能存在的历史 `tensor` 混合值。

## 5. 模型总结构

当前 segmentation head 由 `DAGPSafeHead` 实现。启用 NDR 后，head 的 forward 签名实际使用：

```python
forward(feat, image_68=None, return_aux=False)
```

训练 student 时：

```text
feat = raw cached DINO feature [B,384,37,37]
image_68 = RGB image [B,3,68,68]
return_aux = True
```

teacher forward 时：

```text
feat = raw cached DINO feature [B,384,37,37]
image_68 = RGB image [B,3,68,68]
return_aux = False
```

整体结构为：

```text
raw feature [B,384,37,37]
    |
    +-- base_head 1x1 conv
    |       -> base_logits_37 [B,1,37,37]
    |
    +-- DAGP graph branch
    |       -> graph_logits_37 [B,1,37,37]
    |
    +-- uncertainty output gate on base_logits_37
    |
    -> coarse_logits_37
    |
    -> bilinear upsample to coarse_logits_68
    |
    +-- NDR branch using image_68 and coarse_prob_68
    |       -> residual_logits_68
    |       -> sobel_68
    |       -> detail_gate
    |
    -> final_logits_68
```

当 `return_aux=True` 时，student 会额外返回用于 loss 和日志的中间结果：

```text
logits
base_logits
graph_logits
coarse_logits_37
coarse_logits_68
coarse_prob_68
uncertainty_68
sobel_68
residual_logits_68
detail_gate
dagp_scale
dagp_alpha_eff
dagp_gamma_eff
ndr_beta_eff
```

## 6. DAGP-Safe Head

### 6.1 设计目的

DAGP-Safe 的目标是在 DINO token 空间里传播相似区域的信息，但避免一开始训练就让 graph branch 扰乱简单可靠的 base branch。

它不是替代 1x1 head，而是在 base head 上加一个受控 residual：

```text
coarse_logits_37 = base_logits_37 + alpha_eff * uncertainty_gate_37 * graph_logits_37
```

在前 6 个 epoch：

```text
alpha_eff = 0
gamma_eff = 0
coarse_logits_37 = base_logits_37
```

所以 early training 阶段不让 graph branch 影响输出。

### 6.2 base head

base head 是最直接的线性 segmentation head：

```text
base_head = Conv2d(384, 1, kernel_size=1)
```

输入：

```text
[B,384,37,37]
```

输出：

```text
base_logits_37: [B,1,37,37]
```

这个分支承担两个角色：

1. 提供基础 coarse prediction。
2. 为 graph branch 和 NDR 提供 gating 参考。

### 6.3 graph branch 参数

当前配置：

```text
DAGP_SAFE_HIDDEN = 64
DAGP_SAFE_TOPK = 12
DAGP_SAFE_TAU = 0.07
DAGP_SAFE_ALPHA_MAX = 0.05
DAGP_SAFE_GAMMA_MAX = 0.03
DAGP_SAFE_WARMUP_EPOCH = 6
DAGP_SAFE_RAMP_START_EPOCH = 7
DAGP_SAFE_RAMP_END_EPOCH = 15
DAGP_SAFE_USE_PROB_GATE = True
DAGP_SAFE_PROB_GATE_SIGMA = 0.25
DAGP_SAFE_AFFINITY_DETACH = True
DAGP_SAFE_EXCLUDE_SELF = True
```

模块结构：

```text
proj = Conv2d(384, 64, kernel_size=1)
value = Linear(64, 64)
graph_pred = Conv2d(64, 1, kernel_size=1)
```

`graph_pred` 使用 zero init，因此 graph branch 初始输出为 0。再配合 ramp，训练前期更稳定。

### 6.4 cosine affinity

对 raw DINO feature 做 graph，不对 upsample 后的 feature 做 graph。

输入 feature：

```text
feat: [B,384,37,37]
```

展平为空间节点：

```text
x: [B,N,384]
N = 37 * 37 = 1369
```

计算步骤：

```text
1. 如果 DAGP_SAFE_AFFINITY_DETACH=True，则 x.detach()
2. float32 normalize
3. cosine similarity = x_norm @ x_norm^T
4. mask diagonal，排除自己到自己的边
5. 每个节点取 topk=12 个邻居
6. attention = softmax(topk_similarity / tau)
```

这里不会构造 `[B,N,N,D]` 这种巨大 value expansion。代码使用 batch-offset gather，只收集 top-k 邻居的 value：

```text
value: [B,N,64]
topk_idx: [B,N,12]
neighbor_value: [B,N,12,64]
```

这样显存规模约为 `B * N * K * hidden`，而不是 `B * N * N * hidden`。

### 6.5 probability gate

仅靠 DINO cosine affinity 可能会把语义相近但前景概率差异很大的点连起来。当前配置启用 probability gate：

```text
DAGP_SAFE_USE_PROB_GATE = True
DAGP_SAFE_PROB_GATE_SIGMA = 0.25
```

对 base logits 得到 base probability：

```text
p_base = sigmoid(base_logits_37).detach()
```

对每个节点和 top-k 邻居计算：

```text
prob_diff = abs(p_i - p_j)
prob_gate = exp(-prob_diff / sigma)
```

再把原 attention 乘以 `prob_gate` 并重新归一化。

直观理解：

- base probability 相近的邻居保留较大权重。
- base probability 差异大的邻居被削弱。
- 这减少了前景和背景之间错误传播。

### 6.6 graph propagation

DAGP graph branch 用 `proj` 得到 hidden feature：

```text
z_map = proj(feat)           # [B,64,37,37]
z = flatten(z_map)           # [B,N,64]
value = Linear(z)            # [B,N,64]
```

对 top-k 邻居做 attention aggregation：

```text
agg_i = sum_j attention_ij * value_j
```

然后做 residual propagation：

```text
z_prop_i = z_i + gamma_eff * agg_i
```

最后 reshape 回 feature map：

```text
z_prop_map: [B,64,37,37]
graph_logits_37 = graph_pred(z_prop_map)
```

`gamma_eff` 控制 graph 信息进入 hidden feature 的强度。

### 6.7 DAGP ramp schedule

DAGP branch 的有效强度由 epoch 控制：

```text
epoch <= 6:
    dagp_scale = 0

epoch 7 到 14:
    dagp_scale = (epoch - 7 + 1) / (15 - 7 + 1)

epoch >= 15:
    dagp_scale = 1
```

最终：

```text
alpha_eff = DAGP_SAFE_ALPHA_MAX * dagp_scale
gamma_eff = DAGP_SAFE_GAMMA_MAX * dagp_scale
```

当前最大值：

```text
alpha_max = 0.05
gamma_max = 0.03
```

这意味着 graph residual 永远是小幅修正，不会成为主路径。

### 6.8 uncertainty output gate

当前启用：

```text
DAGP_SAFE_USE_UNCERTAINTY_OUTPUT_GATE = True
DAGP_SAFE_UNCERTAINTY_POWER = 1.0
DAGP_SAFE_UNCERTAINTY_MIN = 0.0
DAGP_SAFE_UNCERTAINTY_MAX = 1.0
DAGP_SAFE_UNCERTAINTY_DETACH = True
```

gate 基于 base probability：

```text
p = sigmoid(base_logits_37)
uncertainty_gate = 1 - 2 * abs(p - 0.5)
```

含义：

- 当 `p` 接近 0.5，base 不确定，gate 接近 1。
- 当 `p` 接近 0 或 1，base 很确定，gate 接近 0。
- graph residual 主要在不确定区域起作用。

最终 coarse logits：

```text
coarse_logits_37 = base_logits_37 + alpha_eff * uncertainty_gate * graph_logits_37
```

由于 `DAGP_SAFE_UNCERTAINTY_DETACH=True`，gate 不把梯度回传到 base probability 的 gate 计算路径。

## 7. NDR Native Detail Residual Branch

### 7.1 设计目的

DAGP-Safe 在 DINO token 空间工作，分辨率是 `37x37`。这有利于稳定语义传播，但边界和细节会比较粗。

NDR 的目标是在 `68x68` loss 空间中做轻量、可训练的 residual refinement：

```text
final_logits_68 = coarse_logits_68 + beta_eff * detail_gate * residual_logits_68
```

它不是后处理，而是 `nn.Module` 的一部分：

- student 有 NDR 参数。
- teacher 也有 NDR 参数。
- optimizer 更新 NDR。
- EMA 同步 NDR。
- train/val/eval 都走同一条 forward。

### 7.2 NDR 输入

当前配置：

```text
USE_NDR_BRANCH = True
NDR_INPUT_RGB = True
NDR_INPUT_SOBEL = True
NDR_INPUT_COARSE_PROB = True
NDR_IN_CHANNELS = 5
```

NDR 输入由三类信息拼接：

```text
image_68:        [B,3,68,68]
sobel_68:        [B,1,68,68]
coarse_prob_68:  [B,1,68,68]
```

拼接后：

```text
ndr_input: [B,5,68,68]
```

`coarse_prob_68` 来自：

```text
coarse_logits_68 = bilinear_upsample(coarse_logits_37, 68x68)
coarse_prob_68 = sigmoid(coarse_logits_68.detach())
```

这里 detach 是有意的。NDR 可以根据 coarse probability 做细节修正，但不通过这个输入反向改变 coarse path。

### 7.3 Sobel map

NDR 内部从 `image_68` 计算 Sobel 边缘：

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
dx = conv(gray, sobel_x)
dy = conv(gray, sobel_y)
sobel = sqrt(dx^2 + dy^2 + 1e-6)
sobel = sobel / max_per_image(sobel)
sobel = clamp(sobel, 0, 1)
```

输出：

```text
sobel_68: [B,1,68,68]
```

Sobel 不读取 GT，不落盘，不作为监督信号，只作为图像边缘先验。

### 7.4 NDR 网络结构

当前配置：

```text
NDR_HIDDEN = 32
NDR_NUM_LAYERS = 3
NDR_USE_GN = True
NDR_GN_GROUPS = 4
NDR_ACT = gelu
NDR_ZERO_INIT_OUT = True
NDR_RESIDUAL_CLIP = 2.0
```

结构为：

```text
Conv3x3 5 -> 32
GroupNorm(4)
GELU

Conv3x3 32 -> 32
GroupNorm(4)
GELU

Conv3x3 32 -> 16
GroupNorm(4)
GELU

Conv1x1 16 -> 1
```

最后一层 `out_conv` zero init：

```text
out_conv.weight = 0
out_conv.bias = 0
```

这保证 NDR 初始 residual 为 0。即使后续 beta ramp 打开，初始阶段也不会突然扰乱 coarse prediction。

raw residual 经过裁剪：

```text
residual_logits_68 = NDR_RESIDUAL_CLIP * tanh(raw_residual)
```

当前：

```text
NDR_RESIDUAL_CLIP = 2.0
```

所以 residual logit 修正范围被限制在 `[-2, 2]`。

### 7.5 NDR beta ramp

当前配置：

```text
NDR_BETA_MAX = 0.10
NDR_WARMUP_EPOCH = 6
NDR_RAMP_START_EPOCH = 7
NDR_RAMP_END_EPOCH = 15
```

beta schedule 与 DAGP 一样：

```text
epoch <= 6:
    beta_eff = 0

epoch 7 到 14:
    beta_eff = 0.10 * (epoch - 7 + 1) / 9

epoch >= 15:
    beta_eff = 0.10
```

因此 epoch 1 到 6：

```text
final_logits_68 = coarse_logits_68
```

注意这里的 `coarse_logits_68` 是 `base_logits_37` 或 DAGP coarse logits 上采样到 `68x68` 后的结果。也就是说，早期训练不是原 simple head 的“先 upsample feature 再 1x1 conv”路径，而是当前 DAGP-Safe/NDR 结构中的“raw 37x37 base head 再 upsample”路径。

### 7.6 detail gate

当前配置：

```text
NDR_USE_UNCERTAINTY_GATE = True
NDR_USE_EDGE_GATE = True
NDR_GATE_MODE = uncertainty_edge_boost
```

先根据 coarse probability 计算 uncertainty：

```text
uncertainty_68 = 1 - 2 * abs(coarse_prob_68 - 0.5)
uncertainty_68 = clamp(uncertainty_68, 0, 1)
```

再根据 RGB Sobel 边缘计算 edge gate：

```text
edge_gate = 0.5 + 0.5 * sobel_68
```

最终：

```text
detail_gate = clamp(uncertainty_68 * edge_gate, 0, 1)
```

直观含义：

- coarse prediction 很确定的区域，NDR residual 被压低。
- coarse prediction 不确定的区域，NDR residual 更容易生效。
- RGB 边缘强的区域，NDR residual 被增强。
- RGB 边缘弱但 uncertainty 高的区域，仍保留 `0.5` 的 edge 基础权重，不完全关闭。

最终 NDR 输出：

```text
final_logits_68 =
    coarse_logits_68
    + beta_eff * detail_gate * residual_logits_68
```

### 7.7 本配置未启用 TADR

当前代码支持 TADR router，但本配置没有设置 `USE_TADR_ROUTER=True`。

因此本版本实际没有：

- router input。
- router map。
- trainable adaptive routing gate。
- TADR loss。

NDR 的 gate 就是：

```text
detail_gate = uncertainty_68 * (0.5 + 0.5 * sobel_68)
```

## 8. 训练 target

### 8.1 DESPL-only initial pseudo

当前训练 pseudo 来自 DESPL：

```text
fixed_target = pseudo_68 = p_despl_68
```

`pseudo_fixed` 被读取，但由于：

```text
P_INIT_DESPL_WEIGHT = 1.0
P_INIT_FIXED_WEIGHT = 0.0
```

所以 fixed pseudo 不参与当前训练 target。

日志中的：

```text
fixed_used_for_training = False
use_fixed_in_pseudo = False
```

对应的就是这一点。

### 8.2 EMA teacher target

teacher 是 student 的 EMA 版本。每个 iteration：

1. teacher 先 forward 生成 target。
2. student 用这个 target 计算 loss 并更新参数。
3. optimizer step 后，再用 student 更新 teacher EMA。

这样避免当前 iteration 的 student 更新泄漏进同一轮 target。

teacher logits 会 resize 到 loss size：

```text
teacher_logits_68: [B,1,68,68]
teacher_prob = sigmoid(teacher_logits_68)
teacher_binary = teacher_prob > 0.5
```

### 8.3 DESPL 与 teacher 的融合 schedule

本配置没有指定特殊 `TEACHER_FUSION_MODE`，所以走默认 schedule。

默认 teacher weight：

```text
epoch <= 20:
    teacher_weight = (epoch - 1) / 20
    fixed_weight = 1 - teacher_weight

epoch >= 21:
    teacher_weight = 1
    fixed_weight = 0
```

关键点：

```text
epoch 1:
    fixed = 1.00
    teacher = 0.00

epoch 10:
    fixed = 0.55
    teacher = 0.45

epoch 20:
    fixed = 0.05
    teacher = 0.95

epoch 21 到 25:
    fixed = 0.00
    teacher = 1.00
```

混合 target：

```text
epoch < FINETUNE_RESET_EPOCH:
    mixed_target = fixed_weight * DESPL_pseudo + teacher_weight * teacher_binary

epoch >= FINETUNE_RESET_EPOCH:
    mixed_target = teacher_binary
```

当前：

```text
FINETUNE_RESET_EPOCH = 21
```

因此 epoch21 开始是 teacher-only target。

## 9. 损失函数

### 9.1 主损失

主输出是 NDR final logits：

```text
student_logits = final_logits_68
```

主损失：

```text
loss_final = BCEWithLogitsLoss(final_logits_68, mixed_target)
```

`mixed_target` 形状为：

```text
[B,1,68,68]
```

### 9.2 NDR coarse auxiliary loss

当前启用：

```text
USE_NDR_COARSE_AUX = True
LAMBDA_NDR_COARSE_AUX = 0.5
```

coarse auxiliary 使用 NDR 前的 coarse logits：

```text
coarse_logits_68 = upsample(coarse_logits_37)
loss_coarse_aux = BCEWithLogitsLoss(coarse_logits_68, mixed_target)
```

作用：

- 防止模型把所有监督压力都转移给 NDR residual。
- 保持 DAGP coarse path 自身可用。
- 让 NDR 更像细节修正，而不是主分割器。

### 9.3 base auxiliary loss

当前启用：

```text
USE_BASE_AUX_LOSS = True
LAMBDA_BASE_AUX = 0.5
LAMBDA_BASE_AUX_AFTER_RESET = 0.3
BASE_AUX_NORMALIZE = True
```

base auxiliary 使用最基础的 `base_logits_37`，先 resize 到 `68x68`：

```text
base_logits_68 = upsample(base_logits_37)
loss_base_aux = BCEWithLogitsLoss(base_logits_68, mixed_target)
```

作用：

- 保护 base 1x1 branch。
- 避免 graph branch 或 NDR branch 学偏后拖垮基础分割能力。
- reset 后仍保留较弱约束，lambda 从 `0.5` 降到 `0.3`。

### 9.4 NDR 版本的总 loss

NDR 配置下，总 loss 使用权重归一化：

```text
loss =
    (1.0 * loss_final
     + 0.5 * loss_coarse_aux
     + lambda_base * loss_base_aux)
    / (1.0 + 0.5 + lambda_base)
```

epoch 1 到 20：

```text
lambda_base = 0.5
loss denominator = 2.0
```

epoch 21 到 25：

```text
lambda_base = 0.3
loss denominator = 1.8
```

这样做的目的不是简单把辅助 loss 加大，而是让总 loss 尺度大致稳定，减少 learning rate 和 loss magnitude 的耦合变化。

### 9.5 未启用的 loss

当前配置没有启用：

- GKD。
- QRA。
- CCR。
- DRE++。
- Anchor PBCE。
- Multi-view consistency。
- Proto contrast。
- NDR residual regularization。
- TADR router loss。

其中配置明确：

```text
GKD_MODE = off
NDR_USE_RES_REG = False
```

## 10. Optimizer、LR floor 和 reset

### 10.1 optimizer

基础 DINO 配置给出：

```text
lr = 6e-4
```

训练使用当前代码的 optimizer/scheduler 构建逻辑。这个版本没有改 optimizer 类型，重点是保留 iter-level StepLR，并加入 LR floor。

### 10.2 StepLR + LR floor

当前：

```text
LR_POLICY = step_floor
USE_LR_FLOOR = True
LR_FLOOR = 2e-5
LR_FLOOR_MODE = global
```

旧 scheduler 仍然是：

```text
StepLR(step_size=25, gamma=0.95)
```

并且是 iteration-level scheduler step。

每次 `scheduler.step()` 后调用：

```text
apply_lr_floor(optimizer, cfg)
```

如果某个 param group 的实际 lr 低于 `2e-5`，就 clamp 回 `2e-5`。

因此日志中的 lr 是真实 optimizer lr，不是 scheduler 理论 lr。

### 10.3 epoch21 reset

当前：

```text
FINETUNE_RESET_EPOCH = 21
```

在 epoch21 进入 teacher-only 阶段时，代码会重建 optimizer 和 scheduler，并重置 global step。teacher 不重置，继续保留 EMA teacher。

直观上：

- epoch1 到 20：DESPL 与 teacher 逐步融合。
- epoch21 到 25：teacher-only finetune。
- reset optimizer/scheduler 让 teacher-only 阶段重新获得较高学习率起点。
- LR floor 仍然生效，防止后期 lr 衰减过低。

## 11. EMA teacher

teacher 是 student 的指数滑动平均：

```text
EMA_WEIGHT = 0.99
```

每次 optimizer step 后更新：

```text
teacher = EMA(teacher, student)
```

teacher 有完整的同构模型参数，包括：

- base head。
- DAGP graph branch。
- uncertainty gate 所依赖的参数。
- NDR branch。

因为 NDR 是 head 内部模块，所以 teacher forward 和 eval 都会使用 NDR final logits，而不是只使用 coarse logits。

## 12. 评估流程

评估时仍使用：

```text
HEAD_TYPE = dagp_safe
USE_NDR_BRANCH = True
```

因此 eval 数据流为：

```text
cached feature [1,384,37,37]
image_68 [1,3,68,68]
    -> DAGP-Safe
    -> NDR final logits [1,1,68,68]
    -> resize to original GT size
    -> sigmoid / threshold / metric
```

评估不使用：

- GT 参与 forward。
- CRF。
- morphology。
- connected component filtering。
- test-time rule refinement。

当前 `eval.py` CLI 要求显式传入 checkpoint：

```bash
python eval.py \
  --config configs/dinov1_s8_despl_dagp_uncgate_ndr_lrfloor_2e5.py \
  --ckpt ../workdir/dinov1_s8_despl_dagp_uncgate_ndr_lrfloor_2e5/train/ckpt/best.pth
```

## 13. checkpoint 保存

checkpoint 中保存：

```text
epoch
backbone_key
student state_dict
teacher state_dict
optimizer state_dict
scheduler state_dict
best_metric
best_epoch
config
```

当前保存逻辑包括：

- `best.pth`。
- 每 `SAVE_INTERVAL=5` 个 epoch 保存一次。
- reset epoch 保存一次。
- 最后五轮全部保存。

对 `MAX_EPOCH=25`、`FINETUNE_RESET_EPOCH=21` 来说，典型 epoch checkpoint 包括：

```text
epoch_005.pth
epoch_010.pth
epoch_015.pth
epoch_020.pth
epoch_021.pth
epoch_022.pth
epoch_023.pth
epoch_024.pth
epoch_025.pth
```

## 14. 关键 shape 表

| 名称 | shape | 说明 |
|---|---:|---|
| `feature` | `[B,384,37,37]` | cached DINOv1-s8 raw feature |
| `image_68` | `[B,3,68,68]` | NDR RGB input |
| `pseudo` | `[B,1,68,68]` | DESPL-only training pseudo |
| `base_logits` | `[B,1,37,37]` | base 1x1 head 输出 |
| `graph_logits` | `[B,1,37,37]` | DAGP graph branch 输出 |
| `coarse_logits_37` | `[B,1,37,37]` | DAGP-Safe coarse 输出 |
| `coarse_logits_68` | `[B,1,68,68]` | coarse logits 上采样到 loss size |
| `coarse_prob_68` | `[B,1,68,68]` | `sigmoid(coarse_logits_68.detach())` |
| `sobel_68` | `[B,1,68,68]` | RGB Sobel edge map |
| `uncertainty_68` | `[B,1,68,68]` | NDR uncertainty gate |
| `detail_gate` | `[B,1,68,68]` | NDR final residual gate |
| `residual_logits_68` | `[B,1,68,68]` | NDR residual logits |
| `final_logits_68` | `[B,1,68,68]` | 最终训练和 eval 输出 |
| `teacher_binary` | `[B,1,68,68]` | EMA teacher 二值 target |
| `mixed_target` | `[B,1,68,68]` | DESPL/teacher 融合 target |

## 15. 一个 batch 的完整训练过程

下面按真实训练顺序展开：

### 15.1 读取 batch

```text
batch["feature"]      -> [B,384,37,37]
batch["image_68"]     -> [B,3,68,68]
batch["pseudo"]       -> [B,1,68,68]
batch["pseudo_despl"] -> [B,1,68,68]
batch["pseudo_fixed"] -> [B,1,68,68]
batch["dataset"]
batch["stem"]
batch["image_path"]
```

### 15.2 设置当前 epoch

每个 epoch 开始，student 和 teacher 都会调用：

```text
set_epoch(epoch)
```

`DAGPSafeHead` 内部的 `current_epoch_tensor` 决定：

- DAGP scale。
- `alpha_eff`。
- `gamma_eff`。
- NDR `beta_eff`。

### 15.3 student forward

```text
student_out = student(feature, image_68=image_68, return_aux=True)
```

输出主 logits：

```text
student_logits = student_out["logits"] = final_logits_68
```

### 15.4 teacher forward

```text
teacher_out = teacher(feature, image_68=image_68, return_aux=False)
teacher_logits = teacher_out
teacher_prob = sigmoid(teacher_logits)
teacher_binary = teacher_prob > 0.5
```

teacher 不返回 aux，因为 teacher 只用于 target。

### 15.5 生成 mixed target

```text
fixed_target = pseudo_68
```

如果 epoch < 21：

```text
mixed_target = fixed_weight * fixed_target + teacher_weight * teacher_binary
```

如果 epoch >= 21：

```text
mixed_target = teacher_binary
```

### 15.6 计算 loss

```text
loss_final = BCE(final_logits_68, mixed_target)
loss_coarse_aux = BCE(coarse_logits_68, mixed_target)
loss_base_aux = BCE(base_logits_68, mixed_target)

loss = weighted_normalized_sum(...)
```

### 15.7 更新 student 和 teacher

```text
optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step()
apply_lr_floor()
update_ema(student, teacher)
```

## 16. 为什么这样设计

### 16.1 为什么 DAGP 在 37x37 上做

DINO feature 的 token grid 是 `37x37`。在这个空间做 affinity 有三个好处：

1. 每个节点对应真实 DINO token，语义更干净。
2. graph 节点数是 1369，计算 top-k affinity 可控。
3. 避免先插值到 68 后产生人工平滑 token，再基于插值结果建图。

如果先把 feature 插值到 `68x68` 再做 graph，节点数会变成 4624，graph 计算量显著增加，而且 affinity 是基于插值 feature，不再是原始 DINO token。

### 16.2 为什么 NDR 在 68x68 上做

训练 pseudo 和 loss size 是 `68x68`，DAGP 输出只有 `37x37`。边界和细节更适合在 loss 空间处理。

NDR 直接使用：

- 原图 RGB。
- RGB Sobel 边缘。
- coarse probability。

因此它能学习“在原图局部纹理和 coarse mask 不确定处如何修正 logits”。

### 16.3 为什么 residual 而不是直接输出 mask

NDR 只输出 residual：

```text
final = coarse + residual
```

这保留了 coarse path 的主导地位。NDR 的职责是修边和补细节，不是重新做一个完整 segmentation head。

再加上：

```text
residual_clip = 2.0
beta_max = 0.10
detail_gate <= 1
```

实际进入 final logits 的 NDR 最大幅度受到强约束。

### 16.4 为什么需要 coarse aux 和 base aux

如果只监督 final logits，模型可能学到：

```text
coarse path 变弱
NDR residual 过度补偿
```

这会让模型依赖局部图像纹理，可能损害 COD10K 这类复杂场景泛化。

coarse aux 和 base aux 的作用是让每一级输出都保持分割能力：

- base aux 保护 1x1 base branch。
- coarse aux 保护 DAGP coarse branch。
- final loss 训练 NDR 后的最终输出。

## 17. 日志应该如何理解

训练启动日志会打印：

- experiment name。
- head type。
- DESPL-only 状态。
- LR floor 状态。
- DAGP-Safe 参数。
- Uncertainty gate 参数。
- NDR 参数。
- cache 路径和 feature shape。

first batch debug 会确认：

```text
raw feature shape = [B,384,37,37]
coarse_logits_37 shape = [B,1,37,37]
coarse_logits_68 shape = [B,1,68,68]
image_68 shape = [B,3,68,68]
sobel_68 shape = [B,1,68,68]
residual_logits_68 shape = [B,1,68,68]
final_logits_68 shape = [B,1,68,68]
pseudo shape = [B,1,68,68]
```

epoch 级别日志里重点看：

- `lr`：真实 optimizer lr，包含 LR floor 后的值。
- fixed/teacher weight：当前 target 融合比例。
- `dagp_scale`、`alpha_eff`、`gamma_eff`：DAGP 是否已打开。
- `unc_gate_mean/min/max`：DAGP uncertainty output gate。
- `ndr_beta_eff`：NDR residual 强度。
- `ndr_gate_mean/min/max`：NDR detail gate。
- `ndr_residual_abs_mean/max`：NDR residual 幅度。
- `loss_final`、`loss_coarse_aux`、`loss_base_aux`：三类 loss 的相对情况。

## 18. 与相关消融版本的区别

### 18.1 与 DESPL-only simple head

DESPL-only simple head 使用 simple segmentation head，不启用：

- DAGP graph propagation。
- uncertainty output gate。
- NDR residual branch。

当前版本保留 DESPL-only pseudo，但替换 head 并加入 detail branch。

### 18.2 与 4-dagp-safe-uncgate-2e5

4-dagp-safe-uncgate-2e5 通常指 DAGP-Safe + Uncertainty Gate + LR floor，但没有 NDR。

当前版本在此基础上新增：

- `image_68` 数据输入。
- NDR residual branch。
- Sobel edge gate。
- NDR coarse aux loss。
- final logits 从 37x37 coarse 输出变为 68x68 refined 输出。

### 18.3 与 beta005

beta005 只把：

```text
NDR_BETA_MAX = 0.05
```

当前主版本是：

```text
NDR_BETA_MAX = 0.10
```

### 18.4 与 gate025075

gate025075 改了 edge gate：

```text
uncertainty_edge_stronger:
    edge_gate = 0.25 + 0.75 * sobel
```

当前主版本是：

```text
uncertainty_edge_boost:
    edge_gate = 0.5 + 0.5 * sobel
```

### 18.5 与 TADR

TADR 在 NDR gate 上再加一个 trainable router map。

当前主版本没有启用 TADR，因此：

```text
final_logits_68 = coarse_logits_68 + beta_eff * detail_gate * residual_logits_68
```

没有：

```text
router_map_68
```

## 19. 设计中的稳定性约束

这个版本的很多设计都是为了降低新分支带来的不稳定性：

1. DAGP graph branch zero-init 或弱初始化。
2. DAGP 前 6 个 epoch 完全不影响输出。
3. DAGP residual 最大系数只有 `0.05`。
4. DAGP propagation hidden aggregation 最大系数只有 `0.03`。
5. DAGP residual 受 uncertainty output gate 限制。
6. NDR output conv zero-init。
7. NDR 前 6 个 epoch beta 为 0。
8. NDR beta 最大只有 `0.10`。
9. NDR residual 被 tanh 和 clip 限制在 `[-2,2]`。
10. NDR residual 受 uncertainty 和 Sobel gate 限制。
11. coarse aux 和 base aux 保护主干输出。
12. LR floor 防止后期学习率过低。
13. epoch21 reset 给 teacher-only 阶段重新优化空间。

## 20. 训练命令

```bash
source /home/dell01/anaconda3/etc/profile.d/conda.sh
conda activate cth
cd /home/dell01/CTH/MY-baseline/main

python train.py --config configs/dinov1_s8_despl_dagp_uncgate_ndr_lrfloor_2e5.py
```

## 21. 评估命令

```bash
source /home/dell01/anaconda3/etc/profile.d/conda.sh
conda activate cth
cd /home/dell01/CTH/MY-baseline/main

python eval.py \
  --config configs/dinov1_s8_despl_dagp_uncgate_ndr_lrfloor_2e5.py \
  --ckpt ../workdir/dinov1_s8_despl_dagp_uncgate_ndr_lrfloor_2e5/train/ckpt/best.pth
```

如果你的实际权重目录是手工命名的，例如：

```text
../workdir/5-dagp-safe-uncgate-ndr-2e5/train/ckpt/best.pth
```

则只需要替换 `--ckpt` 路径，config 仍然使用当前主版本配置。

## 22. 最简伪代码

```python
feature = batch["feature"]        # [B,384,37,37]
image_68 = batch["image_68"]      # [B,3,68,68]
pseudo_68 = batch["pseudo"]       # [B,1,68,68]

student.set_epoch(epoch)
teacher.set_epoch(epoch)

student_out = student(feature, image_68=image_68, return_aux=True)
final_logits_68 = student_out["logits"]

with no_grad():
    teacher_logits_68 = teacher(feature, image_68=image_68, return_aux=False)
    teacher_binary = (sigmoid(teacher_logits_68) > 0.5).float()

if epoch < 21:
    fixed_weight, teacher_weight = default_fusion(epoch)
    target = fixed_weight * pseudo_68 + teacher_weight * teacher_binary
else:
    target = teacher_binary

loss_final = bce(final_logits_68, target)
loss_coarse = bce(student_out["coarse_logits_68"], target)
loss_base = bce(upsample(student_out["base_logits"], 68), target)

lambda_base = 0.5 if epoch < 21 else 0.3
loss = (
    loss_final
    + 0.5 * loss_coarse
    + lambda_base * loss_base
) / (1.0 + 0.5 + lambda_base)

loss.backward()
optimizer.step()
scheduler.step()
apply_lr_floor()
update_ema(student, teacher)
```

## 23. 一句话复述

这个版本不是简单把 head 变复杂，而是把任务拆成两级：

```text
37x37 DINO token space:
    用 DAGP-Safe 做受控语义传播，得到稳定 coarse mask。

68x68 loss/image space:
    用 NDR 根据 RGB 边缘和 coarse uncertainty 做小幅 residual 修正。
```

训练上则用 DESPL-only pseudo 起步，逐渐交给 EMA teacher，最后 teacher-only reset finetune；同时用 LR floor、base aux、coarse aux 和 ramp schedule 控制新分支不会破坏原有 baseline 的稳定性。
