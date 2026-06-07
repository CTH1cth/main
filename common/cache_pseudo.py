import argparse
import inspect
import math
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel

if __package__ in {None, ""}:
    # 支持从 main/common 目录直接执行：python cache_pseudo.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import build_image_items, ensure_dir, load_config, write_jsonl


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(image_path, size):
    # fixed pseudo 生成统一使用 224x224 DINO 输入。
    image = Image.open(image_path).convert("RGB")
    original_size = (image.height, image.width)
    image = image.resize((size, size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0), original_size


def compute_img_bkg_seg(
    attentions,
    feats,
    featmap_dims,
    th_bkg,
    up_size=None,
    dim=64,
    epsilon=1e-10,
    apply_weights=True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 复用 UCOD-DPL 的背景种子思想：找低注意力参考 patch，再按 key 相似度扩展背景。
    w_featmap, h_featmap = featmap_dims
    if up_size is None:
        up_size = w_featmap
    nb, nh = attentions.shape[:2]
    att = attentions[:, :, 0, 1:].reshape(nb, nh, w_featmap, h_featmap)
    att = F.interpolate(att, size=(up_size, up_size), mode="bilinear")
    descs = feats[:, 1:, :]

    threshold = torch.mean(att.reshape(nb, -1), dim=1)
    q = torch.sum(att.reshape(nb, nh, up_size * up_size) > threshold[:, None, None], axis=2)
    q = q / (up_size * up_size)
    beta = torch.log(torch.sum(q + epsilon, dim=1)[:, None] / (q + epsilon))

    if apply_weights:
        descs = (descs.reshape(nb, -1, nh, dim) * beta[:, None, :, None]).reshape(nb, -1, nh * dim)
    else:
        descs = descs.reshape(nb, -1, nh, dim).reshape(nb, -1, nh * dim)

    descs = descs.reshape(nb, w_featmap, h_featmap, -1)
    descs = descs.permute(0, 3, 1, 2)
    descs = F.interpolate(descs, size=(up_size, up_size), mode="bilinear")
    descs = descs.permute(0, 2, 3, 1).reshape(nb, -1, nh * dim)
    w_featmap = h_featmap = up_size

    descs = F.normalize(descs, dim=-1, p=2)
    cos_sim = torch.bmm(descs, descs.permute(0, 2, 1))

    if apply_weights:
        att = att.reshape(nb, nh, w_featmap, h_featmap) * beta[:, :, None, None]
    else:
        att = att.reshape(nb, nh, w_featmap, h_featmap)
    id_pixel_ref = torch.argmin(torch.sum(att, axis=1).reshape(nb, -1), dim=-1)

    cos_sim = cos_sim.reshape(nb, -1, w_featmap * h_featmap)
    batch_index = torch.arange(cos_sim.size(0), device=cos_sim.device)
    bkg_mask = (
        cos_sim[batch_index, id_pixel_ref, :]
        .reshape(nb, w_featmap, h_featmap)
        > th_bkg
    )
    fn_mask = 1 - bkg_mask.float()
    sim_map = (
        cos_sim[batch_index, id_pixel_ref, :]
        .reshape(nb, w_featmap, h_featmap)
        .float()
    )
    sim_map = 1 - sim_map
    sim_map = sim_map / (sim_map.max() + 1e-10)
    return bkg_mask.float(), (sim_map * fn_mask).float()


def refine_post_process(mask, area_threshold=4):
    # 去掉被相反类别包围的小连通域，保持 UCOD-DPL 原始后处理思路。
    mask_np = mask.numpy().astype(np.uint8).squeeze()
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    refined_mask = mask_np.copy()

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= area_threshold:
            continue
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        component_mask = labels_im[y:y + height, x:x + width] == label

        x_start = max(x - 1, 0)
        y_start = max(y - 1, 0)
        x_end = min(x + width + 1, mask_np.shape[1])
        y_end = min(y + height + 1, mask_np.shape[0])

        surrounding = refined_mask[y_start:y_end, x_start:x_end].copy()
        comp_y, comp_x = np.where(component_mask)
        comp_y += y - y_start
        comp_x += x - x_start
        surrounding_mask = np.ones_like(surrounding, dtype=bool)
        surrounding_mask[comp_y, comp_x] = False
        surrounding_pixels = surrounding[surrounding_mask]
        if surrounding_pixels.size == 0:
            continue

        component_label = refined_mask[y + height // 2, x + width // 2]
        opposite_label = 1 - component_label
        if np.all(surrounding_pixels == opposite_label):
            refined_mask[y:y + height, x:x + width][component_mask] = opposite_label

    return torch.tensor(refined_mask).unsqueeze(0).float()


def resolve_key_projection(model):
    # DINO key hook 路径优先使用 transformers ViT 的标准 attention key 投影。
    try:
        return model.encoder.layer[-1].attention.attention.key, "encoder.layer[-1].attention.attention.key"
    except AttributeError:
        pass

    candidates = []
    for name, module in model.named_modules():
        lname = name.lower()
        if lname.endswith(".attention.attention.key") or lname.endswith(".attention.key") or lname.endswith(".key"):
            candidates.append((name, module))
    if not candidates:
        raise AttributeError("Could not find DINO key projection module.")
    name, module = candidates[-1]
    return module, name


def call_dino(model, inputs):
    # 伪标签生成需要 attentions，因此请求 DINO 返回最后一层 attention。
    params = inspect.signature(model.forward).parameters
    kwargs = {}
    if "interpolate_pos_encoding" in params:
        kwargs["interpolate_pos_encoding"] = True
    if "output_attentions" in params:
        kwargs["output_attentions"] = True
    return model(inputs, **kwargs)


def load_dino(cfg, device):
    # 只从配置中的本地 DINO 权重路径加载，不联网下载。
    dino_cfg = cfg.DINO
    model_path = Path(dino_cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Local DINO weight path not found: {model_path}")
    try:
        model = AutoModel.from_pretrained(
            str(model_path),
            output_attentions=True,
            local_files_only=True,
            add_pooling_layer=False,
        )
    except TypeError:
        model = AutoModel.from_pretrained(str(model_path), output_attentions=True, local_files_only=True)
    model.to(device)
    model.eval()
    return model


def generate_pseudo_cache(cfg, overwrite=False, logger=print):
    # 只为训练集生成 fixed foreground pseudo cache。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_cfg = cfg.DINO
    input_size = int(dino_cfg["pseudo_input_size"])
    cache_root = Path(cfg.CACHE_ROOT) / "pseudo_label_cache" / cfg.BACKBONE_KEY
    manifest_path = cache_root / "manifest_train.jsonl"
    ensure_dir(cache_root)

    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists; pass --overwrite to regenerate: {manifest_path}")

    model = load_dino(cfg, device)
    key_holder = {"tensor": None}

    def hook_fn_key(_module, _input, output):
        key_holder["tensor"] = output.detach()

    key_module, key_path = resolve_key_projection(model)
    handle = key_module.register_forward_hook(hook_fn_key)
    logger(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger(f"model_path = {dino_cfg['model_path']}")
    logger(f"key_hook = {key_path}")
    logger(f"pseudo_input_size = {input_size}")
    logger(f"bkg_th = {dino_cfg['bkg_th']}")

    rows = []
    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    try:
        for item in tqdm(items, desc="cache pseudo"):
            dataset = item["dataset"]
            stem = item["stem"]
            out_dir = cache_root / dataset
            ensure_dir(out_dir)
            out_path = out_dir / f"{stem}.pt"
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"Cache exists; pass --overwrite to regenerate: {out_path}")

            inputs, original_size = preprocess_image(item["image_path"], input_size)
            inputs = inputs.to(device)
            key_holder["tensor"] = None
            with torch.no_grad():
                outputs = call_dino(model, inputs)
                if outputs.attentions is None:
                    raise RuntimeError("DINO did not return attentions.")
                attn = outputs.attentions[-1]
                key = key_holder["tensor"]
                if key is None:
                    raise RuntimeError("DINO key hook did not capture a tensor.")
                _, n_tokens, channels = key.shape
                grid = int(math.sqrt(n_tokens - 1))
                if grid * grid != n_tokens - 1:
                    raise RuntimeError(f"Non-square DINO patch token count: {n_tokens - 1}")
                num_heads = attn.shape[1]
                bkg_mask, _ = compute_img_bkg_seg(
                    attentions=attn,
                    feats=key,
                    featmap_dims=(grid, grid),
                    th_bkg=float(dino_cfg["bkg_th"]),
                    dim=channels // num_heads,
                )
                # fixed pseudo 是 foreground，因此取 background mask 的反。
                pseudo = (1 - bkg_mask).detach().cpu()
                pseudo = refine_post_process(pseudo, area_threshold=4).float()

            payload = {
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "original_size": original_size,
                "tensor": pseudo,
            }
            torch.save(payload, out_path)
            rows.append({
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "cache_path": str(out_path.resolve()),
                "shape": list(pseudo.shape),
            })
    finally:
        handle.remove()

    write_jsonl(manifest_path, rows)
    logger(f"wrote_manifest = {manifest_path}")
    logger(f"num_items = {len(rows)}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate fixed DINO pseudo label cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_pseudo_cache(cfg, overwrite=args.overwrite, logger=print)


if __name__ == "__main__":
    main()
