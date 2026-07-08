import argparse
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.utils import ensure_dir, load_config, torch_load, write_jsonl  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    dataloader_worker_kwargs,
    extract_logits,
    forward_seg_head,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
)


def infer_in_channels(state):
    if "base_head.weight" in state:
        weight = state["base_head.weight"]
    elif "proj.weight" in state:
        weight = state["proj.weight"]
    elif "base.weight" in state:
        weight = state["base.weight"]
    elif "base_head.proj.weight" in state:
        weight = state["base_head.proj.weight"]
    elif "proj12.conv.weight" in state:
        weight = state["proj12.conv.weight"]
    else:
        raise KeyError("Cannot infer in_channels from checkpoint state.")
    return int(weight.shape[1])


def infer_epoch(checkpoint, ckpt_path):
    if "epoch" in checkpoint:
        try:
            return int(checkpoint["epoch"])
        except (TypeError, ValueError):
            pass
    match = re.search(r"epoch[_-](\d+)", str(ckpt_path))
    if match:
        return int(match.group(1))
    return -1


def load_head_from_checkpoint(cfg, checkpoint, model_key, device):
    if model_key not in checkpoint:
        raise KeyError(f"Checkpoint missing '{model_key}' state.")
    state = checkpoint[model_key]
    model = build_seg_head(infer_in_channels(state), cfg).to(device)
    missing = sorted(set(model.state_dict()) - set(state))
    if str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp_safe" and missing == ["current_epoch_tensor"]:
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys: {result.unexpected_keys}")
        if hasattr(model, "set_epoch"):
            model.set_epoch(int(getattr(cfg, "MAX_EPOCH", 25)))
    else:
        model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def build_cache(args):
    cfg = load_config(args.config)
    # Coverage cache generation must not depend on an already-existing TCE cache.
    setattr(cfg, "USE_TCE", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch_load(args.ckpt, map_location="cpu")
    if checkpoint.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"Checkpoint backbone mismatch: {checkpoint.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    source_epoch = infer_epoch(checkpoint, args.ckpt)
    model = load_head_from_checkpoint(cfg, checkpoint, args.model_for_cache, device)

    dataset = CachedTrainDataset(cfg, max_samples=int(args.max_samples))
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(cfg.BATCH_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
        **dataloader_worker_kwargs(cfg),
    )

    out_root = Path(args.out).expanduser()
    ensure_dir(out_root)
    rows = []
    num_written = 0
    for batch in loader:
        model_input = make_model_input(cfg, batch, device)
        image_68 = make_image_68(cfg, batch, device)
        output = forward_seg_head(model, model_input, cfg, image_68=image_68, return_aux=False)
        logits = resize_logits_for_loss(extract_logits(output), cfg)
        cover_prob = logits.sigmoid().detach().cpu().float()
        cover_binary = (cover_prob >= 0.5).float()
        cover_conf = (2.0 * (cover_prob - 0.5).abs()).clamp(0.0, 1.0)

        for idx, (dataset_name, stem) in enumerate(zip(batch["dataset"], batch["stem"])):
            out_path = out_root / str(dataset_name) / f"{stem}.pt"
            ensure_dir(out_path.parent)
            if out_path.exists() and not args.overwrite:
                payload = torch_load(out_path, map_location="cpu")
                cover_area = float(payload.get("cover_area", payload["cover_binary_68"].float().mean().item()))
            else:
                prob = cover_prob[idx].contiguous()
                binary = cover_binary[idx].contiguous()
                conf = cover_conf[idx].contiguous()
                cover_area = float(binary.mean().item())
                payload = {
                    "cover_prob_68": prob,
                    "cover_binary_68": binary,
                    "cover_conf_68": conf,
                    "cover_area": cover_area,
                    "dataset": str(dataset_name),
                    "stem": str(stem),
                    "backbone_key": cfg.BACKBONE_KEY,
                    "shape_68": list(prob.shape),
                    "source_ckpt": str(Path(args.ckpt).expanduser()),
                    "source_epoch": int(source_epoch),
                    "model_for_cache": str(args.model_for_cache),
                }
                torch.save(payload, out_path)
                num_written += 1
            rows.append(
                {
                    "dataset": str(dataset_name),
                    "stem": str(stem),
                    "cache_path": str(out_path.resolve()),
                    "backbone_key": cfg.BACKBONE_KEY,
                    "shape_68": [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)],
                    "cover_area": cover_area,
                    "source_ckpt": str(Path(args.ckpt).expanduser()),
                    "source_epoch": int(source_epoch),
                    "model_for_cache": str(args.model_for_cache),
                }
            )
            print(
                f"[TCE Cover] {dataset_name}/{stem} | area={cover_area:.6f} | "
                f"path={out_path}",
                flush=True,
            )

    manifest_path = out_root / f"manifest_{args.split}.jsonl"
    write_jsonl(manifest_path, rows)
    print(
        f"[TCE Cover] done | rows={len(rows)} | written={num_written} | manifest={manifest_path}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--model_for_cache", default="student", choices=["student", "teacher"])
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_cache(args)


if __name__ == "__main__":
    main()
