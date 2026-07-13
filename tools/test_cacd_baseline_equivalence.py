import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.utils import load_config, set_seed, torch_load  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    forward_seg_head,
    make_image_68,
    make_model_input,
    set_model_epoch,
)


def pick_output(output, *keys):
    for key in keys:
        if isinstance(output, dict) and key in output:
            return output[key]
    raise KeyError(f"None of {keys} is present in output keys {sorted(output)}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify that adding CACD did not change the original Long35 numerical path."
    )
    parser.add_argument(
        "--config",
        default="configs/dinov1_s8_dabepu_v11_dagp_uncgate_ndr_rast_v12_esa_asym_long35_lrfloor_2e5.py",
    )
    parser.add_argument(
        "--ckpt",
        default="../workdir/11-dabepu-esa_asym_long35_lrfloor_2e5/train/ckpt/epoch_020.pth",
    )
    parser.add_argument(
        "--reference",
        default="../analysis/cacd_baseline_equivalence/reference_epoch020_4samples.pt",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    reference_path = Path(args.reference)
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Pre-modification CACD baseline reference is missing: {reference_path}"
        )
    reference = torch_load(reference_path, map_location="cpu")
    if reference.get("version") != "cacd_baseline_equivalence_reference_v1":
        raise RuntimeError(f"Unexpected baseline reference version: {reference.get('version')!r}")
    cfg = load_config(args.config)
    set_seed(int(cfg.SEED))
    dataset = CachedTrainDataset(cfg, max_samples=4)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
    datasets = list(batch["dataset"])
    stems = list(batch["stem"])
    if datasets != reference["datasets"] or stems != reference["stems"]:
        raise RuntimeError(
            f"Reference sample mismatch: current={list(zip(datasets, stems))}, "
            f"reference={list(zip(reference['datasets'], reference['stems']))}"
        )
    checkpoint = torch_load(args.ckpt, map_location="cpu")
    if int(checkpoint.get("epoch", -1)) != int(reference["checkpoint_epoch"]):
        raise RuntimeError(
            f"Checkpoint epoch mismatch: {checkpoint.get('epoch')} != {reference['checkpoint_epoch']}"
        )
    model = build_seg_head(dataset.in_channels, cfg)
    model.load_state_dict(checkpoint["student"], strict=True)
    set_model_epoch(model, int(checkpoint["epoch"]))
    model.eval()
    device = torch.device("cpu")
    with torch.no_grad():
        output = forward_seg_head(
            model,
            make_model_input(cfg, batch, device),
            cfg,
            image_68=make_image_68(cfg, batch, device),
            return_aux=True,
        )
    current = {
        "base_logits": pick_output(output, "base_logits"),
        "coarse_logits": pick_output(output, "coarse_logits_68", "coarse_logits"),
        "final_logits": pick_output(output, "final_logits", "logits"),
    }
    failed = []
    for name, value in current.items():
        expected = reference[name]
        if value.shape != expected.shape:
            raise RuntimeError(f"{name} shape mismatch: {list(value.shape)} != {list(expected.shape)}")
        difference = (value.detach().cpu() - expected).abs()
        max_diff = float(difference.max().item())
        mean_diff = float(difference.mean().item())
        print(f"{name}_max_abs_diff = {max_diff:.9g}")
        print(f"{name}_mean_abs_diff = {mean_diff:.9g}")
        if max_diff > float(args.tolerance):
            failed.append((name, max_diff))
    if failed:
        raise RuntimeError(
            f"Original Long35 numerical equivalence failed at tolerance {args.tolerance}: {failed}"
        )
    print("baseline_equivalence = PASS")


if __name__ == "__main__":
    main()
