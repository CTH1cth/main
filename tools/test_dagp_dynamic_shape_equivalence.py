import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.utils import load_config, set_seed  # noqa: E402
from model import build_seg_head  # noqa: E402


BASELINE_CONFIG = (
    "configs/dinov1_s8_dabepu_v11_dagp_uncgate_csd_v1r_"
    "rast_v12_esa_asym_clean35_lrfloor_2e5.py"
)


def main():
    parser = argparse.ArgumentParser(description="Verify DAGP/CSD-v1R dynamic-shape equivalence.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    baseline_cfg = load_config(BASELINE_CONFIG)
    cssd_cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(1234)
    baseline = build_seg_head(384, baseline_cfg).to(device)
    candidate = build_seg_head(384, cssd_cfg).to(device)
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    baseline.train()
    candidate.train()
    feature_37 = torch.randn(2, 384, 37, 37, device=device)
    feature_48 = torch.randn(2, 384, 48, 48, device=device)
    image_68 = torch.rand(2, 3, 68, 68, device=device)
    bg_reliable = (torch.rand(2, 1, 68, 68, device=device) > 0.7).float()
    keys = ("coarse_logits_native", "coarse_logits_68", "base_logits", "final_logits")
    with torch.no_grad():
        for epoch in (1, 7, 15):
            baseline.set_epoch(epoch)
            candidate.set_epoch(epoch)
            old_out = baseline(
                feature_37,
                image_68=image_68,
                return_aux=True,
                bg_reliable_68=bg_reliable,
            )
            new_out = candidate(
                feature_37,
                image_68=image_68,
                return_aux=True,
                bg_reliable_68=bg_reliable,
            )
            for key in keys:
                diff = float((old_out[key] - new_out[key]).abs().max().item())
                print(f"epoch={epoch} | {key} | max_abs_diff={diff:.12g}")
                if diff > 1e-6:
                    raise RuntimeError(f"37x37 equivalence failed at epoch={epoch}, key={key}: {diff}")

        candidate.set_epoch(15)
        high_out = candidate(
            feature_48,
            image_68=image_68,
            return_aux=True,
            bg_reliable_68=bg_reliable,
        )
    expected = {
        "coarse_logits_native": [2, 1, 48, 48],
        "coarse_logits_68": [2, 1, 68, 68],
        "base_logits": [2, 1, 68, 68],
        "final_logits": [2, 1, 68, 68],
    }
    for key, shape in expected.items():
        tensor = high_out[key]
        print(f"48x48 | {key} | shape={list(tensor.shape)} | finite={bool(torch.isfinite(tensor).all())}")
        if list(tensor.shape) != shape or not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(f"48x48 dynamic-shape check failed for {key}")
    print("DAGP/CSD-v1R dynamic-shape equivalence: PASS")


if __name__ == "__main__":
    main()
