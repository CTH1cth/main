import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.utils import load_config, set_seed  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import build_pa_dagp_aux_loss, set_model_epoch  # noqa: E402


BASELINE_CONFIG = (
    "configs/dinov1_s8_dabepu_v11_dagp_uncgate_csd_v1r_"
    "rast_v12_esa_asym_clean35_lrfloor_2e5.py"
)
CONTROL_CONFIG = (
    "configs/dinov1_s8_dabepu_v11_dagp_uncgate_csd_v1r_"
    "rast_v12_noesa_clean35_lrfloor_2e5.py"
)


def compare_shared_state(reference, candidate, tolerance=0.0):
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    missing = sorted(set(reference_state) - set(candidate_state))
    if missing:
        raise RuntimeError(f"Candidate model is missing shared state entries: {missing}")
    extra = sorted(set(candidate_state) - set(reference_state))
    unexpected = [key for key in extra if not key.startswith("coarse_path.pa_dagp.")]
    if unexpected:
        raise RuntimeError(f"Unexpected non-PA state entries: {unexpected}")

    max_difference = 0.0
    max_key = None
    for key, reference_value in reference_state.items():
        candidate_value = candidate_state[key]
        if candidate_value.shape != reference_value.shape:
            raise RuntimeError(
                f"Shared state shape mismatch for {key}: "
                f"{list(reference_value.shape)} vs {list(candidate_value.shape)}"
            )
        difference = float((reference_value - candidate_value).abs().max().item())
        if difference > max_difference:
            max_difference = difference
            max_key = key
    print(f"shared_state_max_abs_diff={max_difference:.12g} | key={max_key}")
    if max_difference > tolerance:
        raise RuntimeError(
            "PA-DAGP changed existing parameter initialization: "
            f"key={max_key}, max_abs_diff={max_difference}"
        )


def compare_outputs(reference, candidate, keys, tolerance=1e-6):
    for key in keys:
        difference = float((reference[key] - candidate[key]).abs().max().item())
        print(f"{key}: max_abs_diff={difference:.12g}")
        if difference > tolerance:
            raise RuntimeError(f"PA-DAGP equivalence failed for {key}: {difference}")


def pa_gradient_sum(model):
    values = []
    for parameter in model.coarse_path.pa_dagp.parameters():
        if parameter.grad is not None:
            values.append(parameter.grad.detach().abs().sum())
    if not values:
        return 0.0
    return float(torch.stack(values).sum().item())


def main():
    parser = argparse.ArgumentParser(description="Verify PA-DAGP-v1 equivalence, schedule, and gradients.")
    parser.add_argument(
        "--config",
        default="configs/dinov1_s8_dabepu_v11_pa_dagp_v1_csd_v1r_rast_v12_noesa_clean35_lrfloor_2e5.py",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    baseline_cfg = load_config(BASELINE_CONFIG)
    control_cfg = load_config(CONTROL_CONFIG)
    pa_cfg = load_config(args.config)
    set_seed(1234)
    baseline = build_seg_head(384, baseline_cfg).to(device)
    baseline_rng_state = torch.get_rng_state().clone()
    set_seed(1234)
    control = build_seg_head(384, control_cfg).to(device)
    set_seed(1234)
    pa_model = build_seg_head(384, pa_cfg).to(device)
    pa_rng_state = torch.get_rng_state().clone()
    print("[0] Same-seed independent construction preserves all existing state")
    compare_shared_state(baseline, control)
    compare_shared_state(control, pa_model)
    if not torch.equal(baseline_rng_state, pa_rng_state):
        raise RuntimeError("PA-DAGP model construction changed the caller's RNG trajectory.")
    print("post_build_rng_state_equal=True")
    if control.coarse_path.pa_dagp is not None:
        raise RuntimeError("USE_PA_DAGP=False unexpectedly instantiated polarity parameters.")
    if pa_model.coarse_path.pa_dagp is None:
        raise RuntimeError("USE_PA_DAGP=True did not instantiate PolarityEdgeGateV1.")

    set_seed(5678)
    feature = torch.randn(2, 384, 37, 37, device=device)
    image = torch.rand(2, 3, 68, 68, device=device)
    bg_reliable = (torch.rand(2, 1, 68, 68, device=device) > 0.6).float()
    batch = {
        "pu_fg_core": (torch.rand(2, 1, 68, 68) > 0.80).float(),
        "pu_bg_core": (torch.rand(2, 1, 68, 68) > 0.45).float(),
    }
    keys = ("base_logits", "coarse_logits_native", "coarse_logits_68", "final_logits")

    print(f"device={device}")
    print("[1] PA disabled: original baseline vs no-ESA control")
    for epoch in (1, 7, 15):
        set_model_epoch(baseline, epoch)
        set_model_epoch(control, epoch)
        with torch.no_grad():
            baseline_out = baseline(
                feature, image_68=image, return_aux=True, bg_reliable_68=bg_reliable
            )
            control_out = control(
                feature, image_68=image, return_aux=True, bg_reliable_68=bg_reliable
            )
        print(f"epoch={epoch}")
        compare_outputs(baseline_out, control_out, keys)

    print("[2] PA enabled at epoch1: logits equivalent, auxiliary weight zero")
    set_model_epoch(control, 1)
    set_model_epoch(pa_model, 1)
    with torch.no_grad():
        control_out = control(
            feature, image_68=image, return_aux=True, bg_reliable_68=bg_reliable
        )
    pa_out = pa_model(
        feature,
        image_68=image,
        return_aux=True,
        bg_reliable_68=bg_reliable,
        pa_compare_original=True,
    )
    compare_outputs(control_out, pa_out, keys)
    loss_pa, stats = build_pa_dagp_aux_loss(pa_cfg, 1, pa_out, batch, device)
    if float(loss_pa.detach().item()) != 0.0 or stats["edge_scale"] != 0.0 or stats["aux_scale"] != 0.0:
        raise RuntimeError(f"epoch1 PA schedule/loss mismatch: loss={loss_pa}, stats={stats}")
    pa_model.zero_grad(set_to_none=True)
    (F.binary_cross_entropy_with_logits(pa_out["final_logits"], torch.zeros_like(pa_out["final_logits"])) + loss_pa).backward()
    if any(parameter.grad is not None for parameter in pa_model.coarse_path.pa_dagp.parameters()):
        raise RuntimeError("PA parameters received gradients before epoch7.")

    print("[3] PA enabled at epoch15: gate/normalization/auxiliary gradients")
    set_model_epoch(pa_model, 15)
    pa_model.zero_grad(set_to_none=True)
    pa_out = pa_model(
        feature,
        image_68=image,
        return_aux=True,
        bg_reliable_68=bg_reliable,
        pa_compare_original=True,
    )
    loss_pa, stats = build_pa_dagp_aux_loss(pa_cfg, 15, pa_out, batch, device)
    if stats["edge_scale"] != 1.0 or stats["aux_scale"] != 1.0:
        raise RuntimeError(f"epoch15 PA scale mismatch: {stats}")
    if stats["gate_min"] < 0.5 - 1e-5 or stats["gate_max"] > 1.0 + 1e-5:
        raise RuntimeError(f"epoch15 PA gate range mismatch: {stats['gate_min']}, {stats['gate_max']}")
    if stats.get("edge_normalization_max_error", 1.0) > 1e-5:
        raise RuntimeError(f"epoch15 PA edge normalization failed: {stats}")
    (pa_out["final_logits"].mean() + loss_pa).backward()
    pol_grad = pa_model.coarse_path.pa_dagp.pol_proj[0].weight.grad
    calib_grad = pa_model.coarse_path.pa_dagp.calib_head[-1].weight.grad
    if pol_grad is None or float(pol_grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("PA polarity projection did not receive an epoch15 gradient.")
    if calib_grad is None or float(calib_grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("PA calibration output layer did not receive an epoch15 gradient.")

    print("[4] PA enabled at epoch21: auxiliary off, segmentation gradient remains")
    with torch.no_grad():
        torch.manual_seed(99)
        pa_model.coarse_path.graph_pred.weight.normal_(mean=0.0, std=0.02)
        pa_model.coarse_path.graph_pred.bias.zero_()
    set_model_epoch(pa_model, 21)
    pa_model.zero_grad(set_to_none=True)
    pa_out = pa_model(
        feature,
        image_68=image,
        return_aux=True,
        bg_reliable_68=bg_reliable,
    )
    loss_pa, stats = build_pa_dagp_aux_loss(pa_cfg, 21, pa_out, batch, device)
    if stats["edge_scale"] != 1.0 or stats["aux_scale"] != 0.0 or float(loss_pa.item()) != 0.0:
        raise RuntimeError(f"epoch21 PA schedule/loss mismatch: loss={loss_pa}, stats={stats}")
    target = (torch.rand_like(pa_out["final_logits"]) > 0.5).float()
    segmentation_loss = F.binary_cross_entropy_with_logits(pa_out["final_logits"], target)
    segmentation_loss.backward()
    segmentation_pa_grad = pa_gradient_sum(pa_model)
    if segmentation_pa_grad <= 0.0:
        raise RuntimeError("PA module did not receive a segmentation gradient at epoch21.")
    print(
        "PA-DAGP-v1 equivalence: PASS | "
        f"epoch15_pol_grad={float(pol_grad.abs().sum().item()):.8g} | "
        f"epoch15_calib_grad={float(calib_grad.abs().sum().item()):.8g} | "
        f"epoch21_seg_pa_grad={segmentation_pa_grad:.8g}"
    )


if __name__ == "__main__":
    main()
