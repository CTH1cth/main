import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedEvalDataset, CachedTrainDataset  # noqa: E402
from common.utils import check_cacd_feature_cache, load_config, set_seed  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    build_cacd_anchor_partial_ce,
    forward_seg_head,
    make_image_68,
    make_model_input,
    make_sobel_68,
)


def grad_norm(parameter):
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm().item())


def build_loss(model, cfg, batch, device):
    model_input = make_model_input(cfg, batch, device)
    output = forward_seg_head(
        model,
        model_input,
        cfg,
        image_68=make_image_68(cfg, batch, device),
        sobel_68=make_sobel_68(cfg, batch, device),
        return_aux=True,
    )
    target = batch["pu_target_soft"].to(device).float()
    weight = batch["pu_weight_map"].to(device).float()

    def weighted_bce(logits):
        loss_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (loss_map * weight).sum() / weight.sum().clamp_min(1e-6)

    loss_final = weighted_bce(output["final_logits"])
    loss_coarse = weighted_bce(output["coarse_logits"])
    loss_base = weighted_bce(output["base_logits"])
    loss_anchor, anchor_stats = build_cacd_anchor_partial_ce(
        output["anchor_logits"], batch, cfg, device=device
    )
    loss = (loss_final + 0.5 * loss_coarse + 0.5 * loss_base) / 2.0
    loss = loss + float(cfg.CACD_ANCHOR_LOSS_WEIGHT) * loss_anchor
    return output, loss, anchor_stats


def main():
    parser = argparse.ArgumentParser(description="CACD-v1-Base cache/model/two-step-gradient sanity.")
    parser.add_argument(
        "--config",
        default="configs/dinov1_s8_dabepu_v11_cacd_v1_base_rast_v12_esa_asym_long35_lrfloor_2e5.py",
    )
    parser.add_argument("--cache-root", default="../analysis/cacd_cache_sanity")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.max_samples < 2:
        raise ValueError("CACD sanity requires at least two samples.")
    cfg = load_config(args.config)
    cfg.CACD_EXTRA_FEATURE_CACHE_ROOT = args.cache_root
    set_seed(int(cfg.SEED))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    for split in ("train", "val", "test"):
        _, reason = check_cacd_feature_cache(cfg, split, max_samples=args.max_samples)
        print(f"cache_{split} = {reason}")

    dataset = CachedTrainDataset(cfg, max_samples=args.max_samples)
    batch = next(iter(DataLoader(dataset, batch_size=min(2, args.max_samples), shuffle=False, num_workers=0)))
    forbidden_train_fields = [key for key in batch if key in {"gt", "gt_path"}]
    if forbidden_train_fields:
        raise RuntimeError(f"CACD training batch leaked GT fields: {forbidden_train_fields}")
    for field in ("feature_l10", "feature_l11", "feature"):
        if list(batch[field].shape[1:]) != [384, 37, 37] or batch[field].dtype != torch.float32:
            raise RuntimeError(f"Invalid {field}: shape={list(batch[field].shape)}, dtype={batch[field].dtype}")
    if list(batch["image_68"].shape[1:]) != [3, 68, 68] or list(batch["sobel_68"].shape[1:]) != [1, 68, 68]:
        raise RuntimeError("CACD image_68/sobel_68 dataset contract failed.")

    student = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher.load_state_dict(student.state_dict(), strict=True)
    if list(student.state_dict()) != list(teacher.state_dict()):
        raise RuntimeError("CACD student/teacher state keys differ.")
    max_initial_diff = max(
        (
            float((left - right).abs().max().item())
            for left, right in zip(student.state_dict().values(), teacher.state_dict().values())
            if torch.is_tensor(left) and left.is_floating_point()
        ),
        default=0.0,
    )
    if max_initial_diff != 0.0:
        raise RuntimeError(f"CACD student/teacher initial diff is {max_initial_diff}.")
    forbidden_tokens = ("graph_pred", "ndr_branch", "csd_residual", "pa_dagp", "hr_bfr")
    bad_keys = [key for key in student.state_dict() if any(token in key.lower() for token in forbidden_tokens)]
    if bad_keys:
        raise RuntimeError(f"CACD unexpectedly instantiated old-head parameters: {bad_keys[:10]}")

    relation_final = student.context_reasoner.relation_mlp[-1]
    if float(relation_final.weight.detach().abs().max().item()) != 0.0:
        raise RuntimeError("CACD relation final layer is not zero-initialized.")
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-2)
    student.train()
    optimizer.zero_grad(set_to_none=True)
    output1, loss1, anchor_stats1 = build_loss(student, cfg, batch, device)
    expected_shapes = {
        "base_logits": [batch["feature"].shape[0], 1, 68, 68],
        "coarse_logits": [batch["feature"].shape[0], 1, 68, 68],
        "final_logits": [batch["feature"].shape[0], 1, 68, 68],
        "anchor_logits": [batch["feature"].shape[0], 3, 37, 37],
    }
    for field, expected in expected_shapes.items():
        if list(output1[field].shape) != expected or not bool(torch.isfinite(output1[field]).all().item()):
            raise RuntimeError(f"CACD output {field} invalid: {list(output1[field].shape)} != {expected}")
    aux1 = output1["cacd_aux"]
    if float(aux1["anchor_probability_sum_error"]) > 1e-5:
        raise RuntimeError("CACD anchor probability sum failed.")
    if max(float(aux1["fg_slot_attention_sum_error"]), float(aux1["bg_slot_attention_sum_error"])) > 1e-5:
        raise RuntimeError("CACD slot attention normalization failed.")
    if float(aux1["context_delta_abs_mean"]) != 0.0:
        raise RuntimeError("CACD zero-init relation did not produce exact zero context delta.")
    loss1.backward()
    first_step_grads = {
        "proj10": grad_norm(student.consensus_encoder.proj10[0].weight),
        "proj11": grad_norm(student.consensus_encoder.proj11[0].weight),
        "proj12": grad_norm(student.consensus_encoder.proj12[0].weight),
        "anchor": grad_norm(student.anchor_estimator.anchor_head[-1].weight),
        "detail": grad_norm(student.decoder.detail_encoder[0].weight),
        "relation_final": grad_norm(relation_final.weight),
        "fg_slot_query": grad_norm(student.context_reasoner.fg_slot_queries),
        "bg_slot_query": grad_norm(student.context_reasoner.bg_slot_queries),
    }
    for name in ("proj10", "proj11", "proj12", "anchor", "detail", "relation_final"):
        if first_step_grads[name] <= 0.0:
            raise RuntimeError(f"CACD first-step gradient is zero for {name}: {first_step_grads}")
    relation_before = relation_final.weight.detach().clone()
    optimizer.step()
    if torch.equal(relation_before, relation_final.weight.detach()):
        raise RuntimeError("CACD first optimizer step did not update relation final layer.")

    optimizer.zero_grad(set_to_none=True)
    output2, loss2, _ = build_loss(student, cfg, batch, device)
    loss2.backward()
    second_slot_grads = {
        "fg_slot_query": grad_norm(student.context_reasoner.fg_slot_queries),
        "bg_slot_query": grad_norm(student.context_reasoner.bg_slot_queries),
    }
    if min(second_slot_grads.values()) <= 0.0:
        raise RuntimeError(f"CACD slot queries lack second-step gradients: {second_slot_grads}")

    eval_dataset = CachedEvalDataset(cfg, split="val", max_samples=1)
    eval_sample = eval_dataset[0]
    leaked = [key for key in eval_sample if key.startswith("pu_") or "dabe" in key.lower()]
    if leaked:
        raise RuntimeError(f"CACD eval sample unexpectedly reads DABE fields: {leaked}")
    eval_batch = next(iter(DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)))
    student.eval()
    with torch.no_grad():
        eval_out = forward_seg_head(
            student,
            make_model_input(cfg, eval_batch, device),
            cfg,
            image_68=make_image_68(cfg, eval_batch, device),
            sobel_68=make_sobel_68(cfg, eval_batch, device),
            return_aux=False,
        )
    if list(eval_out["final_logits"].shape) != [1, 1, 68, 68]:
        raise RuntimeError("CACD eval single-forward output shape failed.")

    print(f"device = {device}")
    print(f"train_samples = {len(dataset)}")
    print(f"student_teacher_initial_max_diff = {max_initial_diff:.9g}")
    print(f"loss_step1 = {float(loss1.detach()):.8f}")
    print(f"loss_step2 = {float(loss2.detach()):.8f}")
    print(f"anchor_stats_step1 = {anchor_stats1}")
    print(f"first_step_grad_norms = {first_step_grads}")
    print(f"second_step_slot_grad_norms = {second_slot_grads}")
    print("eval_dabe_input = False")
    print("cacd_sanity = PASS")


if __name__ == "__main__":
    main()
