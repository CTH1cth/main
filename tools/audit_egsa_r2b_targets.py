import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.ecst import (  # noqa: E402
    TemporalTeacherMemory,
    build_ecst_evidence_states,
)
from common.source_arbiter import (  # noqa: E402
    RouteTrajectoryMemory,
    build_directional_utility_target,
    evaluate_r2b_target_audit_v2_admission,
    summarize_r2b_target_audit_branch_v2,
)
from common.utils import (  # noqa: E402
    check_dabe_pu_cache,
    check_hflip_feature_cache,
    dabe_pu_manifest_path,
    ensure_dir,
    feature_manifest_path,
    hflip_feature_cache_manifest_path,
    load_config,
    set_seed,
    torch_load,
)
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    extract_logits,
    forward_seg_head,
    infer_checkpoint_source_arbiter_mode,
    make_hflip_image_68,
    make_hflip_model_input,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / (
        "dinov1_s8_dabepu_v11_egsa_r2b_signaware_dagp_uncgate_ndr_"
        "stage20_lrfloor_2e5.py"
    )
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "workdir"
    / "13-egsa_r1-long45"
    / "train"
    / "ckpt"
    / "epoch_006.pth"
)
DEFAULT_OUT = (
    PROJECT_ROOT
    / "workdir"
    / (
        "dinov1_s8_dabepu_v11_egsa_r2b_signaware_dagp_uncgate_ndr_"
        "stage20_lrfloor_2e5"
    )
    / "audit_r2b_v2"
)
LEGACY_V1_OUT = PROJECT_ROOT / "analysis" / "egsa_r2b_target_audit"
EXPECTED_CHECKPOINT_SHA256 = (
    "ef8f445aa6f440d0540e8546785b01cd22018092b3557f4a3cb4dffe0065dca5"
)
EXPECTED_R1_EXP_NAME = (
    "dinov1_s8_dabepu_v11_ecst_egsa_r1_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5"
)
SOURCES = ("TR-CAMO", "TR-COD10K")
AUDIT_EPOCH = 7


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only EGSA-R2b directional utility target audit from the "
            "canonical EGSA-R1 epoch6 checkpoint (Audit v2)."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ckpt", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="-1 audits all 4040 samples; a positive value is diagnostic only.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="-1 uses cfg.NUM_WORKERS.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or an explicit CUDA device.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def resolve_existing(path_value, label):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_output(path_value):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise RuntimeError(
            f"Audit output must stay inside {PROJECT_ROOT}, got {path}."
        ) from exc
    return path


def resolve_cfg_path(path_value):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def tensor_mapping_digest(mapping):
    digest = hashlib.sha256()
    for name in sorted(mapping):
        value = mapping[name]
        digest.update(str(name).encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def resolve_device(value):
    value = str(value).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {value}")
    return device


def validate_config(cfg):
    expected = {
        "SOURCE_ARBITER_MODE": "sign_aware_pure_loss_space",
        "SOURCE_ARBITER_UTILITY_MODE": "directional_gradient_alignment",
        "SOURCE_ARBITER_UTILITY_CLASS_BALANCE": (
            "audit_v2_fixed_sqrt_inverse"
        ),
        "SOURCE_ARBITER_TARGET_AUDIT_SCHEMA": "egsa_r2b_target_audit_v2",
        "DABE_PU_VERSION": "pu_v11",
        "TEACHER_TARGET_MODE": "binary",
        "DABE_PU_STATIC_TARGET_MODE": "soft",
    }
    for name, wanted in expected.items():
        actual = getattr(cfg, name, None)
        if str(actual).lower() != str(wanted).lower():
            raise RuntimeError(
                f"R2b target audit config mismatch: {name}={actual!r}, "
                f"expected={wanted!r}."
            )
    required_true = (
        "USE_SOURCE_ARBITER",
        "USE_ECST",
        "USE_DABE_PU",
        "SOURCE_ARBITER_USE_STRUCTURED_EVIDENCE",
        "SOURCE_ARBITER_UTILITY_SEPARATE_SIGN_BRANCHES",
    )
    disabled = [name for name in required_true if not bool(getattr(cfg, name, False))]
    if disabled:
        raise RuntimeError(f"R2b target audit required flags disabled: {disabled}")
    if bool(getattr(cfg, "SOURCE_ARBITER_USE_ECST_TEACHER_MAP", True)):
        raise RuntimeError("R2b target audit requires raw binary teacher source.")


def validate_checkpoint(checkpoint, checkpoint_path):
    required = {
        "student",
        "teacher",
        "utility_evaluator",
        "route_trajectory_memory",
        "ecst_temporal_memory",
        "source_arbiter",
        "source_arbiter_optimizer",
        "checkpoint_phase",
        "source_arbiter_lifecycle",
        "config",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise RuntimeError(
            f"Canonical R1 checkpoint missing audit state: {missing}"
        )
    if int(checkpoint.get("epoch", -1)) != 6:
        raise RuntimeError(
            f"R2b target audit requires epoch6, got {checkpoint.get('epoch')!r}."
        )
    if str(checkpoint.get("checkpoint_phase")) != "active_pre_reset":
        raise RuntimeError(
            "R2b target audit requires active_pre_reset checkpoint phase."
        )
    if infer_checkpoint_source_arbiter_mode(checkpoint) != "residual_over_ecst":
        raise RuntimeError("R2b target audit source must be canonical EGSA-R1.")
    config = checkpoint["config"]
    if (
        not isinstance(config, dict)
        or config.get("EXP_NAME") != EXPECTED_R1_EXP_NAME
    ):
        raise RuntimeError(
            "R2b target audit checkpoint config identity mismatch: "
            f"{config.get('EXP_NAME') if isinstance(config, dict) else config!r}."
        )
    lifecycle = checkpoint["source_arbiter_lifecycle"]
    for field in (
        "route_memory_active",
        "utility_evaluator_active",
        "ecst_memory_active",
    ):
        if not isinstance(lifecycle, dict) or lifecycle.get(field) is not True:
            raise RuntimeError(
                f"R2b target audit requires lifecycle {field}=True."
            )
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Canonical R1 epoch6 SHA256 mismatch: "
            f"{actual_sha} != {EXPECTED_CHECKPOINT_SHA256}."
        )
    return actual_sha


def build_memory(memory_cls, state):
    memory = memory_cls(
        num_samples=int(state["num_samples"]),
        height=int(state["height"]),
        width=int(state["width"]),
        dtype=str(state["dtype"]),
    )
    memory.load_state_dict(state)
    return memory


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path, value):
    def scalar(item):
        if item is None:
            return "null"
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, (int, float)):
            return repr(item)
        return json.dumps(str(item), ensure_ascii=False)

    def emit(item, indent=0):
        prefix = " " * indent
        lines = []
        if isinstance(item, dict):
            for key, child in item.items():
                if isinstance(child, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(emit(child, indent + 2))
                else:
                    lines.append(f"{prefix}{key}: {scalar(child)}")
        elif isinstance(item, list):
            for child in item:
                if isinstance(child, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.extend(emit(child, indent + 2))
                else:
                    lines.append(f"{prefix}- {scalar(child)}")
        else:
            lines.append(f"{prefix}{scalar(item)}")
        return lines

    Path(path).write_text("\n".join(emit(value)) + "\n", encoding="utf-8")


def cache_hashes(cfg):
    paths = {
        "normal_feature_manifest": resolve_cfg_path(
            feature_manifest_path(cfg, "train")
        ),
        "hflip_feature_manifest": resolve_cfg_path(
            hflip_feature_cache_manifest_path(cfg)
        ),
        "dabe_pu_manifest": resolve_cfg_path(dabe_pu_manifest_path(cfg)),
    }
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Audit cache manifest missing: {path}")
        result[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return result


def main():
    args = parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max-samples must be -1 or positive.")
    if args.num_workers < -1:
        raise ValueError("--num-workers must be -1 or non-negative.")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive.")

    config_path = resolve_existing(args.config, "config")
    checkpoint_path = resolve_existing(args.ckpt, "checkpoint")
    out_dir = resolve_output(args.out)
    if out_dir == LEGACY_V1_OUT.resolve():
        raise RuntimeError(
            "Audit v2 refuses to overwrite the legacy Audit v1 output: "
            f"{out_dir}"
        )
    cfg = load_config(str(config_path))
    validate_config(cfg)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("R2b audit checkpoint must be a dictionary.")
    checkpoint_sha = validate_checkpoint(checkpoint, checkpoint_path)

    sample_limit = None if args.max_samples < 0 else int(args.max_samples)
    check_dabe_pu_cache(cfg, max_samples=sample_limit)
    check_hflip_feature_cache(cfg, max_samples=sample_limit)
    dataset = CachedTrainDataset(
        cfg,
        max_samples=-1 if sample_limit is None else sample_limit,
    )
    if len(dataset) <= 0:
        raise RuntimeError("R2b target audit dataset is empty.")
    batch_size = int(args.batch_size or cfg.BATCH_SIZE)
    num_workers = int(cfg.NUM_WORKERS if args.num_workers < 0 else args.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    student = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher = build_seg_head(dataset.in_channels, cfg).to(device)
    utility_evaluator = build_seg_head(dataset.in_channels, cfg).to(device)
    student.load_state_dict(checkpoint["student"], strict=True)
    teacher.load_state_dict(checkpoint["teacher"], strict=True)
    utility_evaluator.load_state_dict(
        checkpoint["utility_evaluator"], strict=True
    )
    for module in (student, teacher, utility_evaluator):
        module.eval()
        set_model_epoch(module, AUDIT_EPOCH)
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    route_memory = build_memory(
        RouteTrajectoryMemory,
        checkpoint["route_trajectory_memory"],
    )
    temporal_memory = build_memory(
        TemporalTeacherMemory,
        checkpoint["ecst_temporal_memory"],
    )
    before = {
        "student": tensor_mapping_digest(student.state_dict()),
        "teacher": tensor_mapping_digest(teacher.state_dict()),
        "utility_evaluator": tensor_mapping_digest(
            utility_evaluator.state_dict()
        ),
        "route_memory": tensor_mapping_digest(route_memory.state_dict()),
        "temporal_memory": tensor_mapping_digest(
            temporal_memory.state_dict()
        ),
    }

    batch_rows = []
    image_rows = []
    processed_images = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            leaked_gt = sorted(
                name
                for name in batch
                if str(name).lower() in {"gt", "mask", "ground_truth"}
            )
            if leaked_gt:
                raise RuntimeError(
                    f"R2b target audit batch leaked GT fields: {leaked_gt}"
                )
            sample_indices = batch["sample_index"].long()
            model_input = make_model_input(cfg, batch, device)
            image_68 = make_image_68(cfg, batch, device)
            hflip_input = make_hflip_model_input(cfg, batch, device)
            hflip_image_68 = make_hflip_image_68(cfg, batch, device)

            teacher_prob = resize_logits_for_loss(
                extract_logits(
                    forward_seg_head(
                        teacher,
                        model_input,
                        cfg,
                        image_68=image_68,
                        return_aux=False,
                    )
                ),
                cfg,
            ).sigmoid()
            evaluator_weak = resize_logits_for_loss(
                extract_logits(
                    forward_seg_head(
                        utility_evaluator,
                        model_input,
                        cfg,
                        image_68=image_68,
                        return_aux=False,
                    )
                ),
                cfg,
            ).sigmoid()
            evaluator_flip = torch.flip(
                resize_logits_for_loss(
                    extract_logits(
                        forward_seg_head(
                            utility_evaluator,
                            hflip_input,
                            cfg,
                            image_68=hflip_image_68,
                            return_aux=False,
                        )
                    ),
                    cfg,
                ).sigmoid(),
                dims=[-1],
            )

            old_route = route_memory.fetch(sample_indices, device)
            temporal_mean, temporal_second, history_count = (
                temporal_memory.fetch(sample_indices, device)
            )
            states = build_ecst_evidence_states(
                cfg=cfg,
                batch=batch,
                teacher_prob=teacher_prob.detach(),
                temporal_mean=temporal_mean,
                temporal_second=temporal_second,
                history_count=history_count,
                device=device,
            )
            target = build_directional_utility_target(
                old_teacher_prob=old_route["teacher_prob"],
                old_student_prob=old_route["student_prob"],
                evaluator_prob_weak=evaluator_weak.detach(),
                evaluator_prob_flip=evaluator_flip.detach(),
                old_history_count=old_route["history_count"],
                old_epoch=old_route["epoch"],
                current_epoch=AUDIT_EPOCH,
                old_valid=old_route["valid"],
                dabe_target=batch["pu_target_soft"].to(
                    device, non_blocking=True
                ).float(),
                dabe_weight=batch["pu_weight_map"].to(
                    device, non_blocking=True
                ).float(),
                masks=states["masks"],
                dino_margin=states["margin_68"],
                prototype_valid=states["margin_stats"]["ecst_proto_valid"],
                cfg=cfg,
            )

            batch_counts = defaultdict(
                int,
                {
                    f"{source}_{branch}": 0
                    for source in SOURCES
                    for branch in ("positive", "negative")
                },
            )
            for local_index, source in enumerate(batch["dataset"]):
                source = str(source)
                if source not in SOURCES:
                    raise RuntimeError(
                        f"Unexpected training source in R2b audit: {source!r}."
                    )
                batch_counts[f"{source}_images"] += 1
                image_row = {
                    "sample_index": int(sample_indices[local_index].item()),
                    "dataset": source,
                    "stem": str(batch["stem"][local_index]),
                }
                for branch_name, branch_mask in (
                    ("positive", target.positive_valid),
                    ("negative", target.negative_valid),
                ):
                    mask = branch_mask[local_index]
                    selected = target.target_teacher[local_index][mask]
                    valid_pixels = int(mask.sum().item())
                    image_row[f"{branch_name}_valid"] = valid_pixels
                    image_row[f"{branch_name}_teacher_preferred"] = int(
                        (selected > 0.5).sum().item()
                    )
                    image_row[f"{branch_name}_dabe_preferred"] = int(
                        (selected < 0.5).sum().item()
                    )
                    image_row[f"{branch_name}_soft_teacher_mass"] = float(
                        selected.sum().item()
                    )
                image_rows.append(image_row)
                batch_counts[f"{source}_positive"] += int(
                    target.positive_valid[local_index].sum().item()
                )
                batch_counts[f"{source}_negative"] += int(
                    target.negative_valid[local_index].sum().item()
                )
            processed_images += int(sample_indices.numel())
            batch_rows.append(
                {
                    "batch_index": int(batch_index),
                    "images": int(sample_indices.numel()),
                    "sample_index_min": int(sample_indices.min().item()),
                    "sample_index_max": int(sample_indices.max().item()),
                    "positive_valid_pixels": int(
                        target.positive_valid.sum().item()
                    ),
                    "negative_valid_pixels": int(
                        target.negative_valid.sum().item()
                    ),
                    "positive_active": int(
                        bool(target.positive_valid.any().item())
                    ),
                    "negative_active": int(
                        bool(target.negative_valid.any().item())
                    ),
                    **{
                        name: int(value)
                        for name, value in sorted(batch_counts.items())
                    },
                }
            )
            if (batch_index + 1) % int(args.progress_every) == 0:
                print(
                    f"[R2b TargetAudit] batches={batch_index + 1}/"
                    f"{len(loader)} | images={processed_images}",
                    flush=True,
                )

    after = {
        "student": tensor_mapping_digest(student.state_dict()),
        "teacher": tensor_mapping_digest(teacher.state_dict()),
        "utility_evaluator": tensor_mapping_digest(
            utility_evaluator.state_dict()
        ),
        "route_memory": tensor_mapping_digest(route_memory.state_dict()),
        "temporal_memory": tensor_mapping_digest(
            temporal_memory.state_dict()
        ),
    }
    changed = sorted(name for name in before if before[name] != after[name])
    if changed:
        raise RuntimeError(
            f"R2b target audit mutated frozen state: {changed}"
        )
    if any(parameter.grad is not None for module in (student, teacher, utility_evaluator) for parameter in module.parameters()):
        raise RuntimeError("R2b target audit unexpectedly populated gradients.")

    def summarize_rows(selected_rows, branch_name, source_name):
        observations = [
            {
                "valid_pixels": row[f"{branch_name}_valid"],
                "teacher_preferred_pixels": row[
                    f"{branch_name}_teacher_preferred"
                ],
                "dabe_preferred_pixels": row[
                    f"{branch_name}_dabe_preferred"
                ],
                "soft_teacher_mass": row[
                    f"{branch_name}_soft_teacher_mass"
                ],
            }
            for row in selected_rows
        ]
        if source_name == "ALL":
            relevant_batches = batch_rows
            active_batches = sum(
                int(row[f"{branch_name}_active"] > 0) for row in batch_rows
            )
        else:
            relevant_batches = [
                row
                for row in batch_rows
                if int(row.get(f"{source_name}_images", 0)) > 0
            ]
            active_batches = sum(
                int(row.get(f"{source_name}_{branch_name}", 0) > 0)
                for row in relevant_batches
            )
        return summarize_r2b_target_audit_branch_v2(
            observations,
            active_batches=active_batches,
            total_batches=len(relevant_batches),
            weight_min=float(
                getattr(cfg, "SOURCE_ARBITER_UTILITY_CLASS_WEIGHT_MIN", 0.5)
            ),
            weight_max=float(
                getattr(cfg, "SOURCE_ARBITER_UTILITY_CLASS_WEIGHT_MAX", 4.0)
            ),
        )

    finalized = {}
    rows = []
    for source_name in ("ALL", *SOURCES):
        selected_rows = (
            image_rows
            if source_name == "ALL"
            else [row for row in image_rows if row["dataset"] == source_name]
        )
        finalized[source_name] = {"images": len(selected_rows)}
        for branch_name in ("positive", "negative"):
            branch_summary = summarize_rows(
                selected_rows,
                branch_name,
                source_name,
            )
            finalized[source_name][branch_name] = branch_summary
            rows.append(
                {
                    "source": source_name,
                    "branch": branch_name,
                    **branch_summary,
                }
            )

    all_branches = {
        branch_name: finalized["ALL"][branch_name]
        for branch_name in ("positive", "negative")
    }
    for branch_name in ("positive", "negative"):
        minority_class = all_branches[branch_name]["hard_minority_class"]
        for row in image_rows:
            row[f"{branch_name}_hard_minority"] = row[
                f"{branch_name}_{minority_class}_preferred"
            ]
    full_audit = int(args.max_samples) < 0 and len(dataset) == 4040
    diagnostic_only = int(args.max_samples) != -1
    admission = evaluate_r2b_target_audit_v2_admission(
        all_branches,
        cfg,
        full_audit=full_audit,
        processed_images=processed_images,
        expected_images=4040,
        diagnostic_only=diagnostic_only,
    )
    failures = admission["failures"]
    passed = admission["passed"]
    suggested_class_weights = {
        branch_name: {
            "teacher": float(all_branches[branch_name]["teacher_class_weight"]),
            "dabe": float(all_branches[branch_name]["dabe_class_weight"]),
        }
        for branch_name in ("positive", "negative")
    }

    ensure_dir(out_dir)
    hashes = cache_hashes(cfg)
    summary = {
        "schema_version": "egsa_r2b_target_audit_v2",
        "passed": bool(passed),
        "mini_run_authorized": bool(admission["mini_run_authorized"]),
        "stage20_authorized": False,
        "diagnostic_only": diagnostic_only,
        "failures": failures,
        "audit_epoch": AUDIT_EPOCH,
        "full_audit": full_audit,
        "processed_images": int(processed_images),
        "teacher_source": "checkpoint_teacher_epoch6",
        "future_proxy": "checkpoint_utility_evaluator_normal_hflip_consensus",
        "optimizer_step": False,
        "backward": False,
        "memory_update": False,
        "ema_update": False,
        "gt_used": False,
        "source_checkpoint": {
            "path": str(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "sha256": checkpoint_sha,
            "config_exp_name": checkpoint["config"]["EXP_NAME"],
            "checkpoint_phase": checkpoint["checkpoint_phase"],
        },
        "audit_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "exp_name": cfg.EXP_NAME,
        },
        "cache_manifests": hashes,
        "criteria": admission["criteria"],
        "condition_results": admission["condition_results"],
        "branches": all_branches,
        "by_source": finalized,
        "suggested_class_weights": suggested_class_weights,
        "state_digests_before": before,
        "state_digests_after": after,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(out_dir / "branch_stats.csv", rows)
    write_csv(out_dir / "batch_stats.csv", batch_rows)
    write_csv(out_dir / "image_stats.csv", image_rows)
    write_yaml(
        out_dir / "audit_config.yaml",
        {
            "schema_version": "egsa_r2b_target_audit_v2",
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "max_samples": int(args.max_samples),
            "processed_images": int(processed_images),
            "full_audit": full_audit,
            "diagnostic_only": diagnostic_only,
            "criteria": admission["criteria"],
        },
    )
    write_yaml(
        out_dir / "suggested_class_weights.yaml",
        {
            "schema_version": "egsa_r2b_target_audit_v2",
            "weight_source": "full_audit_sqrt_inverse_clipped",
            "weight_min": float(
                getattr(cfg, "SOURCE_ARBITER_UTILITY_CLASS_WEIGHT_MIN", 0.5)
            ),
            "weight_max": float(
                getattr(cfg, "SOURCE_ARBITER_UTILITY_CLASS_WEIGHT_MAX", 4.0)
            ),
            "positive_teacher_class_weight": suggested_class_weights[
                "positive"
            ]["teacher"],
            "positive_dabe_class_weight": suggested_class_weights["positive"][
                "dabe"
            ],
            "negative_teacher_class_weight": suggested_class_weights[
                "negative"
            ]["teacher"],
            "negative_dabe_class_weight": suggested_class_weights["negative"][
                "dabe"
            ],
        },
    )

    lines = [
        "# EGSA-R2b Directional Target Audit v2",
        "",
        f"- Full 4040-sample audit: `{full_audit}`",
        f"- Processed images: `{processed_images}`",
        f"- Source checkpoint: `{checkpoint_path}`",
        f"- Source SHA256: `{checkpoint_sha}`",
        "- Teacher/evaluator/memory source: checkpoint state; no reset or update.",
        "- GT used: `False`",
        f"- Diagnostic only: `{diagnostic_only}`",
        f"- Overall Passed: `{passed}`",
        f"- Mini-run authorized: `{admission['mini_run_authorized']}`",
        "- Stage20 authorized: `False`",
        "",
        "## Branch Distribution",
        "",
        "| Branch | Valid | Teacher preferred | DABE preferred | Hard minority | Soft minority mass | Minority images | Active batches | Top1% share | Top10% share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for branch_name in ("positive", "negative"):
        values = all_branches[branch_name]
        lines.append(
            f"| {branch_name} | {values['valid_pixels']} | "
            f"{values['teacher_preferred_pixels']} | "
            f"{values['dabe_preferred_pixels']} | "
            f"{values['hard_minority_pixels']} "
            f"({values['hard_minority_ratio']:.6f}) | "
            f"{values['soft_minority_mass']:.6f} | "
            f"{values['minority_valid_images']} | "
            f"{values['active_batches']}/{values['total_batches']} "
            f"({values['active_batch_ratio']:.6f}) | "
            f"{values['top_1_percent_images_minority_share']:.6f} | "
            f"{values['top_10_percent_images_minority_share']:.6f} |"
        )
    lines.extend(["", "## Branch Conditions", ""])
    for branch_name in ("positive", "negative"):
        lines.extend(
            [
                f"### {branch_name.capitalize()}",
                "",
                "| Condition | Actual | Rule | Result |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for field, condition in admission["condition_results"][
            branch_name
        ].items():
            lines.append(
                f"| {field} | {condition['actual']:.8f} | "
                f"{condition['operator']} {condition['threshold']:.8f} | "
                f"{'PASS' if condition['passed'] else 'FAIL'} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Admission",
            "",
            f"- Failures: `{failures or ['none']}`",
            "- A limited `--max-samples` run is diagnostic only and cannot "
            "authorize any training.",
            "- Audit v2 can authorize only the epoch7-8 mini-run; it never "
            "directly authorizes R2b-20.",
            "",
            "## Suggested Fixed Class Weights",
            "",
            f"- positive teacher: `{suggested_class_weights['positive']['teacher']:.10f}`",
            f"- positive DABE: `{suggested_class_weights['positive']['dabe']:.10f}`",
            f"- negative teacher: `{suggested_class_weights['negative']['teacher']:.10f}`",
            f"- negative DABE: `{suggested_class_weights['negative']['dabe']:.10f}`",
        ]
    )
    (out_dir / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        f"[R2b TargetAudit v2] passed={passed} | images={processed_images} | "
        f"summary={out_dir / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
