from types import SimpleNamespace
from pathlib import Path

import torch

from common.dabev2hard_static_only import validate_dabev2hard_static_only_config
from common.dataset import resolve_dabe_clean_dabe_v2_hard_threshold
from common.r1hard_linear_pure_student import validate_gbsp_lcic_config
from common.teacher_routing import validate_teacher_routing_config
from common.utils import load_config
from eval import apply_lcic_inference_scales
from model import SimpleConvSegHead, build_seg_head
from train import (
    LeanPureStudentLogger,
    add_lcic_soft_dice_loss,
    apply_step_then_floor_lr,
    apply_lcic_structural_only_freeze,
    build_dabe_clean_static_target,
    build_lcic_dataset_loss_weight,
    build_optimizer_scheduler,
    compute_linear_floor_two_stage_lr,
    dabe_static_bce_with_logits,
    get_step_lr_params,
    gbsp_ssboc_static_loss,
    is_finetune_reset_enabled,
    lcic_soft_dice_loss,
    should_emit_lean_pure_student_log,
    should_step_iter_scheduler,
    validate_lcic_structural_only_resume_checkpoint,
)
from models.lcic import (
    Conv3x3LiteHead,
    DWLiteHead,
    LCICHead,
    SELF_OFFSET_INDEX,
    build_local_dino_affinity,
    local_affinity_propagate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "a_linear": ROOT / "configs/dinov1_s8_gbsp_lcic_a_linear.py",
    "b_consensus": ROOT / "configs/dinov1_s8_gbsp_lcic_b_consensus.py",
    "c_innovation": ROOT / "configs/dinov1_s8_gbsp_lcic_c_innovation.py",
    "d_full": ROOT / "configs/dinov1_s8_gbsp_lcic_d_full.py",
    "e_dwlite": ROOT / "configs/dinov1_s8_gbsp_lcic_e_dwlite.py",
}
DATASET_THRESHOLD_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_lcic_d_full_camo058_cod10k066.py"
)
DATASET_LOSS_WEIGHT_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_lcic_d_full_camow15.py"
)
LR_STEP_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_lcic_d_full_lrstep50.py"
)
SOFT_DICE_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_lcic_d_full_dice005.py"
)
R32_BW1P30_SIGMAF085_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_"
    "bw1p30_sigmaf085.py"
)
R32_SEED2027_STRUCTONLY_E3_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_"
    "seed2027_structonly_e3.py"
)
R32_SEED2027_CONSENSUS_GAIN3_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_"
    "seed2027_consensusgain3.py"
)
R32_SEED2027_ADAPTIVE_GATE_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_"
    "seed2027_adaptivegate.py"
)
R32_SEED2027_SSBOC_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_ssboc_dice005_t050_"
    "seed2027.py"
)
R32_DWLITE16_SEED2027_SSBOC_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_r32_dwlite16_ssboc_dice005_t050_seed2027.py"
)
R32_DWLITE16_SEED2027_SSBOC_LR0003_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_r32_dwlite16_ssboc_dice005_t050_"
    "seed2027_lr0003.py"
)
R32_DWLITE16_SEED2027_SSBOC_LR1E4_FLOOR1E5_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_r32_dwlite16_ssboc_dice005_t050_"
    "seed2027_lr1e4_floor1e5.py"
)
R32_DWLITE16_SEED2027_SSBOC_LR1E4_FLOOR1E5_LINEAR45_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_r32_dwlite16_ssboc_dice005_t050_"
    "seed2027_lr1e4_floor1e5_linear45.py"
)
R32_CONV3X3_C16_NOSSBOC_STEP2FLOOR_CONFIG = ROOT / (
    "configs/dinov1_s8_gbsp_r32_conv3x3_c16_nossboc_"
    "step2floor_dice005_t050_seed2027.py"
)
EXPECTED_PARAMS = {
    "a_linear": 385,
    "b_consensus": 386,
    "c_innovation": 770,
    "d_full": 771,
    "e_dwlite": 6337,
}


def test_local_affinity_rows_self_loop_and_border_mask():
    feature = torch.ones(2, 384, 7, 7)
    affinity, diagnostics = build_local_dino_affinity(
        feature, return_diagnostics=True
    )
    assert affinity.shape == (2, 9, 7, 7)
    assert diagnostics["row_sum_max_abs_error"].item() < 1e-5
    assert diagnostics["self_similarity_min"].item() == 1.0
    assert diagnostics["self_similarity_max"].item() == 1.0
    assert not diagnostics["affinity_requires_grad"]
    assert not diagnostics["dense_affinity_materialized"]
    assert torch.count_nonzero(affinity[0, :, 0, 0]).item() == 4
    assert torch.count_nonzero(affinity[0, :, 3, 3]).item() == 9
    assert affinity[0, SELF_OFFSET_INDEX, 0, 0].item() > 0.0


def test_uniform_feature_has_zero_local_innovation():
    feature = torch.full((2, 384, 7, 7), 0.25)
    affinity = build_local_dino_affinity(feature)
    local_consensus_feature = local_affinity_propagate(affinity, feature)
    local_innovation_feature = feature - local_consensus_feature
    assert local_innovation_feature.abs().max().item() < 1e-5


def test_constant_logits_have_zero_consensus_correction():
    feature = torch.randn(2, 384, 7, 7)
    affinity = build_local_dino_affinity(feature)
    logits = torch.full((2, 1, 7, 7), 2.75)
    propagated = local_affinity_propagate(affinity, logits)
    assert (propagated - logits).abs().max().item() < 1e-5


def test_full_lcic_initialization_equals_seed_matched_linear():
    torch.manual_seed(3407)
    linear = SimpleConvSegHead(384)
    torch.manual_seed(3407)
    full = LCICHead(384, use_consensus=True, use_innovation=True)
    assert torch.equal(linear.proj.weight, full.anchor.weight)
    assert torch.equal(linear.proj.bias, full.anchor.bias)
    feature = torch.randn(2, 384, 7, 7)
    error = (linear(feature) - full(feature)).abs().max().item()
    assert error < 1e-6


def test_full_lcic_gradient_contract_and_detached_graph():
    torch.manual_seed(11)
    feature = torch.randn(2, 384, 7, 7)
    model = LCICHead(384, use_consensus=True, use_innovation=True)
    affinity = build_local_dino_affinity(feature)
    assert not affinity.requires_grad
    output = model(feature)
    output.square().mean().backward()
    assert feature.grad is None
    assert model.anchor.weight.grad is not None
    assert model.innovation_head.weight.grad is not None
    assert model.alpha.grad is not None
    assert model.beta.grad is not None


def test_variant_modules_and_parameter_counts():
    for variant, config_path in CONFIGS.items():
        cfg = load_config(config_path)
        model = build_seg_head(384, cfg)
        params = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        assert params == EXPECTED_PARAMS[variant]
        output = model(torch.randn(2, 384, 7, 7))
        assert output.shape == (2, 1, 7, 7)
    assert isinstance(
        build_seg_head(384, SimpleNamespace(HEAD_TYPE="lcic_dwlite")),
        DWLiteHead,
    )


def test_all_lcic_configs_pass_decoder_only_protocol_audit():
    for variant, config_path in CONFIGS.items():
        cfg = load_config(config_path)
        report = validate_gbsp_lcic_config(cfg)
        static_report = validate_dabev2hard_static_only_config(cfg)
        assert report["status"] == "PASS"
        assert static_report["status"] == "PASS"
        assert validate_teacher_routing_config(cfg) == "none"
        assert report["variant"] == variant
        assert report["trainable_params"] == EXPECTED_PARAMS[variant]
        assert all(
            item["matches"]
            for item in report["protected_protocol_fields"].values()
        )
        assert report["contract"]["decoder_input"] == "DINO_feature_only"
        assert not report["contract"]["gbsp_residual_decoder_input"]


def test_lcic_d_full_dataset_specific_threshold_config():
    cfg = load_config(DATASET_THRESHOLD_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert report["variant"] == "d_full"
    assert report["trainable_params"] == EXPECTED_PARAMS["d_full"]
    assert resolve_dabe_clean_dabe_v2_hard_threshold(cfg, "TR-CAMO") == 0.58
    assert resolve_dabe_clean_dabe_v2_hard_threshold(cfg, "TR-COD10K") == 0.66
    assert report["contract"]["target"] == (
        "GBSP_abs_minmax_strict_gt_CAMO_0.58_COD10K_0.66"
    )


def test_lcic_dataset_thresholds_are_applied_per_sample_in_one_batch():
    cfg = load_config(DATASET_THRESHOLD_CONFIG)
    source = torch.tensor([0.60, 0.60]).view(2, 1, 1, 1)
    clean = torch.zeros_like(source)
    thresholds = torch.tensor([0.58, 0.66])
    target = build_dabe_clean_static_target(
        cfg,
        clean,
        source,
        hard_threshold=thresholds,
    )
    assert target.flatten().tolist() == [1.0, 0.0]


def test_lcic_d_full_camo_weight_config_keeps_uniform_t058():
    cfg = load_config(DATASET_LOSS_WEIGHT_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert resolve_dabe_clean_dabe_v2_hard_threshold(cfg, "TR-CAMO") == 0.58
    assert resolve_dabe_clean_dabe_v2_hard_threshold(cfg, "TR-COD10K") == 0.58
    assert cfg.TR_CAMO_LOSS_WEIGHT == 1.5
    assert cfg.TR_COD10K_LOSS_WEIGHT == 1.0
    assert "normalized_by_batch_weight_sum" in report["contract"]["loss"]


def test_lcic_dataset_loss_weight_normalizes_per_sample_bce():
    cfg = load_config(DATASET_LOSS_WEIGHT_CONFIG)
    logits = torch.zeros(2, 1, 1, 1, requires_grad=True)
    target = torch.ones_like(logits)
    diagnostic_map = torch.ones_like(logits)
    sample_weight = build_lcic_dataset_loss_weight(
        cfg,
        ["TR-CAMO", "TR-COD10K"],
        logits,
    )
    assert sample_weight.tolist() == [1.5, 1.0]

    loss = dabe_static_bce_with_logits(
        logits,
        target,
        diagnostic_map,
        cfg,
        sample_weight=sample_weight,
    )
    loss.backward()
    gradients = logits.grad.detach().abs().flatten()
    assert torch.allclose(gradients[0] / gradients[1], torch.tensor(1.5))
    assert torch.allclose(loss.detach(), torch.tensor(0.6931472), atol=1e-6)


def test_lcic_d_full_lrstep50_is_a_uniform_t058_single_variable_config():
    cfg = load_config(LR_STEP_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert resolve_dabe_clean_dabe_v2_hard_threshold(cfg, "TR-CAMO") == 0.58
    assert resolve_dabe_clean_dabe_v2_hard_threshold(cfg, "TR-COD10K") == 0.58
    assert not bool(getattr(cfg, "LCIC_DATASET_LOSS_WEIGHT_VARIANT", False))
    assert get_step_lr_params(cfg) == (50, 0.95)
    assert report["contract"]["loss"] == "unchanged_single_mean_BCEWithLogits"
    assert "step50" in report["contract"]["scheduler"]


def test_lcic_d_full_lrstep50_builds_the_requested_scheduler():
    cfg = load_config(LR_STEP_CONFIG)
    model = torch.nn.Linear(2, 1)
    _, scheduler = build_optimizer_scheduler(cfg, model)
    assert scheduler.step_size == 50
    assert scheduler.gamma == 0.95


def test_lcic_d_full_dice005_is_a_uniform_t058_single_variable_config():
    cfg = load_config(SOFT_DICE_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert resolve_dabe_clean_dabe_v2_hard_threshold(cfg, "TR-CAMO") == 0.58
    assert resolve_dabe_clean_dabe_v2_hard_threshold(cfg, "TR-COD10K") == 0.58
    assert not bool(getattr(cfg, "LCIC_DATASET_LOSS_WEIGHT_VARIANT", False))
    assert not bool(getattr(cfg, "LCIC_LR_STEP_VARIANT", False))
    assert cfg.LCIC_SOFT_DICE_WEIGHT == 0.05
    assert report["contract"]["loss"] == (
        "mean_BCEWithLogits_plus_0.05_soft_Dice"
    )
    assert "step25" in report["contract"]["scheduler"]


def test_lcic_pure_student_reset_is_permanently_disabled():
    cfg = load_config(CONFIGS["d_full"])
    report = validate_gbsp_lcic_config(cfg)
    assert report["status"] == "PASS"
    assert cfg.PURE_STUDENT_RESET_PERMANENTLY_DISABLED
    assert cfg.FINETUNE_RESET_EPOCH == 0
    assert not cfg.FINETUNE_RESET_REBUILD_OPTIMIZER
    assert not cfg.FINETUNE_RESET_REBUILD_SCHEDULER
    assert not cfg.FINETUNE_RESET_GLOBAL_STEP
    assert not cfg.FINETUNE_RESET_FORCE_LR_FLOOR
    assert not cfg.FINETUNE_RESET_TEACHER
    assert not cfg.LR_FLOOR_APPLY_AFTER_FINETUNE_RESET
    assert not is_finetune_reset_enabled(cfg)

    cfg.FINETUNE_RESET_EPOCH = 20
    assert not is_finetune_reset_enabled(cfg)
    try:
        validate_gbsp_lcic_config(cfg)
    except RuntimeError as error:
        assert "FINETUNE_RESET_EPOCH" in str(error)
    else:
        raise AssertionError("LCIC pure-Student reset was re-enabled")


def test_r32_bw1p30_sigmaf085_uniform_training_config_contract():
    cfg = load_config(R32_BW1P30_SIGMAF085_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert report["variant"] == "d_full"
    assert report["trainable_params"] == EXPECTED_PARAMS["d_full"]
    assert report["contract"]["target"] == (
        "GBSP_R32_BW1_P30_SIGMAF085_abs_minmax_strict_gt_uniform_0.50"
    )
    assert cfg.EXP_NAME == (
        "30-gbsp-r32-lcic-d-full-dice005-t050-bw1-p30-sf085"
    )
    assert cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET == {
        "TR-CAMO": 0.50,
        "TR-COD10K": 0.50,
    }
    assert cfg.DABE_CLEAN_GBSP_VERSION == (
        "gbsp_r32_pathcand_bw1_p30_sigmaf085_v1"
    )
    assert cfg.DABE_CLEAN_GBSP_GRAPH_CANDIDATE_DIRECT
    assert cfg.GBSP_CANDIDATE_BORDER_WIDTH == 1
    assert cfg.GBSP_CANDIDATE_TOP_PERCENT == 30.0
    assert cfg.GBSP_GRAPH_SIGMA_F == 0.085


def test_lcic_seed2027_structural_only_epoch3_fork_contract():
    cfg = load_config(R32_SEED2027_STRUCTONLY_E3_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert report["active_trainable_params"] == 386
    assert cfg.STOP_AFTER_EPOCH == 4

    model = build_seg_head(384, cfg)
    freeze_report = apply_lcic_structural_only_freeze(cfg, model)
    assert freeze_report["frozen_params"] == 385
    assert freeze_report["active_params"] == 386
    assert {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    } == {"alpha", "beta", "innovation_head.weight"}

    resume_report = validate_lcic_structural_only_resume_checkpoint(
        cfg,
        {
            "epoch": 3,
            "config": {
                "EXP_NAME": (
                    "26-gbsp-r32-lcic-d-full-dice005-uniform050-seed2027"
                )
            },
        },
    )
    assert resume_report["saved_epoch"] == 3
    assert resume_report["stop_after_epoch"] == 4


def test_lcic_inference_scales_modify_only_loaded_branch_gates():
    cfg = load_config(CONFIGS["d_full"])
    model = build_seg_head(384, cfg)
    model.alpha.data.fill_(0.2)
    model.beta.data.fill_(0.3)
    anchor_weight = model.anchor.weight.detach().clone()
    innovation_weight = model.innovation_head.weight.detach().clone()
    report = apply_lcic_inference_scales(
        cfg, model, consensus_scale=3.0, innovation_scale=1.0
    )
    assert torch.allclose(model.alpha, torch.tensor(0.6))
    assert torch.allclose(model.beta, torch.tensor(0.3))
    assert torch.equal(model.anchor.weight, anchor_weight)
    assert torch.equal(model.innovation_head.weight, innovation_weight)
    assert abs(report["original_alpha"] - 0.2) < 1e-6
    assert abs(report["effective_alpha"] - 0.6) < 1e-6


def test_lcic_seed2027_consensus_gain3_training_config():
    cfg = load_config(R32_SEED2027_CONSENSUS_GAIN3_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert cfg.LCIC_CONSENSUS_GAIN == 3.0
    assert cfg.LCIC_INNOVATION_GAIN == 1.0
    assert report["contract"]["consensus_gain"] == 3.0
    assert report["contract"]["innovation_gain"] == 1.0

    model = build_seg_head(384, cfg).eval()
    assert model.consensus_gain == 3.0
    assert model.innovation_gain == 1.0
    assert sum(parameter.numel() for parameter in model.parameters()) == 771
    feature = torch.randn(1, 384, 7, 7)
    assert torch.allclose(model(feature), model.anchor(feature), atol=1e-6)
    model.alpha.data.fill_(0.2)
    diagnostics = model.epoch_diagnostics()
    assert abs(diagnostics["effective_alpha"] - 0.6) < 1e-6


def test_lcic_seed2027_adaptive_gate_training_config():
    cfg = load_config(R32_SEED2027_ADAPTIVE_GATE_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert report["trainable_params"] == 827
    assert report["contract"]["adaptive_gate"]
    assert report["contract"]["adaptive_gate_hidden"] == 8
    assert report["contract"]["adaptive_residual_normalization"] == (
        "per_image_unit_rms_detached_scale"
    )
    assert cfg.LCIC_CONSENSUS_GAIN == 1.0
    assert cfg.LCIC_INNOVATION_GAIN == 1.0


def test_lcic_seed2027_ssboc_config_and_backward_contract():
    cfg = load_config(R32_SEED2027_SSBOC_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert validate_teacher_routing_config(cfg) == "none"
    assert cfg.GBSP_SSBOC_VARIANT
    assert cfg.LCIC_SSBOC_VARIANT
    assert cfg.HEAD_TYPE == "lcic"
    assert cfg.SEED == 2027
    assert report["trainable_params"] == EXPECTED_PARAMS["d_full"]
    assert "parameter_free" in report["contract"]["loss"]

    torch.manual_seed(2027)
    model = build_seg_head(384, cfg).train()
    feature = torch.randn(2, 384, 7, 7)
    score = torch.rand(2, 1, 7, 7)
    target = (score > 0.50).float()
    logits = model(feature)
    loss, stats = gbsp_ssboc_static_loss(
        logits,
        target,
        score,
        torch.ones_like(target),
        cfg,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert stats["valid_ratio"] == 1.0
    assert model.anchor.weight.grad is not None
    assert model.innovation_head.weight.grad is not None


def test_dwlite16_seed2027_ssboc_is_feature_space_decoder():
    cfg = load_config(R32_DWLITE16_SEED2027_SSBOC_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert validate_teacher_routing_config(cfg) == "none"
    assert cfg.HEAD_TYPE == "lcic_dwlite"
    assert cfg.LCIC_VARIANT == "e_dwlite"
    assert not cfg.LCIC_SSBOC_VARIANT
    assert cfg.GBSP_SSBOC_VARIANT
    assert not cfg.LCIC_USE_CONSENSUS
    assert not cfg.LCIC_USE_INNOVATION
    assert report["contract"]["decoder_operation_space"] == "feature_space_only"
    assert report["contract"]["intermediate_logit_refinement"] is False

    torch.manual_seed(2027)
    model = build_seg_head(384, cfg).train()
    assert isinstance(model, DWLiteHead)
    assert sum(parameter.numel() for parameter in model.parameters()) == 6337
    assert not hasattr(model, "anchor")
    assert apply_lcic_inference_scales(cfg, model) is None
    try:
        apply_lcic_inference_scales(
            cfg,
            model,
            consensus_scale=3.0,
            innovation_scale=1.0,
        )
    except RuntimeError as error:
        assert "feature-space decoders" in str(error)
    else:
        raise AssertionError("DW-Lite accepted an LCIC logit-branch scale")
    feature = torch.randn(2, 384, 7, 7)
    score = torch.rand(2, 1, 7, 7)
    target = (score > 0.50).float()
    logits = model(feature)
    loss, stats = gbsp_ssboc_static_loss(
        logits,
        target,
        score,
        torch.ones_like(target),
        cfg,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert stats["valid_ratio"] == 1.0
    assert model.reduce.weight.grad is not None
    assert model.depthwise.weight.grad is not None
    assert model.out.weight.grad is not None


def test_dwlite16_ssboc_lr0003_changes_only_initial_lr_identity():
    control = load_config(R32_DWLITE16_SEED2027_SSBOC_CONFIG)
    cfg = load_config(R32_DWLITE16_SEED2027_SSBOC_LR0003_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert validate_teacher_routing_config(cfg) == "none"
    assert cfg.LR == 3e-4
    assert cfg.DINO["lr"] == 3e-4
    assert control.LR == 6e-4
    assert control.DINO["lr"] == 6e-4
    assert report["contract"]["learning_rate"] == 3e-4
    assert not cfg.GBSP_R32_DWLITE16_SSBOC_DICE005_T050_SEED2027
    assert cfg.GBSP_R32_DWLITE16_SSBOC_LR0003_DICE005_T050_SEED2027
    model = build_seg_head(384, cfg)
    assert isinstance(model, DWLiteHead)
    assert sum(parameter.numel() for parameter in model.parameters()) == 6337


def test_dwlite16_ssboc_lr1e4_floor1e5_contract():
    cfg = load_config(R32_DWLITE16_SEED2027_SSBOC_LR1E4_FLOOR1E5_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert validate_teacher_routing_config(cfg) == "none"
    assert cfg.LR == 1e-4
    assert cfg.DINO["lr"] == 1e-4
    assert cfg.LR_FLOOR == 1e-5
    assert report["contract"]["learning_rate"] == 1e-4
    assert report["contract"]["learning_rate_floor"] == 1e-5
    assert not cfg.GBSP_R32_DWLITE16_SSBOC_DICE005_T050_SEED2027
    assert cfg.GBSP_R32_DWLITE16_SSBOC_LR1E4_FLOOR1E5_DICE005_T050_SEED2027
    model = build_seg_head(384, cfg)
    assert isinstance(model, DWLiteHead)
    assert sum(parameter.numel() for parameter in model.parameters()) == 6337


def test_dwlite16_ssboc_lr1e4_floor1e5_linear45_contract():
    cfg = load_config(R32_DWLITE16_SEED2027_SSBOC_LR1E4_FLOOR1E5_LINEAR45_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert validate_teacher_routing_config(cfg) == "none"
    assert cfg.LR == 1e-4
    assert cfg.DINO["lr"] == 1e-4
    assert cfg.LR_FLOOR == 1e-5
    assert cfg.LR_POLICY == "linear_floor_two_stage"
    assert cfg.LR_LINEAR_STAGE1_EPOCHS == cfg.MAX_EPOCH == 45
    assert cfg.LR_LINEAR_STAGE2_EPOCHS == 0
    assert report["contract"]["scheduler"] == (
        "manual_linear_1e-4_to_1e-5_over_45_epochs"
    )
    assert not cfg.GBSP_R32_DWLITE16_SSBOC_DICE005_T050_SEED2027
    assert cfg.GBSP_R32_DWLITE16_SSBOC_LR1E4_FLOOR1E5_LINEAR45_DICE005_T050_SEED2027

    steps_per_epoch = 253
    first = compute_linear_floor_two_stage_lr(1, 0, steps_per_epoch, cfg)
    middle = compute_linear_floor_two_stage_lr(23, 126, steps_per_epoch, cfg)
    last = compute_linear_floor_two_stage_lr(45, 252, steps_per_epoch, cfg)
    assert first == 1e-4
    assert abs(middle - 5.5e-5) < 1e-12
    assert last == 1e-5

    model = build_seg_head(384, cfg)
    assert isinstance(model, DWLiteHead)
    assert sum(parameter.numel() for parameter in model.parameters()) == 6337


def test_conv3x3_c16_nossboc_step2floor_contract():
    cfg = load_config(R32_CONV3X3_C16_NOSSBOC_STEP2FLOOR_CONFIG)
    report = validate_gbsp_lcic_config(cfg)
    static_report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert static_report["status"] == "PASS"
    assert validate_teacher_routing_config(cfg) == "none"
    assert cfg.HEAD_TYPE == "lcic_conv3x3"
    assert cfg.LCIC_VARIANT == "f_conv3x3"
    assert cfg.LCIC_CONV3X3_CHANNELS == 16
    assert not cfg.GBSP_SSBOC_VARIANT
    assert not cfg.LCIC_SSBOC_VARIANT
    assert cfg.LCIC_SOFT_DICE_VARIANT
    assert cfg.LCIC_SOFT_DICE_WEIGHT == 0.05
    assert cfg.LR == 6e-4
    assert cfg.LR_FLOOR == 2e-5
    assert cfg.LR_POLICY == "step_then_floor"
    assert cfg.LR_STEP_THEN_FLOOR_EPOCHS == 2
    assert report["contract"]["scheduler"] == (
        "iteration_StepLR_step25_gamma0.95_first2_then_floor2e-5"
    )
    assert report["contract"]["loss"] == (
        "mean_BCEWithLogits_plus_0.05_soft_Dice"
    )

    model = build_seg_head(384, cfg)
    assert isinstance(model, Conv3x3LiteHead)
    assert model.conv3x3.groups == 1
    assert sum(parameter.numel() for parameter in model.parameters()) == 8497
    feature = torch.randn(2, 384, 11, 13)
    output = model(feature)
    assert output.shape == (2, 1, 11, 13)
    output.square().mean().backward()
    assert model.reduce.weight.grad is not None
    assert model.conv3x3.weight.grad is not None
    assert model.out.weight.grad is not None

    optimizer, _ = build_optimizer_scheduler(cfg, model)
    assert should_step_iter_scheduler(1, cfg)
    assert should_step_iter_scheduler(2, cfg)
    assert not should_step_iter_scheduler(3, cfg)
    assert apply_step_then_floor_lr(optimizer, 2, cfg) == 6e-4
    assert apply_step_then_floor_lr(optimizer, 3, cfg) == 2e-5
    assert all(group["lr"] == 2e-5 for group in optimizer.param_groups)


def test_lcic_adaptive_gate_initializes_as_linear_and_learns_gate_strengths():
    cfg = load_config(R32_SEED2027_ADAPTIVE_GATE_CONFIG)
    torch.manual_seed(2027)
    model = build_seg_head(384, cfg).train()
    assert model.adaptive_gate_enabled
    assert model.alpha is None
    assert model.beta is None
    assert sum(parameter.numel() for parameter in model.parameters()) == 827

    feature = torch.randn(3, 384, 7, 7)
    output = model(feature)
    anchor = model.anchor(feature)
    assert torch.equal(output, anchor)
    output.square().mean().backward()
    assert model.adaptive_gate[-1].weight.grad is not None
    assert model.adaptive_gate[-1].bias.grad is not None
    assert model.adaptive_gate[-1].bias.grad.abs().max().item() > 0.0

    model.reset_epoch_diagnostics()
    with torch.no_grad():
        model.adaptive_gate[-1].bias.copy_(torch.tensor([0.4, 0.1]))
    changed = model(feature)
    assert not torch.equal(changed, model.anchor(feature))
    diagnostics = model.epoch_diagnostics()
    assert diagnostics["adaptive_gate_enabled"]
    assert diagnostics["gate_samples"] == 3
    assert abs(diagnostics["consensus_gate_mean"] - 0.4) < 1e-6
    assert abs(diagnostics["innovation_gate_mean"] - 0.1) < 1e-6


def test_lcic_adaptive_gate_optimizer_has_no_gate_weight_decay():
    cfg = load_config(R32_SEED2027_ADAPTIVE_GATE_CONFIG)
    model = build_seg_head(384, cfg)
    optimizer, _ = build_optimizer_scheduler(cfg, model)
    groups = {group.get("group_name"): group for group in optimizer.param_groups}
    assert groups["decoder_base"]["weight_decay"] > 0.0
    assert groups["lcic_adaptive_gate"]["weight_decay"] == 0.0
    assert sum(
        parameter.numel()
        for parameter in groups["lcic_adaptive_gate"]["params"]
    ) == 58


def test_lcic_adaptive_gate_eval_forbids_manual_branch_scales():
    cfg = load_config(R32_SEED2027_ADAPTIVE_GATE_CONFIG)
    model = build_seg_head(384, cfg)
    assert apply_lcic_inference_scales(cfg, model) is None
    try:
        apply_lcic_inference_scales(
            cfg, model, consensus_scale=3.0, innovation_scale=1.0
        )
    except RuntimeError as error:
        assert "forbid manual inference scales" in str(error)
    else:
        raise AssertionError("Adaptive LCIC accepted a manual inference scale")


def test_lcic_soft_dice_weight_adds_exact_declared_term_and_gradient():
    cfg = load_config(SOFT_DICE_CONFIG)
    logits = torch.zeros(2, 1, 2, 2, requires_grad=True)
    target = torch.ones_like(logits)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    dice = lcic_soft_dice_loss(logits, target)
    combined = add_lcic_soft_dice_loss(bce, logits, target, cfg)
    assert torch.allclose(combined, bce + 0.05 * dice)
    combined.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all().item())


def test_innovation_readout_has_no_bias():
    model = LCICHead(384, use_consensus=False, use_innovation=True)
    assert model.innovation_head.bias is None


def test_teacher_routing_rejects_lcic_variant_head_mismatch():
    cfg = load_config(CONFIGS["d_full"])
    cfg.HEAD_TYPE = "simple"
    try:
        validate_teacher_routing_config(cfg)
    except RuntimeError as error:
        assert "decoder contract" in str(error)
    else:
        raise AssertionError("LCIC decoder contract accepted a wrong head")


def test_lcic_pure_student_logger_keeps_only_active_fields():
    kept = [
        "[Train] epoch=020/045 | loss=0.100000 | lr=0.00002000",
        (
            "[TrainArea] epoch=020 | train_pseudo_fg_area=0.095966 | "
            "student_mean_prob=0.097400 | student_fg_area@0.5=0.088547"
        ),
        (
            "[LCIC] epoch=020 | alpha=0.1 | beta=0.2 | "
            "consensus_abs=0.3 | innovation_abs=0.4"
        ),
        "[Validation] Epoch 020/045 | Dataset: TE-CAMO",
        "+-------+",
        "| adp E↑ |",
    ]
    removed = [
        "[DABE-v2 Hard StaticOnly] epoch=020",
        "[DABE-Clean] epoch=020 | target_mean=0.095966",
        "[DABE-Clean-Schedule] epoch=020",
        "[TeacherRouting] epoch=020 | mode=none",
        "[Hard-R1 PureStudent] teacher_instantiated=False",
        "[PredArea PureStudent] static_target_area_mean=0.245000",
        "finetune_reset_epoch = 20",
        "finetune_reset_teacher = False",
        "[Reset] enabled=False | epoch=disabled",
        "[FinetuneReset] epoch=020 | rebuild_optimizer=True",
        "ema_weight = 0.99",
        "PROTO_MODE = global",
    ]

    assert all(should_emit_lean_pure_student_log(line) for line in kept)
    assert not any(should_emit_lean_pure_student_log(line) for line in removed)

    class CaptureLogger:
        def __init__(self):
            self.lines = []

        def log(self, text=""):
            self.lines.append(text)

    capture = CaptureLogger()
    logger = LeanPureStudentLogger(capture)
    for line in kept + removed:
        logger.log(line)
    assert capture.lines == kept
