from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from common.utils import load_config
from common.teacher_routing import validate_teacher_routing_config
from eval import infer_in_channels, make_model_input as make_eval_model_input
from model import DAGPSafeHead, SimpleConvSegHead, build_seg_head
from models.hsd import HSDV1Head, LAST4_KEYS, Last4LinearProbe
from train import build_hsd_r1_group_loss


ROOT = Path(__file__).resolve().parents[1]


def _features(batch=2, device="cpu"):
    return {
        key: torch.randn(batch, 384, 37, 37, device=device)
        for key in LAST4_KEYS
    }


def _shared_semantic_state(source, target):
    target_state = target.state_dict()
    shared = {
        name: value
        for name, value in source.state_dict().items()
        if name in target_state and target_state[name].shape == value.shape
    }
    target.load_state_dict(shared, strict=False)


def test_last4_linear_probe_is_exactly_one_conv():
    model = Last4LinearProbe(384)
    convs = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    assert len(convs) == 1
    assert convs[0].in_channels == 1536
    assert convs[0].out_channels == 1
    assert convs[0].kernel_size == (1, 1)
    assert len(list(model.children())) == 1
    assert model(_features(batch=1)).shape == (1, 1, 37, 37)


def test_last4_missing_layer_is_a_hard_error():
    features = _features(batch=1)
    del features["f10"]
    with pytest.raises(KeyError, match="f10"):
        Last4LinearProbe(384)(features)
    with pytest.raises(KeyError, match="f10"):
        HSDV1Head(use_detail=False)(features)


def test_hsd_semantic_outputs_and_never_reads_detail_image():
    model = HSDV1Head(use_detail=False)
    # An arbitrary non-tensor sentinel would fail immediately if the semantic
    # path tried to resize, cast, or otherwise consume the RGB argument.
    output = model(_features(batch=2), image_148=object())
    assert output["logits"].shape == (2, 1, 148, 148)
    assert output["final_logits"].shape == (2, 1, 148, 148)
    assert output["coarse_logits_37"].shape == (2, 1, 37, 37)
    assert output["base_logits_37"].shape == (2, 1, 37, 37)
    assert output["semantic_feat_37"].shape == (2, 64, 37, 37)
    assert output["semantic_feat_74"].shape == (2, 64, 74, 74)
    assert output["semantic_feat_148"].shape == (2, 64, 148, 148)
    assert "detail_feat_148" not in output
    assert "detail_gate_148" not in output


def test_hsd_full_zero_init_equals_shared_semantic_path_and_gates_are_bounded():
    torch.manual_seed(7)
    semantic = HSDV1Head(use_detail=False).eval()
    full = HSDV1Head(use_detail=True).eval()
    _shared_semantic_state(semantic, full)
    features = _features(batch=1)
    image = torch.rand(1, 3, 148, 148)
    with torch.no_grad():
        sem_output = semantic(features)
        full_output = full(features, image_148=image)
    torch.testing.assert_close(
        full_output["final_logits"], sem_output["final_logits"], atol=1e-7, rtol=0.0
    )
    assert float(full_output["detail_logits_residual_148"].abs().max()) == 0.0
    for key in ("cross_gate_11", "cross_gate_10", "cross_gate_9", "detail_gate_148"):
        gate = full_output[key]
        assert float(gate.min()) >= 0.0
        assert float(gate.max()) <= 1.0


def test_hsd_r1_loss_uses_strict_binary_37_and_nearest_148():
    cfg = SimpleNamespace(
        HSD_OUTPUT_SIZE=148,
        LAMBDA_NDR_COARSE_AUX=0.5,
        LAMBDA_BASE_AUX=0.5,
        LAMBDA_BASE_AUX_AFTER_RESET=0.3,
        FINETUNE_RESET_EPOCH=20,
    )
    residual = torch.tensor([[[[0.49, 0.50], [0.50001, 1.0]]]])
    strict = (residual > 0.5).float()
    assert strict.flatten().tolist() == [0.0, 0.0, 1.0, 1.0]

    target_37 = torch.zeros(1, 1, 37, 37)
    target_37[:, :, 0, 0] = 1.0
    model = HSDV1Head(use_detail=False)
    output = model(_features(batch=1))
    result = build_hsd_r1_group_loss(cfg, 1, output, target_37)
    assert result["target_148"].shape == (1, 1, 148, 148)
    # nearest 4x lift preserves an exact 4x4 positive block.
    assert int(result["target_148"].sum()) == 16
    expected = (
        result["loss_final"]
        + 0.5 * result["loss_coarse"]
        + 0.5 * result["loss_base"]
    ) / 2.0
    torch.testing.assert_close(result["loss"], expected)

    bad = target_37.clone()
    bad[:, :, 0, 0] = 0.5
    with pytest.raises(RuntimeError, match="strict binary"):
        build_hsd_r1_group_loss(cfg, 1, output, bad)


def test_hsd_multiscale_supervision_is_37_74_148():
    model = HSDV1Head(use_detail=False, coarse_size=74)
    output = model(_features(batch=1))
    assert output["base_logits_37"].shape == (1, 1, 37, 37)
    assert output["coarse_logits_74"].shape == (1, 1, 74, 74)
    assert output["final_logits"].shape == (1, 1, 148, 148)

    target = torch.zeros(1, 1, 37, 37)
    target[:, :, 0, 0] = 1.0
    cfg = SimpleNamespace(
        HSD_OUTPUT_SIZE=148,
        HSD_COARSE_SIZE=74,
        LAMBDA_NDR_COARSE_AUX=0.5,
        LAMBDA_BASE_AUX=0.5,
        LAMBDA_BASE_AUX_AFTER_RESET=0.5,
        FINETUNE_RESET_EPOCH=0,
    )
    result = build_hsd_r1_group_loss(cfg, 1, output, target)
    assert result["target_37"].shape == (1, 1, 37, 37)
    assert result["target_coarse"].shape == (1, 1, 74, 74)
    assert result["target_148"].shape == (1, 1, 148, 148)
    assert int(result["target_coarse"].sum()) == 4
    assert int(result["target_148"].sum()) == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hsd_full_cuda_batch_backward_has_finite_gradients():
    device = torch.device("cuda")
    model = HSDV1Head(use_detail=True).to(device).train()
    output = model(
        _features(batch=2, device=device),
        image_148=torch.rand(2, 3, 148, 148, device=device),
    )
    target = torch.randint(0, 2, (2, 1, 37, 37), device=device).float()
    cfg = SimpleNamespace(
        HSD_OUTPUT_SIZE=148,
        LAMBDA_NDR_COARSE_AUX=0.5,
        LAMBDA_BASE_AUX=0.5,
        LAMBDA_BASE_AUX_AFTER_RESET=0.3,
        FINETUNE_RESET_EPOCH=20,
    )
    loss = build_hsd_r1_group_loss(cfg, 1, output, target)["loss"]
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
    assert model.detail_to_semantic.weight.grad is not None
    assert float(model.detail_to_semantic.weight.grad.abs().sum()) > 0.0


def test_new_configs_are_pure_student_and_old_heads_still_build_unchanged():
    for name, decoder, detail in (
        ("dinov1_s8_r1hard_last4_linear.py", "last4_linear", False),
        ("dinov1_s8_r1hard_hsd_v1_sem_148.py", "hsd_v1", False),
        ("dinov1_s8_r1hard_hsd_v1_full_148.py", "hsd_v1", True),
    ):
        cfg = load_config(ROOT / "configs" / name)
        assert cfg.DECODER_TYPE == decoder
        assert cfg.USE_DETAIL is detail
        assert cfg.DINO_FEATURE_KEYS == ["f9", "f10", "f11", "f12"]
        assert cfg.R1_HARD_SOURCE_KEY == "residual_pass1_37"
        assert cfg.R1_HARD_THRESHOLD == 0.5
        assert cfg.DABEV2HARD_PURE_STUDENT is True
        assert cfg.FINETUNE_RESET_TEACHER is False
        assert cfg.USE_TEACHER_BINARY_FULL_LOSS is False
        assert cfg.USE_TEACHER_SOFT_FULL_LOSS is False
        assert cfg.USE_TEACHER_CONF_LOSS is False
        assert validate_teacher_routing_config(cfg) == "none"

    simple_cfg = SimpleNamespace(HEAD_TYPE="simple", DECODER_TYPE="")
    assert type(build_seg_head(384, simple_cfg)) is SimpleConvSegHead

    dagp_cfg = load_config(
        ROOT
        / "configs"
        / "dinov1_s8_dabe_clean_v1_dp_r1hard_dagp_uncgate_ndr_staticonly_purestudent_long45_lrfloor_2e5.py"
    )
    assert type(build_seg_head(384, dagp_cfg)) is DAGPSafeHead


@pytest.mark.parametrize(
    ("config_name", "wrong_head"),
    (
        ("dinov1_s8_r1hard_last4_linear.py", "simple"),
        ("dinov1_s8_r1hard_hsd_v1_sem_148.py", "simple"),
        ("dinov1_s8_r1hard_hsd_v1_full_148.py", "dagp_safe"),
    ),
)
def test_new_config_teacher_routing_rejects_wrong_decoder_head(
    config_name, wrong_head
):
    cfg = load_config(ROOT / "configs" / config_name)
    cfg.HEAD_TYPE = wrong_head
    with pytest.raises(RuntimeError, match="decoder contract"):
        validate_teacher_routing_config(cfg)


@pytest.mark.parametrize(
    "config_name",
    (
        "dinov1_s8_r1hard_last4_linear_online.py",
        "dinov1_s8_r1hard_hsd_v1_sem_ms148_online.py",
        "dinov1_s8_r1hard_hsd_v1_full_ms148_online.py",
    ),
)
def test_formal_online_configs_disable_reset_and_feature_cache(config_name):
    cfg = load_config(ROOT / "configs" / config_name)
    assert cfg.R1_FORMAL_ONLINE_V1 is True
    assert cfg.ONLINE_DINO_LAST4 is True
    assert cfg.DINO_FEATURE_MODE == "online_last4"
    assert cfg.FINETUNE_RESET_EPOCH == 0
    assert cfg.FINETUNE_RESET_REBUILD_OPTIMIZER is False
    assert cfg.FINETUNE_RESET_REBUILD_SCHEDULER is False
    assert cfg.FINETUNE_RESET_GLOBAL_STEP is False
    assert cfg.FINETUNE_RESET_TEACHER is False
    assert validate_teacher_routing_config(cfg) == "none"
    if cfg.R1_HSD_V1:
        assert cfg.HSD_COARSE_SIZE == 74
        assert cfg.HSD_SPATIAL_SUPERVISION_SIZES == [37, 74, 148]
        assert cfg.LAMBDA_BASE_AUX_AFTER_RESET == cfg.LAMBDA_BASE_AUX == 0.5


def test_eval_entrypoint_understands_last4_and_hsd_checkpoints():
    hsd_cfg = load_config(ROOT / "configs" / "dinov1_s8_r1hard_hsd_v1_full_148.py")
    hsd_model = build_seg_head(384, hsd_cfg)
    assert infer_in_channels(hsd_model.state_dict()) == 384

    linear_cfg = load_config(ROOT / "configs" / "dinov1_s8_r1hard_last4_linear.py")
    linear_model = build_seg_head(384, linear_cfg)
    assert infer_in_channels(linear_model.state_dict()) == 384

    batch = {
        f"feature_l{layer}": torch.randn(1, 384, 37, 37)
        for layer in (9, 10, 11, 12)
    }
    model_input = make_eval_model_input(hsd_cfg, batch, torch.device("cpu"))
    assert tuple(model_input) == LAST4_KEYS
