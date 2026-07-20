from types import SimpleNamespace

import pytest
import torch

from common.pssf_retention import (
    bilinear_resize_gain,
    teacher_binary_observation,
    update_supervision_state,
)
from common.pssf_state import PSSFStateBank, build_pssf_split
from common.teacher_routing import validate_teacher_routing_config
from models.pssf import PredictiveSupervisionStateFilter
import train as train_module
from train import (
    build_pssf_segmentation_group,
    resolve_epoch_supervision_for_training,
    use_pssf,
    validate_pssf_config,
)


def test_strict_teacher_observation_and_state_update():
    teacher_prob = torch.tensor([0.49, 0.50, 0.51]).view(1, 1, 1, 3)
    teacher_binary = teacher_binary_observation(teacher_prob)
    assert torch.equal(
        teacher_binary,
        torch.tensor([0.0, 0.0, 1.0]).view_as(teacher_prob),
    )

    q_prev = torch.tensor([0.2, 0.4, 0.8]).view_as(teacher_prob)
    gain = torch.tensor([0.0, 0.5, 1.0]).view_as(teacher_prob)
    q_current = update_supervision_state(q_prev, teacher_binary, gain)
    expected = q_prev + gain * (teacher_binary - q_prev)
    assert torch.equal(q_current, expected)
    assert not q_current.requires_grad


def test_gain_resize_and_zero_initialized_network():
    model = PredictiveSupervisionStateFilter(
        feature_channels=384,
        feature_proj_dim=32,
        state_channels=9,
        hidden_dim=32,
        gn_groups=4,
        init_gain=0.01,
    )
    feature = torch.randn(2, 384, 37, 37)
    state = torch.rand(2, 9, 37, 37)
    gain_37 = model(feature, state)
    gain_68 = bilinear_resize_gain(gain_37, 68)
    assert gain_37.shape == (2, 1, 37, 37)
    assert gain_68.shape == (2, 1, 68, 68)
    assert torch.allclose(
        gain_37,
        torch.full_like(gain_37, 0.01),
        atol=1e-7,
        rtol=0.0,
    )
    assert model.input_channels == 41


def test_state_bank_round_trip_and_range_guard():
    keys = [("TR-CAMO", "a"), ("TR-COD10K", "b")]
    targets = [
        torch.full((1, 4, 4), 0.2),
        torch.full((1, 4, 4), 0.8),
    ]
    bank = PSSFStateBank(keys, targets, loss_size=4, dtype=torch.float16)
    saved = bank.state_dict()
    saved["q_state"] = saved["q_state"].clone()
    bank.update([0], torch.full((1, 1, 4, 4), 0.6))
    bank.load_state_dict(saved)
    assert torch.allclose(
        bank.fetch([0, 1], "cpu"),
        torch.stack(targets),
        atol=3e-4,
        rtol=0.0,
    )

    invalid = dict(saved)
    invalid["q_state"] = saved["q_state"].clone()
    invalid["q_state"][0, 0, 0, 0] = 1.5
    with pytest.raises(RuntimeError, match="outside"):
        bank.load_state_dict(invalid)


def test_source_stratified_split_is_complete_and_disjoint():
    keys = [
        ("TR-CAMO", f"camo_{index:03d}") for index in range(10)
    ] + [
        ("TR-COD10K", f"cod_{index:03d}") for index in range(30)
    ]
    train_mask, audit_mask, manifest = build_pssf_split(
        keys,
        audit_val_ratio=0.10,
        seed=2027,
    )
    assert int(train_mask.sum()) == 36
    assert int(audit_mask.sum()) == 4
    assert torch.equal(train_mask ^ audit_mask, torch.ones_like(train_mask))
    assert not bool((train_mask & audit_mask).any().item())
    assert manifest["counts"]["TR-CAMO"] == {
        "total": 10,
        "train": 9,
        "audit_val": 1,
    }
    assert manifest["counts"]["TR-COD10K"] == {
        "total": 30,
        "train": 27,
        "audit_val": 3,
    }


def test_flag_off_is_a_noop_and_pssf_config_guard_is_strict():
    assert not use_pssf(SimpleNamespace(USE_PSSF=False, SUPERVISION_MODE=""))
    assert validate_pssf_config(
        SimpleNamespace(USE_PSSF=False, SUPERVISION_MODE="")
    ) is False


def test_pssf_none_routing_bypasses_only_noecst_protocol_fields():
    cfg = SimpleNamespace(
        USE_PSSF=True,
        SUPERVISION_MODE="pssf_state",
        TEACHER_FUSION_MODE="pssf_state",
        TEACHER_ROUTING_MODE="none",
        USE_ECST=False,
        GKD_MODE="off",
    )
    assert validate_teacher_routing_config(cfg) == "none"

    cfg.TEACHER_FUSION_MODE = "dabe_pu_despl_sched"
    with pytest.raises(RuntimeError, match="requires matching"):
        validate_teacher_routing_config(cfg)


def test_pssf_path_cannot_call_legacy_schedule_or_weight_routing(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy supervision helper was called")

    monkeypatch.setattr(
        train_module,
        "get_fixed_teacher_weights",
        forbidden,
    )
    monkeypatch.setattr(
        train_module,
        "build_effective_static_weight",
        forbidden,
    )
    monkeypatch.setattr(
        train_module,
        "teacher_route_bce_with_logits",
        forbidden,
    )
    cfg = SimpleNamespace(
        USE_PSSF=True,
        SUPERVISION_MODE="pssf_state",
        LOSS_SIZE=4,
        FINETUNE_RESET_EPOCH=29,
        FINETUNE_RESET_TIMING="after_epoch",
    )
    state = resolve_epoch_supervision_for_training(cfg, epoch=7)
    assert state == {
        "fixed_weight": 0.0,
        "teacher_weight": 0.0,
        "fusion_mode": "pssf_state",
        "effective_despl_weight": 0.0,
        "effective_teacher_weight": 0.0,
        "target_mode": "pssf_state",
        "teacher_binary_used": True,
    }

    final_logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    output = {
        "coarse_logits_68": torch.randn(
            2, 1, 4, 4, requires_grad=True
        ),
        "base_logits": torch.randn(2, 1, 4, 4, requires_grad=True),
    }
    result = build_pssf_segmentation_group(
        cfg,
        epoch=7,
        student_out=output,
        student_logits=final_logits,
        q_target=torch.rand(2, 1, 4, 4),
    )
    result["loss"].backward()
    assert final_logits.grad is not None
