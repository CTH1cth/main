import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

from common.pssf_state import (
    PSSFHistoryBank,
    PSSFStateBank,
    build_pssf_split,
    save_runtime_payload_atomic,
)
from models.pssf import PredictiveSupervisionStateFilter
from train import (
    build_pssf_checkpoint_extra,
    build_pssf_runtime_payload,
    load_pssf_resume_state,
    pssf_module_state_hash,
    sync_ppse_v2_actor,
)


ROOT = Path(__file__).resolve().parents[2]


def _cfg():
    return SimpleNamespace(
        EXP_NAME="ppse_v2_runtime_test",
        FINETUNE_RESET_EPOCH=4,
        FINETUNE_RESET_TIMING="after_epoch",
        USE_PSSF=True,
        SUPERVISION_MODE="ppse_v2_state",
        PPSE_VERSION="horizon_normalized_prior_anchored_v2",
    )


def _network():
    return PredictiveSupervisionStateFilter(
        feature_channels=4,
        feature_proj_dim=4,
        state_channels=9,
        hidden_dim=8,
        gn_groups=4,
        init_gain=0.03,
        output_semantics="horizon_innovation_retention",
    )


def _write_history_epoch(bank, epoch):
    indices = torch.arange(bank.sample_count)
    value = torch.full(
        (bank.sample_count, 1, bank.patch_size, bank.patch_size),
        0.4,
    )
    bank.begin_epoch(epoch)
    bank.write(
        indices,
        epoch,
        value,
        value,
        value,
        value,
        value,
        torch.zeros_like(value),
    )
    bank.end_epoch(epoch)


def test_ppse_v2_runtime_restores_actor_learner_and_preserves_q_on_reset():
    output_dir = (
        ROOT
        / "workdir"
        / "_ppse_v2_test_runtime"
        / f"pid_{os.getpid()}"
    )
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    try:
        cfg = _cfg()
        keys = [("TR-CAMO", "a"), ("TR-COD10K", "b")]
        targets = [
            torch.full((1, 4, 4), 0.2),
            torch.full((1, 4, 4), 0.8),
        ]
        state_bank = PSSFStateBank(keys, targets, loss_size=4)
        history_bank = PSSFHistoryBank(2, patch_size=3, horizon=3)
        _write_history_epoch(history_bank, 4)
        _, _, split = build_pssf_split(
            keys,
            audit_val_ratio=0.5,
            seed=2027,
        )
        learner = _network()
        actor = _network()
        sync_ppse_v2_actor(learner, actor, epoch=4)
        optimizer = torch.optim.AdamW(learner.parameters(), lr=1e-3)

        loss = learner(
            torch.randn(1, 4, 3, 3),
            torch.rand(1, 9, 3, 3),
        ).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert pssf_module_state_hash(learner) != pssf_module_state_hash(actor)

        generator = torch.Generator().manual_seed(2027)
        runtime_path = output_dir / "ppse_v2_runtime_latest.pt"
        payload = build_pssf_runtime_payload(
            cfg=cfg,
            epoch=4,
            protocol_fingerprint="protocol",
            state_bank=state_bank,
            history_bank=history_bank,
            pssf_optimizer=optimizer,
            split_manifest=split,
            global_step=17,
            train_loader_generator=generator,
            pssf=learner,
            pssf_actor=actor,
            actor_sync_epoch=4,
        )
        runtime_sha = save_runtime_payload_atomic(runtime_path, payload)
        checkpoint = {
            "epoch": 4,
            "config": {"EXP_NAME": cfg.EXP_NAME},
            **build_pssf_checkpoint_extra(
                cfg=cfg,
                epoch=4,
                global_step=17,
                pssf=learner,
                pssf_actor=actor,
                actor_sync_epoch=4,
                protocol_fingerprint="protocol",
                runtime_path=runtime_path,
                runtime_sha256=runtime_sha,
            ),
        }
        expected_q = state_bank.q_state.clone()
        expected_learner_hash = pssf_module_state_hash(learner)
        expected_actor_hash = pssf_module_state_hash(actor)

        state_bank.q_state.zero_()
        for module in (learner, actor):
            for parameter in module.parameters():
                parameter.data.zero_()
        loaded = load_pssf_resume_state(
            cfg=cfg,
            checkpoint=checkpoint,
            pssf=learner,
            pssf_actor=actor,
            pssf_optimizer=optimizer,
            protocol_fingerprint="protocol",
            state_bank=state_bank,
            history_bank=history_bank,
            split_manifest=split,
            train_loader_generator=generator,
        )
        assert loaded["phase"] == "pending_after_epoch_reset"
        assert loaded["actor_sync_epoch"] == 4
        assert loaded["learner_restored"]
        assert loaded["actor_restored"]
        assert torch.equal(state_bank.q_state, expected_q)
        assert pssf_module_state_hash(learner) == expected_learner_hash
        assert pssf_module_state_hash(actor) == expected_actor_hash
        assert all(not parameter.requires_grad for parameter in actor.parameters())

        q_before_reset = state_bank.q_state.clone()
        learner_before_reset = pssf_module_state_hash(learner)
        actor_before_reset = pssf_module_state_hash(actor)
        history_bank.clear(next_epoch=5)
        assert torch.equal(state_bank.q_state, q_before_reset)
        assert pssf_module_state_hash(learner) == learner_before_reset
        assert pssf_module_state_hash(actor) == actor_before_reset
        assert bool((history_bank.epoch_tag == -1).all())

        next_audit = sync_ppse_v2_actor(learner, actor, epoch=5)
        assert next_audit["actor_hash_start"] == learner_before_reset
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
