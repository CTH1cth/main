import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
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
    pssf_checkpoint_phase,
)


ROOT = Path(__file__).resolve().parents[2]


def _cfg():
    return SimpleNamespace(
        EXP_NAME="pssf_runtime_test",
        FINETUNE_RESET_EPOCH=4,
        FINETUNE_RESET_TIMING="after_epoch",
        USE_PSSF=True,
        SUPERVISION_MODE="pssf_state",
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


def test_pending_reset_runtime_round_trip_and_atomic_pairing():
    output_dir = (
        ROOT
        / "workdir"
        / "_pssf_test_runtime"
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
        _write_history_epoch(history_bank, 1)
        _, _, split = build_pssf_split(
            keys,
            audit_val_ratio=0.5,
            seed=2027,
        )
        pssf = PredictiveSupervisionStateFilter(
            feature_channels=4,
            feature_proj_dim=4,
            state_channels=9,
            hidden_dim=8,
            gn_groups=4,
        )
        optimizer = torch.optim.AdamW(pssf.parameters(), lr=1e-3)
        loss = pssf(
            torch.randn(1, 4, 3, 3),
            torch.rand(1, 9, 3, 3),
        ).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(2027)

        runtime_path = output_dir / "pssf_runtime_latest.pt"
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
        )
        runtime_sha = save_runtime_payload_atomic(runtime_path, payload)
        checkpoint = {
            "epoch": 4,
            "config": {"EXP_NAME": cfg.EXP_NAME},
            **build_pssf_checkpoint_extra(
                cfg=cfg,
                epoch=4,
                global_step=17,
                pssf=pssf,
                protocol_fingerprint="protocol",
                runtime_path=runtime_path,
                runtime_sha256=runtime_sha,
            ),
        }
        checkpoint["pssf"] = {
            name: value.detach().clone()
            for name, value in checkpoint["pssf"].items()
        }
        expected_q = state_bank.q_state.clone()
        expected_parameters = {
            name: value.detach().clone()
            for name, value in pssf.state_dict().items()
        }

        state_bank.q_state.zero_()
        for parameter in pssf.parameters():
            parameter.data.zero_()
        loaded = load_pssf_resume_state(
            cfg=cfg,
            checkpoint=checkpoint,
            pssf=pssf,
            pssf_optimizer=optimizer,
            protocol_fingerprint="protocol",
            state_bank=state_bank,
            history_bank=history_bank,
            split_manifest=split,
            train_loader_generator=generator,
        )
        assert loaded["phase"] == "pending_after_epoch_reset"
        assert loaded["global_step"] == 17
        assert torch.equal(state_bank.q_state, expected_q)
        for name, value in pssf.state_dict().items():
            assert torch.equal(value, expected_parameters[name])
        assert not runtime_path.with_suffix(".pt.tmp").exists()

        q_before_reset = state_bank.q_state.clone()
        history_bank.clear(next_epoch=5)
        assert torch.equal(state_bank.q_state, q_before_reset)
        assert history_bank.segment_start_epoch == 5
        assert bool((history_bank.epoch_tag == -1).all().item())
        assert pssf_checkpoint_phase(cfg, 4) == "pending_after_epoch_reset"
        assert pssf_checkpoint_phase(cfg, 5) == "post_reset_active"

        bad = dict(checkpoint)
        bad["pssf_runtime_sha256"] = "0" * 64
        with pytest.raises(RuntimeError, match="hash mismatch"):
            load_pssf_resume_state(
                cfg=cfg,
                checkpoint=bad,
                pssf=pssf,
                pssf_optimizer=optimizer,
                protocol_fingerprint="protocol",
                state_bank=state_bank,
                history_bank=history_bank,
                split_manifest=split,
                train_loader_generator=generator,
            )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
