import torch
from types import SimpleNamespace

from common.ecst import TemporalTeacherMemory
from common.source_arbiter import RouteTrajectoryMemory, SourceArbiter
from train import build_source_arbiter_checkpoint_extra


def test_dynamic_state_contract_round_trip():
    router = SourceArbiter()
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-4)
    route = RouteTrajectoryMemory(3, 4, 4)
    temporal = TemporalTeacherMemory(3, 4, 4)
    payload = {
        "source_arbiter": router.state_dict(),
        "source_arbiter_optimizer": optimizer.state_dict(),
        "route_trajectory_memory": route.state_dict(),
        "ecst_temporal_memory": temporal.state_dict(),
        "checkpoint_phase": "active_pre_reset",
        "global_step": 12,
    }
    restored_router = SourceArbiter()
    restored_router.load_state_dict(payload["source_arbiter"], strict=True)
    restored_route = RouteTrajectoryMemory(3, 4, 4)
    restored_route.load_state_dict(payload["route_trajectory_memory"])
    restored_temporal = TemporalTeacherMemory(3, 4, 4)
    restored_temporal.load_state_dict(payload["ecst_temporal_memory"])
    assert list(restored_router.state_dict()) == list(router.state_dict())
    assert payload["checkpoint_phase"] == "active_pre_reset"
    assert payload["global_step"] == 12


def test_training_checkpoint_extra_contains_full_active_contract():
    cfg = SimpleNamespace(
        USE_SOURCE_ARBITER=True,
        FINETUNE_RESET_EPOCH=20,
        FINETUNE_RESET_TIMING="after_epoch",
        SOURCE_ARBITER_MEMORY_UPDATE_END_EPOCH=20,
    )
    router = SourceArbiter()
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-4)
    evaluator = torch.nn.Conv2d(1, 1, 1)
    route = RouteTrajectoryMemory(3, 4, 4)
    temporal = TemporalTeacherMemory(3, 4, 4)
    generator = torch.Generator().manual_seed(3407)
    extra = build_source_arbiter_checkpoint_extra(
        cfg,
        epoch=20,
        global_step=99,
        source_arbiter=router,
        arbiter_optimizer=optimizer,
        utility_evaluator=evaluator,
        route_memory=route,
        ecst_memory=temporal,
        train_loader_generator=generator,
    )
    required = {
        "source_arbiter",
        "source_arbiter_optimizer",
        "utility_evaluator",
        "route_trajectory_memory",
        "ecst_temporal_memory",
        "global_step",
        "rng_state",
        "train_loader_generator_state",
        "checkpoint_phase",
        "source_arbiter_lifecycle",
    }
    assert required.issubset(extra)
    assert extra["checkpoint_phase"] == "pending_after_epoch_reset"
    assert extra["route_trajectory_memory"] is not None
    assert extra["ecst_temporal_memory"] is not None
