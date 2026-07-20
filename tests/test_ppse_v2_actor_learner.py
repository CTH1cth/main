import torch

from models.pssf import PredictiveSupervisionStateFilter
from train import pssf_module_state_hash, sync_ppse_v2_actor


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


def test_ppse_v2_actor_is_frozen_within_epoch_and_syncs_next_epoch():
    torch.manual_seed(3407)
    learner = _network()
    actor = _network()
    audit = sync_ppse_v2_actor(learner, actor, epoch=4)
    actor_start = pssf_module_state_hash(actor)
    assert audit["actor_hash_start"] == audit["learner_hash_start"]
    assert all(not parameter.requires_grad for parameter in actor.parameters())

    optimizer = torch.optim.AdamW(learner.parameters(), lr=1e-2)
    feature = torch.randn(2, 4, 3, 3)
    state = torch.randn(2, 9, 3, 3)
    target = torch.full((2, 1, 3, 3), 0.8)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (learner(feature, state) - target).square().mean()
        loss.backward()
        optimizer.step()

    assert pssf_module_state_hash(actor) == actor_start
    assert pssf_module_state_hash(learner) != actor_start
    assert all(parameter.grad is None for parameter in actor.parameters())

    next_audit = sync_ppse_v2_actor(learner, actor, epoch=5)
    assert next_audit["actor_hash_start"] == next_audit["learner_hash_start"]
    assert pssf_module_state_hash(actor) == pssf_module_state_hash(learner)
