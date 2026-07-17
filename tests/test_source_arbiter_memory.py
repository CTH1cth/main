import pytest
import torch

from common.source_arbiter import RouteTrajectoryMemory


def _update(memory, indices, epoch):
    batch = len(indices)
    shape = (batch, 1, memory.height, memory.width)
    probability = torch.full(shape, 0.6)
    memory.update(
        indices=indices,
        teacher_prob=probability,
        student_prob=probability - 0.1,
        temporal_mean=probability - 0.2,
        temporal_variance=torch.full(shape, 0.02),
        history_count=torch.full((batch,), epoch - 1),
        teacher_prior=torch.full((batch,), 0.4),
        epoch=epoch,
    )


def test_memory_round_trip_and_epoch_guard():
    memory = RouteTrajectoryMemory(4, 3, 5, dtype="float16")
    _update(memory, [0, 2], epoch=1)
    fetched = memory.fetch([0, 2], "cpu")
    assert fetched["valid"].all()
    restored = RouteTrajectoryMemory(4, 3, 5, dtype="float16")
    restored.load_state_dict(memory.state_dict())
    for key, value in fetched.items():
        assert torch.equal(value, restored.fetch([0, 2], "cpu")[key])
    with pytest.raises(RuntimeError):
        _update(memory, [0, 2], epoch=1)


def test_memory_rejects_bad_indices_and_shape():
    memory = RouteTrajectoryMemory(2, 3, 3)
    with pytest.raises(RuntimeError):
        memory.fetch([0, 0], "cpu")
    with pytest.raises(RuntimeError):
        memory.fetch([2], "cpu")
    with pytest.raises(RuntimeError):
        memory.update(
            [0],
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 3, 3),
            torch.zeros(1, 1, 3, 3),
            torch.zeros(1, 1, 3, 3),
            torch.zeros(1),
            torch.zeros(1),
            1,
        )


def test_memory_two_epoch_delayed_state():
    memory = RouteTrajectoryMemory(2, 3, 3)
    _update(memory, [0, 1], epoch=1)
    old = memory.fetch([1, 0], "cpu")
    assert old["epoch"].tolist() == [1, 1]
    _update(memory, [1, 0], epoch=2)
    current = memory.fetch([0, 1], "cpu")
    assert current["epoch"].tolist() == [2, 2]
    assert current["valid"].all()


def test_memory_rejects_dtype_mismatch_on_resume():
    memory = RouteTrajectoryMemory(2, 3, 3, dtype="float16")
    state = memory.state_dict()
    restored = RouteTrajectoryMemory(2, 3, 3, dtype="float32")
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        restored.load_state_dict(state)
