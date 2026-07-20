import pytest
import torch

from common.pssf_state import PSSFHistoryBank


def _map(value, count=2, size=3):
    return torch.full((count, 1, size, size), float(value))


def _write_epoch(bank, epoch, value, consume=True):
    indices = torch.arange(bank.sample_count)
    bank.begin_epoch(epoch)
    mean, variance = bank.compute_temporal_stats(
        indices,
        epoch,
        _map(value, bank.sample_count, bank.patch_size),
        history_window=3,
    )
    bank.write(
        indices,
        epoch,
        q_prev_37=_map(0.2, bank.sample_count, bank.patch_size),
        teacher_soft_37=_map(value, bank.sample_count, bank.patch_size),
        teacher_bin_37=_map(value > 0.5, bank.sample_count, bank.patch_size),
        student_soft_37=_map(0.4, bank.sample_count, bank.patch_size),
        temporal_mean_37=mean,
        temporal_var_37=variance,
    )
    context = bank.matured_context(indices, epoch, torch.device("cpu"))
    if context is not None and consume:
        bank.mark_consumed(indices, context["source_epoch"])
    bank.end_epoch(epoch)
    return context


def test_four_slot_history_tracks_fixed_source_and_future_epochs():
    bank = PSSFHistoryBank(
        sample_count=2,
        patch_size=3,
        horizon=3,
        dtype=torch.float16,
    )
    assert bank.num_slots == 4
    assert set(bank.maps) == set(PSSFHistoryBank.MAP_NAMES)

    assert _write_epoch(bank, 1, 0.1) is None
    assert _write_epoch(bank, 2, 0.2) is None
    assert _write_epoch(bank, 3, 0.3) is None
    context = _write_epoch(bank, 4, 0.9)
    assert context["source_epoch"] == 1
    assert context["future_teacher_bin_37"].shape == (3, 2, 1, 3, 3)
    assert torch.equal(
        context["q_prev_37"],
        _map(0.2, count=2, size=3),
    )
    assert torch.equal(
        context["future_teacher_bin_37"][2],
        torch.ones(2, 1, 3, 3),
    )
    _write_epoch(bank, 5, 0.8)


def test_duplicate_sample_and_unconsumed_overwrite_are_rejected():
    bank = PSSFHistoryBank(
        sample_count=1,
        patch_size=2,
        horizon=1,
        dtype=torch.float16,
    )
    values = _map(0.2, count=1, size=2)
    bank.begin_epoch(1)
    bank.write(
        [0],
        1,
        values,
        values,
        values,
        values,
        values,
        values,
    )
    with pytest.raises(RuntimeError, match="more than once"):
        bank.write(
            [0],
            1,
            values,
            values,
            values,
            values,
            values,
            values,
        )
    bank.end_epoch(1)
    _write_epoch(bank, 2, 0.2, consume=False)
    bank.begin_epoch(3)
    with pytest.raises(RuntimeError, match="unconsumed"):
        bank.write(
            [0],
            3,
            values,
            values,
            values,
            values,
            values,
            values,
        )


def test_clear_starts_a_new_history_segment_without_cross_reset_target():
    bank = PSSFHistoryBank(
        sample_count=2,
        patch_size=3,
        horizon=3,
        dtype=torch.float16,
    )
    for epoch in range(1, 5):
        _write_epoch(bank, epoch, 0.2 + 0.1 * epoch)
    bank.clear(next_epoch=30)
    assert bank.segment_start_epoch == 30
    assert bool((bank.epoch_tag == -1).all().item())
    assert not bool(bank.target_consumed.any().item())
    assert _write_epoch(bank, 30, 0.3) is None
    assert _write_epoch(bank, 31, 0.4) is None
    assert _write_epoch(bank, 32, 0.5) is None
    context = _write_epoch(bank, 33, 0.6)
    assert context["source_epoch"] == 30


def test_history_state_round_trip_validates_shapes_and_flags():
    source = PSSFHistoryBank(2, patch_size=3, horizon=3)
    _write_epoch(source, 1, 0.4)
    state = source.state_dict()
    target = PSSFHistoryBank(2, patch_size=3, horizon=3)
    target.load_state_dict(state)
    assert torch.equal(target.epoch_tag, source.epoch_tag)
    assert torch.equal(target.maps["teacher_soft_37"], source.maps["teacher_soft_37"])

    invalid = dict(state)
    invalid["epoch_tag"] = torch.zeros(1, 1)
    with pytest.raises(RuntimeError, match="epoch tags"):
        target.load_state_dict(invalid)

