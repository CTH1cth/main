import torch

from models.gbsp_calibration import exact_area_transfer_mask


def test_area_transfer_preserves_the_exact_foreground_count_with_ties():
    score = torch.tensor([[[0.9, 0.8, 0.8], [0.8, 0.2, 0.1]]])
    source = torch.tensor([[[1, 0, 1], [0, 0, 0]]], dtype=torch.bool)

    result = exact_area_transfer_mask(score, source)

    assert result.source_foreground_count == 2
    assert result.selected_foreground_count == 2
    assert int(result.binary_mask.sum()) == 2
    assert result.boundary_tie_count == 3


def test_area_transfer_is_deterministic_and_gt_free():
    score = torch.tensor([[[0.4, 0.4], [0.4, 0.1]]])
    source = torch.tensor([[[1, 0], [0, 0]]], dtype=torch.bool)

    first = exact_area_transfer_mask(score, source)
    second = exact_area_transfer_mask(score, source)

    assert torch.equal(first.binary_mask, second.binary_mask)
    assert torch.equal(first.binary_mask, torch.tensor([[[True, False], [False, False]]]))


def test_area_transfer_supports_empty_and_full_source_masks():
    score = torch.tensor([[[0.3, 0.2], [0.1, 0.0]]])

    empty = exact_area_transfer_mask(score, torch.zeros_like(score, dtype=torch.bool))
    full = exact_area_transfer_mask(score, torch.ones_like(score, dtype=torch.bool))

    assert int(empty.binary_mask.sum()) == 0
    assert int(full.binary_mask.sum()) == score.numel()
    assert torch.isfinite(torch.tensor(empty.cutoff_score))
