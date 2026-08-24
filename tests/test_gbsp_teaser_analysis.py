from pathlib import Path

import numpy as np
import torch

from tools.gbsp_teaser_analysis.common import NUM_PATCHES, labels, load_settings, probability_hist
from tools.gbsp_teaser_analysis.compute_gbsp_residual_hist import summarize_residual
from tools.gbsp_teaser_analysis.compute_pairwise_similarity_hist import pairwise_values, summarize_pairwise
from tools.gbsp_teaser_analysis.run_camo import _high_similarity_ambiguity


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/gbsp_teaser_real.yaml"


def test_frozen_protocol_settings() -> None:
    settings = load_settings(CONFIG)
    assert settings.dataset == "CAMO"
    assert settings.gbsp_rank == 8
    assert np.array_equal(settings.similarity_bins, np.linspace(-1, 1, 81))
    assert np.array_equal(settings.residual_bins, np.linspace(0, 1, 51))


def test_main_and_core_gt_grouping() -> None:
    settings = load_settings(CONFIG)
    occupancy = np.linspace(0, 1, NUM_PATCHES)
    main, main_valid = labels(occupancy, settings, "allpatch_0.5")
    core, core_valid = labels(occupancy, settings, "core_0.2_0.8")
    assert main_valid.all()
    assert np.array_equal(main, occupancy >= .5)
    assert np.array_equal(core_valid, (occupancy <= .2) | (occupancy >= .8))
    assert np.all(core[occupancy >= .8] == 1)
    assert np.all(core[occupancy <= .2] == 0)


def test_pairwise_policy_excludes_diagonal_and_symmetric_duplicates() -> None:
    feature = torch.eye(4)
    similarity = feature @ feature.T
    label = np.array([0, 0, 1, 1], dtype=np.uint8)
    valid = np.ones(4, dtype=bool)
    bb, fb = pairwise_values(similarity, label, valid)
    assert bb.shape == (1,)  # C(2,2), not four ordered pairs and no diagonal.
    assert fb.shape == (4,)  # two foreground by two background patches.
    assert np.array_equal(bb, [0.0])
    assert np.array_equal(fb, np.zeros(4))


def test_histograms_are_unit_sum_per_image() -> None:
    generator = torch.Generator().manual_seed(2026)
    feature = torch.nn.functional.normalize(torch.randn(NUM_PATCHES, 8, generator=generator), dim=1)
    similarity = feature @ feature.T
    label = np.zeros(NUM_PATCHES, dtype=np.uint8); label[-10:] = 1
    valid = np.ones(NUM_PATCHES, dtype=bool)
    pair = summarize_pairwise(similarity, label, valid, np.linspace(-1, 1, 81), (.7, .8, .9))
    residual = np.linspace(0, 1, NUM_PATCHES)
    res = summarize_residual(residual, label, valid, np.linspace(0, 1, 51))
    assert np.isclose(pair["bb_hist"].sum(), 1.0)
    assert np.isclose(pair["fb_hist"].sum(), 1.0)
    assert np.isclose(res["bg_hist"].sum(), 1.0)
    assert np.isclose(res["fg_hist"].sum(), 1.0)


def test_high_similarity_ambiguity_requires_overlap_and_high_tail() -> None:
    bb = {"P(sim>0.7)": .12, "P(sim>0.8)": .05, "P(sim>0.9)": .01}
    fb_low_tail = {"P(sim>0.7)": .003, "P(sim>0.8)": .0004, "P(sim>0.9)": .00001}
    supported, ratios = _high_similarity_ambiguity(bb, fb_low_tail, ovl=.55)
    assert not supported
    assert ratios["0.7"] < .25

    fb_comparable = {"P(sim>0.7)": .04, "P(sim>0.8)": .015, "P(sim>0.9)": .002}
    supported, _ = _high_similarity_ambiguity(bb, fb_comparable, ovl=.55)
    assert supported
