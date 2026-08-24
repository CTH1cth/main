from pathlib import Path
import sys

import torch


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from common.dabe_pseudo import DABE_V2_DEFAULT_PARAMS  # noqa: E402
from models.gbsp_resolution import (  # noqa: E402
    fit_coarse_background_fine_query,
    fit_gbsp_from_prepared,
    prepare_resolution_graph,
)


def _params() -> dict:
    value = dict(DABE_V2_DEFAULT_PARAMS)
    value.update({"BORDER_WIDTH": 2, "SIGMA_F": .1, "SIGMA_C": .05, "SIGMA_E": .3,
                  "TAU_BC": .3, "BG_ANCHOR_TOP_PERCENT": 30.,
                  "BG_ANCHOR_MIN_RATIO": .05, "BG_ANCHOR_FALLBACK_TOP_PERCENT": 40.})
    return value


def test_fine_and_coarse_return_finite_native_query(monkeypatch) -> None:
    import models.gbsp_resolution as module

    torch.manual_seed(5)
    feature = torch.randn(24, 8, 8)
    monkeypatch.setattr(module, "_load_rgb_grid", lambda _path, grid: torch.rand(3, grid, grid))
    prepared = prepare_resolution_graph(feature, "unused.png", _params())
    fine = fit_gbsp_from_prepared(prepared, _params(), pca_energy=.9, pca_min_rank=1, pca_max_rank=4)
    coarse = fit_coarse_background_fine_query(feature, "unused.png", _params(),
        pca_energy=.9, pca_min_rank=1, pca_max_rank=4, pooling=2)
    assert fine.minmax_residual.shape == coarse.minmax_residual.shape == (1, 8, 8)
    assert fine.bc.shape == (1, 8, 8)
    assert coarse.bc.shape == (1, 4, 4)
    assert torch.isfinite(fine.minmax_residual).all()
    assert torch.isfinite(coarse.minmax_residual).all()
    assert 0 <= fine.energy_at_rank_8 <= 1
    assert 0 <= coarse.energy_at_rank_8 <= 1


def test_candidate_ratio_is_the_only_r20_r10_config_change() -> None:
    from common.utils import load_config

    base = load_config(MAIN / "configs/dinov1_s8_gbsp_resolution_512_native64_bw2.py")
    r20 = load_config(MAIN / "configs/dinov1_s8_gbsp_resolution_512_native64_bw2_r20.py")
    r10 = load_config(MAIN / "configs/dinov1_s8_gbsp_resolution_512_native64_bw2_r10.py")
    for cfg, ratio in ((r20, 20.), (r10, 10.)):
        assert cfg.DABE_BG_ANCHOR_TOP_PERCENT == ratio
        assert cfg.GBSP_BG_RATIO == ratio / 100.
        assert cfg.DABE_SIGMA_F == base.DABE_SIGMA_F
        assert cfg.DABE_SIGMA_C == base.DABE_SIGMA_C
        assert cfg.DABE_SIGMA_E == base.DABE_SIGMA_E
        assert cfg.GBSP_PCA_MAX_RANK == base.GBSP_PCA_MAX_RANK == 8
        assert cfg.GBSP_THRESHOLD == base.GBSP_THRESHOLD == .58
