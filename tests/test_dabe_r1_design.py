from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_f
from PIL import Image

from common.cache_dabe_r1_design import build_r1_design_cache
from common.dabe_pseudo import (
    DABE_V2_DEFAULT_PARAMS,
    _background_anchor,
    _background_connectivity,
    _background_residual,
    _build_local_graph,
    _load_rgb_grid,
    _sobel_magnitude,
    _validate_feature,
)
from common.dabe_r1_design import (
    border_source_mask,
    build_local_graph_variant,
    build_r1_design_candidates,
    reconstruct_background,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = MAIN_ROOT.parent


@contextmanager
def cth_temporary_directory():
    root = BASELINE_ROOT / "workdir" / "pytest_r1_design_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="case-", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _synthetic_inputs(seed=11):
    generator = torch.Generator().manual_seed(seed)
    feature = torch.rand((384, 37, 37), generator=generator)
    feat_n = torch_f.normalize(feature.permute(1, 2, 0).reshape(1369, 384), dim=1)
    rgb_chw = torch.rand((3, 37, 37), generator=generator)
    rgb = rgb_chw.permute(1, 2, 0).reshape(1369, 3)
    image296 = torch.rand((3, 296, 296), generator=generator)
    return feature, feat_n, rgb_chw, rgb, image296


def test_border_width_counts_and_strict_subset():
    bw1, bw2 = border_source_mask(37, 1), border_source_mask(37, 2)
    assert int(bw1.sum()) == 144
    assert int(bw2.sum()) == 280
    assert torch.all(~bw1 | bw2)
    assert torch.any(bw2 & ~bw1)


def test_m0_graph_bc_anchor_and_residual_match_current():
    _, feat_n, rgb_chw, rgb, image296 = _synthetic_inputs()
    params = dict(DABE_V2_DEFAULT_PARAMS)
    edge = _sobel_magnitude(rgb_chw).reshape(-1)
    current_idx, current_weight = _build_local_graph(feat_n, rgb, edge, 37, params)
    graph = build_local_graph_variant(
        feat_n=feat_n, rgb_37=rgb, image_rgb_296=image296,
        grid=37, params=params, edge_mode="node_sobel",
    )
    assert torch.equal(graph.neigh_idx, current_idx)
    assert torch.equal(graph.neigh_weight, current_weight)
    bc_current, border_current = _background_connectivity(current_idx, current_weight, 37, params)
    bc, border = _background_connectivity(graph.neigh_idx, graph.neigh_weight, 37, params)
    assert torch.equal(border, border_current)
    assert torch.equal(bc, bc_current)
    anchor_current = _background_anchor(bc_current, border_current, params)
    anchor = _background_anchor(bc, border, params)
    assert torch.equal(anchor, anchor_current)
    current = _background_residual(feat_n, rgb, anchor, params)
    detailed = reconstruct_background(feat_n, rgb, anchor, params, "prototype")
    assert torch.allclose(detailed.normalized_residual, current, atol=1e-6)


def test_noedge_cue_zero_and_weight_formula():
    _, feat_n, _, rgb, image296 = _synthetic_inputs()
    params = dict(DABE_V2_DEFAULT_PARAMS)
    graph = build_local_graph_variant(
        feat_n=feat_n, rgb_37=rgb, image_rgb_296=image296,
        grid=37, params=params, edge_mode="none",
    )
    valid = graph.neigh_weight > 0
    assert torch.equal(graph.edge_cue, torch.zeros_like(graph.edge_cue))
    src = torch.arange(1369)[:, None].expand_as(graph.neigh_idx)
    dst = graph.neigh_idx
    df = 1 - (feat_n[src] * feat_n[dst]).sum(-1)
    dc = torch.square(rgb[src] - rgb[dst]).sum(-1)
    expected = torch.exp(-df / params["SIGMA_F"] - dc / params["SIGMA_C"]).clamp_min(1e-8)
    assert torch.allclose(graph.neigh_weight[valid], expected[valid], atol=1e-7)


def _slot(graph, src, dst):
    slots = torch.where((graph.neigh_idx[src] == dst) & (graph.neigh_weight[src] > 0))[0]
    assert slots.numel() == 1
    return int(slots[0])


def test_hr_interface_vertical_boundary_symmetry_range_and_indices():
    _, feat_n, _, rgb, _ = _synthetic_inputs()
    image = torch.zeros((3, 296, 296))
    image[:, :, 144:] = 1.0
    params = dict(DABE_V2_DEFAULT_PARAMS)
    graph = build_local_graph_variant(
        feat_n=feat_n, rgb_37=rgb, image_rgb_296=image,
        grid=37, params=params, edge_mode="hr_interface",
    )
    left, right = 10 * 37 + 17, 10 * 37 + 18
    high = float(graph.edge_cue[left, _slot(graph, left, right)])
    low_src, low_dst = 10 * 37 + 10, 10 * 37 + 11
    low = float(graph.edge_cue[low_src, _slot(graph, low_src, low_dst)])
    assert high > 0.5 and low < 0.05
    assert high == pytest.approx(float(graph.edge_cue[right, _slot(graph, right, left)]))
    diagonal = 10 * 37 + 10, 11 * 37 + 11
    assert float(graph.edge_cue[diagonal[0], _slot(graph, *diagonal)]) == pytest.approx(
        float(graph.edge_cue[diagonal[1], _slot(graph, diagonal[1], diagonal[0])])
    )
    assert float(graph.edge_cue.min()) >= 0 and float(graph.edge_cue.max()) <= 1
    assert torch.all(graph.edge_cue[graph.neigh_weight == 0] == 0)


def test_prototype_details_contract_weights_and_constant_minmax():
    generator = torch.Generator().manual_seed(3)
    feat = torch_f.normalize(torch.rand((12, 8), generator=generator), dim=1)
    rgb = torch.rand((12, 3), generator=generator)
    anchor = torch.zeros(12, dtype=torch.bool); anchor[:5] = True
    params = {**DABE_V2_DEFAULT_PARAMS, "K_RECON": 4}
    details = reconstruct_background(feat, rgb, anchor, params, "prototype")
    assert details.raw_residual.shape == (12,)
    assert details.topk_weight.shape == (12, 4)
    assert torch.allclose(details.topk_weight.sum(1), torch.ones(12), atol=1e-6)
    assert float(details.raw_residual.min()) >= 0
    assert 0 <= float(details.normalized_residual.min()) <= float(details.normalized_residual.max()) <= 1
    constant_feat = torch_f.normalize(torch.ones_like(feat), dim=1)
    constant_rgb = torch.zeros_like(rgb)
    constant = reconstruct_background(constant_feat, constant_rgb, anchor, params, "prototype")
    assert torch.equal(constant.normalized_residual, torch.zeros_like(constant.normalized_residual))


def test_support_consistent_detects_inconsistent_atom_composition():
    root3 = 3 ** 0.5 / 2
    feat = torch.tensor([[1.0, 0.0], [0.5, root3], [0.5, -root3]])
    rgb = torch.zeros((3, 3))
    anchor = torch.tensor([False, True, True])
    params = {**DABE_V2_DEFAULT_PARAMS, "K_RECON": 2, "LAMBDA_COLOR_RECON": 0.0}
    proto = reconstruct_background(feat, rgb, anchor, params, "prototype")
    sc = reconstruct_background(feat, rgb, anchor, params, "support_consistent")
    assert float(proto.semantic_residual[0]) < 1e-6
    assert float(sc.semantic_residual[0]) == pytest.approx(0.5, abs=1e-6)
    expected_sem = (
        sc.topk_weight[0]
        * (1 - (feat[0][None] * feat[anchor][sc.topk_anchor_local_index[0]]).sum(1))
    ).sum()
    assert float(sc.semantic_residual[0]) == pytest.approx(float(expected_sem), abs=1e-6)
    assert float(sc.raw_residual[0]) > float(proto.raw_residual[0])
    consistent = torch.tensor([[1.0, 0.0], [0.999, 0.0447], [0.999, -0.0447]])
    consistent = torch_f.normalize(consistent, dim=1)
    detail = reconstruct_background(consistent, rgb, anchor, params, "support_consistent")
    assert float(detail.semantic_residual[0]) < 0.01
    assert float(detail.support_norm[0]) > 0.99


def _prepare_cache_case(case: Path):
    generator = torch.Generator().manual_seed(31)
    image_path = case / "image.png"
    Image.fromarray((torch.rand((296, 296, 3), generator=generator) * 255).byte().numpy(), mode="RGB").save(image_path)
    feature = torch.rand((384, 37, 37), generator=generator)
    params = dict(DABE_V2_DEFAULT_PARAMS)
    validated = _validate_feature(feature, 37)
    rgb_chw = _load_rgb_grid(image_path, 37)
    feat_n = torch_f.normalize(validated.permute(1, 2, 0).reshape(1369, 384), dim=1)
    rgb = rgb_chw.permute(1, 2, 0).reshape(1369, 3)
    edge = _sobel_magnitude(rgb_chw).reshape(-1)
    idx, weight = _build_local_graph(feat_n, rgb, edge, 37, params)
    bc, border = _background_connectivity(idx, weight, 37, params)
    anchor = _background_anchor(bc, border, params)
    r1 = _background_residual(feat_n, rgb, anchor, params).reshape(1, 37, 37)
    dabe_root = case / "dabe"; dabe_path = dabe_root / "CHAMELEON" / "synthetic.pt"
    dabe_path.parent.mkdir(parents=True)
    torch.save({"dataset":"CHAMELEON","stem":"synthetic","image_path":str(image_path.resolve()),"dabe_version":"v2","augs":["identity"],"num_views":1,"residual_pass1_37":r1},dabe_path)
    dabe_manifest=dabe_root/'manifest_test.jsonl'; drow={"dataset":"CHAMELEON","stem":"synthetic","cache_path":str(dabe_path.resolve()),"image_path":str(image_path.resolve())}; _write_jsonl(dabe_manifest,[drow])
    cache_root=case/'datasets'/'cache'; froot=cache_root/'features_cache'/'dinov1-s8'; fpath=froot/'test'/'CHAMELEON'/'synthetic.pt'; fpath.parent.mkdir(parents=True)
    torch.save({"dataset":"CHAMELEON","stem":"synthetic","image_path":str(image_path.resolve()),"tensor":feature},fpath)
    fmanifest=froot/'manifest_test.jsonl'; frow={"dataset":"CHAMELEON","stem":"synthetic","cache_path":str(fpath.resolve()),"image_path":str(image_path.resolve())}; _write_jsonl(fmanifest,[frow])
    config=case/'config.py'; config.write_text(f"BACKBONE_KEY='dinov1-s8'\nCACHE_ROOT={str(cache_root)!r}\nDINO={{'feature_input_size':296}}\n",encoding='utf-8')
    return locals()


def test_build_candidates_payload_and_m0_exact():
    with cth_temporary_directory() as case:
        paths = _prepare_cache_case(case)
        result = build_r1_design_candidates(
            feature_37=paths['feature'], image_path=str(paths['image_path']),
            cached_r1_37=paths['r1'], effective_params=paths['params'], feature_input_size=296,
        )
        assert result['diagnostics']['m0_cached_r1_max_abs'] <= 1e-6
        assert result['diagnostics']['border_source_count_bw1'] == 144
        assert result['diagnostics']['border_source_count_bw2'] == 280
        for index in range(6):
            value=result[["m0_bw2_node_proto_37","m1_bw1_node_proto_37","m2_bw2_noedge_proto_37","m3_bw2_hrinterface_proto_37","m4_bw2_node_sc_37","m5_bw1_node_sc_37"][index]]
            assert value.shape==(1,37,37) and torch.isfinite(value).all()


def test_cache_io_contract_and_sources_unchanged():
    with cth_temporary_directory() as case:
        p=_prepare_cache_case(case)
        source=[p['dabe_path'],p['dabe_manifest'],p['fpath'],p['fmanifest']]
        before={x:x.read_bytes() for x in source}
        out=case/'out'; protocol=build_r1_design_cache(p['config'],p['dabe_root'],out,max_samples=1)
        assert protocol['num_samples']==1 and protocol['baseline_recompute_max_abs']<=1e-6
        for path,data in before.items(): assert path.read_bytes()==data
        row=json.loads((out/'manifest_test.jsonl').read_text().strip())
        payload=torch.load(row['cache_path'],map_location='cpu',weights_only=False)
        assert payload['design_version']=='dabe_r1_design_v1'
        assert payload['m5_bw1_node_sc_37'].shape==(1,37,37)


@pytest.mark.parametrize('mutation',['missing','duplicate','nonidentity','bad_feature','bad_r1','backbone'])
def test_cache_rejections(mutation):
    with cth_temporary_directory() as case:
        p=_prepare_cache_case(case)
        if mutation=='missing': _write_jsonl(p['fmanifest'],[])
        elif mutation=='duplicate': _write_jsonl(p['dabe_manifest'],[p['drow'],p['drow']])
        elif mutation in {'nonidentity','bad_r1'}:
            value=torch.load(p['dabe_path'],map_location='cpu',weights_only=False)
            if mutation=='nonidentity': value['augs']=['hflip']
            else: value['residual_pass1_37']=torch.zeros(1,37,37)
            torch.save(value,p['dabe_path'])
        elif mutation=='bad_feature':
            value=torch.load(p['fpath'],map_location='cpu',weights_only=False); value['tensor']=torch.zeros(384,36,37); torch.save(value,p['fpath'])
        else: p['config'].write_text("BACKBONE_KEY='dinov2-b14'\nDINO={'feature_input_size':296}\n",encoding='utf-8')
        with pytest.raises((ValueError,RuntimeError)):
            build_r1_design_cache(p['config'],p['dabe_root'],case/'out',max_samples=1)
