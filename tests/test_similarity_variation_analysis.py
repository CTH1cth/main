import numpy as np
import torch

from models.gbsp_similarity_baselines import compute_similarity_baselines
from tools.similarity_variation_common import equal_frequency_bins, labels_from_area, quantile_mask, similarity_matched_pairs

def features(n=1369,c=16):
 g=torch.Generator().manual_seed(7); return torch.randn(n,c,generator=g)

def test_self_match_is_excluded_for_nn_and_knn():
 bg=torch.arange(20); r=compute_similarity_baselines(features(),bg,k=8)
 assert r.self_match_violation_count==0
 assert torch.all(r.nn_background_index[bg]!=bg)

def test_knn_requires_nine_candidates_under_leave_one_out():
 try: compute_similarity_baselines(features(),torch.arange(8),k=8)
 except ValueError as e: assert 'at least 9' in str(e)
 else: raise AssertionError('expected ValueError')

def test_shapes_finiteness_and_foreground_direction():
 x=features(); bg=torch.arange(20); r=compute_similarity_baselines(x,bg,k=8)
 assert all(v.shape==(1369,) and torch.isfinite(v).all() for v in r.scores.values())
 # A query identical to a non-self background has zero NN anomaly.
 x[100]=x[3]; r=compute_similarity_baselines(x,bg,k=8)
 assert r.scores['nn_cos'][100] < 1e-5

def test_patch_area_label_rules():
 a=np.array([0,.2,.21,.49,.5,.79,.8,1.])
 y,v=labels_from_area(a,False); assert y.tolist()==[0,0,0,0,1,1,1,1] and v.all()
 y,v=labels_from_area(a,True); assert y[v].tolist()==[0,0,1,1] and v.tolist()==[1,1,0,0,0,0,1,1]

def test_high_similarity_subset_is_gt_independent_and_has_expected_size():
 s=np.arange(100,dtype=float); m1=quantile_mask(s,.8); m2=quantile_mask(s,.8)
 assert np.array_equal(m1,m2) and m1.sum()==20
 assert all(np.bincount(equal_frequency_bins(s,5))[1:]==20)

def test_similarity_matching_respects_tolerance_and_allows_bg_reuse():
 s=np.array([.50,.505,.20,.9]); y=np.array([1,0,1,0],np.uint8); v=np.ones(4,bool)
 fg,bg=similarity_matched_pairs(s,y,v,.01); assert fg.tolist()==[0] and bg.tolist()==[1]
 assert np.max(np.abs(s[fg]-s[bg]))<=.01
