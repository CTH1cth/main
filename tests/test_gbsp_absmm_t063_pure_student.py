import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from common.dataset import _load_dabe_clean_dabe_v2_static
from common.dabev2hard_static_only import validate_dabev2hard_static_only_config
from common.r1hard_linear_pure_student import (
    GBSP_ABSMM_T063_CONFIG_PATH,
    is_gbsp_absmm_t063_pure_student_config,
)
from common.utils import load_config


class GBSPAbsMinMaxT063PureStudentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(GBSP_ABSMM_T063_CONFIG_PATH)

    def test_config_contract(self):
        report = validate_dabev2hard_static_only_config(self.cfg)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["schema"],
            "gbsp_absmm_t063_linear_pure_student_config_audit_v1",
        )
        self.assertTrue(is_gbsp_absmm_t063_pure_student_config(self.cfg))
        self.assertEqual(report["contract"]["source"], "gbsp_abs_minmax_37")
        self.assertEqual(report["contract"]["target"], "strict_greater_than_0.63")
        self.assertEqual(report["contract"]["student"], "single_1x1_conv")
        self.assertTrue(self.cfg.GBSP_ORACLE_THRESHOLD_VERIFICATION)

    def test_loader_resizes_continuous_response_before_strict_threshold(self):
        response = torch.linspace(0.0, 1.0, 37 * 37).reshape(1, 37, 37)
        payload = {
            "dataset": "TR-CAMO",
            "stem": "synthetic",
            "backbone_key": "dinov1-s8",
            "source_dabe_version": "v2",
            "gbsp_version": "gbsp_pca_absmm_v1",
            "source_augs": ["identity"],
            "source_num_views": 1,
            "gt_used_for_generation": False,
            "gbsp_abs_minmax_37": response,
        }
        work_root = Path(__file__).resolve().parents[2] / "workdir"
        work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work_root) as temporary:
            cache_path = Path(temporary) / "synthetic.pt"
            torch.save(payload, cache_path)
            cfg = SimpleNamespace(
                BACKBONE_KEY="dinov1-s8",
                LOSS_SIZE=68,
                DABE_CLEAN_STATIC_TARGET_SOURCE="gbsp_abs_minmax_hard_68",
                DABE_CLEAN_DABE_V2_VERSION="v2",
                DABE_CLEAN_DABE_V2_SOURCE_KEY="gbsp_abs_minmax_37",
                DABE_CLEAN_GBSP_VERSION="gbsp_pca_absmm_v1",
                DABE_CLEAN_GBSP_AUGS=["identity"],
                DABE_CLEAN_DABE_V2_HARD_THRESHOLD=0.63,
                USE_EAOGP=False,
            )
            result = _load_dabe_clean_dabe_v2_static(
                {"cache_path": str(cache_path)},
                "TR-CAMO",
                "synthetic",
                cfg,
            )
        expected_soft = F.interpolate(
            response.unsqueeze(0),
            size=(68, 68),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        self.assertTrue(torch.equal(result["dabe_v2_soft_68"], expected_soft))
        self.assertTrue(
            torch.equal(
                result["dabe_v2_hard_68"],
                (expected_soft > 0.63).float(),
            )
        )


if __name__ == "__main__":
    unittest.main()
