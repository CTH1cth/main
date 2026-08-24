import unittest

from common.dabev2hard_static_only import validate_dabev2hard_static_only_config
from common.r1hard_linear_pure_student import (
    GBSP_ABSMM_T058_CONFIG_PATH,
    GBSP_ABSMM_T058_DAGP_ONLY_CONFIG_PATH,
    is_gbsp_absmm_t058_dagp_only_pure_student_config,
)
from common.utils import load_config
from common.teacher_routing import validate_teacher_routing_config
from model import DAGPSafeHead, build_seg_head


class GBSPAbsMinMaxT058DAGPOnlyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.linear = load_config(GBSP_ABSMM_T058_CONFIG_PATH)
        cls.dagp = load_config(GBSP_ABSMM_T058_DAGP_ONLY_CONFIG_PATH)

    def test_audited_pure_student_contract(self):
        self.assertEqual(validate_teacher_routing_config(self.dagp), "none")
        report = validate_dabev2hard_static_only_config(self.dagp)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["schema"],
            "gbsp_absmm_t058_dagp_only_pure_student_config_audit_v1",
        )
        self.assertTrue(
            is_gbsp_absmm_t058_dagp_only_pure_student_config(self.dagp)
        )
        self.assertEqual(report["contract"]["student"], "dagp_safe_only")
        self.assertEqual(report["contract"]["dagp_scale_epoch1"], 1.0)

    def test_supervision_protocol_matches_linear_control(self):
        protected = (
            "DABE_CLEAN_DABE_V2_ROOT",
            "DABE_CLEAN_DABE_V2_SOURCE_KEY",
            "DABE_CLEAN_STATIC_TARGET_SOURCE",
            "R1_HARD_SOURCE_KEY",
            "R1_HARD_RESIZE_MODE",
            "R1_HARD_THRESHOLD",
            "DABE_CLEAN_DABE_V2_HARD_THRESHOLD",
            "MAX_EPOCH",
            "LR",
            "LR_FLOOR",
            "BATCH_SIZE",
            "SEED",
        )
        for name in protected:
            self.assertEqual(getattr(self.dagp, name), getattr(self.linear, name), name)

    def test_head_is_epoch1_full_dagp_without_ndr(self):
        head = build_seg_head(384, self.dagp)
        self.assertIsInstance(head, DAGPSafeHead)
        self.assertFalse(head.use_ndr_branch)
        self.assertFalse(head.use_ndr_v2)
        self.assertIsNone(head.ndr_branch)
        head.set_epoch(1)
        self.assertEqual(head._ramp_scale(), 1.0)


if __name__ == "__main__":
    unittest.main()
