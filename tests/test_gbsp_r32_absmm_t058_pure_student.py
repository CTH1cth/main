import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from common.dataset import _load_dabe_clean_dabe_v2_static
from common.dabev2hard_static_only import validate_dabev2hard_static_only_config
from common.r1hard_linear_pure_student import (
    GBSP_R32_ABSMM_T058_CONFIG_PATH,
    LCIC_R32_D_FULL_SOFT_DICE_CONFIG_PATH,
    LCIC_R32_D_FULL_SOFT_DICE_DATASET_THRESHOLD_CONFIG_PATH,
    LCIC_R32_D_FULL_SOFT_DICE_T050_CONFIG_PATH,
    LCIC_R32_D_FULL_SOFT_DICE_010_T050_CONFIG_PATH,
    LCIC_R32_D_FULL_SOFT_DICE_010_T050_SEED2027_CONFIG_PATH,
    LCIC_R32_D_FULL_SOFT_DICE_T050_LR0003_CONFIG_PATH,
    LCIC_R32_D_FULL_SOFT_DICE_T050_SEED_CONFIG_PATHS,
    LCIC_R32_D_FULL_T050_NO_DICE_SEED2027_CONFIG_PATH,
    is_gbsp_r32_absmm_t058_pure_student_config,
    validate_gbsp_lcic_config,
)
from common.utils import load_config


class GBSPR32AbsMinMaxT058PureStudentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(GBSP_R32_ABSMM_T058_CONFIG_PATH)

    def test_config_contract(self):
        report = validate_dabev2hard_static_only_config(self.cfg)
        self.assertEqual(report["status"], "PASS", report.get("errors"))
        self.assertEqual(
            report["schema"],
            "gbsp_r32_absmm_t058_linear_pure_student_config_audit_v1",
        )
        self.assertTrue(is_gbsp_r32_absmm_t058_pure_student_config(self.cfg))
        self.assertEqual(self.cfg.EXP_NAME, "25-gbsp-r32-absmm-t058-1×1")
        self.assertEqual(self.cfg.GBSP_PCA_RANK_MODE, "fixed")
        self.assertEqual(self.cfg.GBSP_FIXED_PCA_RANK, 32)
        self.assertEqual(self.cfg.DABE_CLEAN_GBSP_VERSION, "gbsp_pca_absmm_r32_v1")
        self.assertTrue(self.cfg.LEAN_PURE_STUDENT_LOGGING)
        self.assertTrue(self.cfg.PURE_STUDENT_RESET_PERMANENTLY_DISABLED)
        self.assertEqual(self.cfg.FINETUNE_RESET_EPOCH, 0)
        self.assertFalse(self.cfg.FINETUNE_RESET_REBUILD_OPTIMIZER)
        self.assertFalse(self.cfg.FINETUNE_RESET_REBUILD_SCHEDULER)
        self.assertFalse(self.cfg.FINETUNE_RESET_GLOBAL_STEP)
        self.assertEqual(report["contract"]["optimizer_reset"], "permanently_disabled")
        self.assertEqual(report["contract"]["source"], "gbsp_abs_minmax_37")
        self.assertEqual(report["contract"]["target"], "strict_greater_than_0.58")
        self.assertEqual(report["contract"]["student"], "single_1x1_conv")
        self.assertEqual(report["contract"]["pca_rank_mode"], "fixed")
        self.assertEqual(report["contract"]["fixed_pca_rank"], 32)

    def test_loader_enforces_fixed_r32_and_thresholds_after_resize(self):
        response = torch.linspace(0.0, 1.0, 37 * 37).reshape(1, 37, 37)
        payload = {
            "dataset": "TR-CAMO",
            "stem": "synthetic",
            "backbone_key": "dinov1-s8",
            "source_dabe_version": "v2",
            "gbsp_version": "gbsp_pca_absmm_r32_v1",
            "source_augs": ["identity"],
            "source_num_views": 1,
            "gt_used_for_generation": False,
            "pca_rank_mode": "fixed",
            "fixed_pca_rank": 32,
            "pca_max_rank": 32,
            "selected_ranks": torch.tensor([32]),
            "fallback_used": False,
            "gbsp_abs_minmax_37": response,
        }
        cfg = SimpleNamespace(
            BACKBONE_KEY="dinov1-s8",
            LOSS_SIZE=68,
            DABE_CLEAN_STATIC_TARGET_SOURCE="gbsp_abs_minmax_hard_68",
            DABE_CLEAN_DABE_V2_VERSION="v2",
            DABE_CLEAN_DABE_V2_SOURCE_KEY="gbsp_abs_minmax_37",
            DABE_CLEAN_GBSP_VERSION="gbsp_pca_absmm_r32_v1",
            DABE_CLEAN_GBSP_AUGS=["identity"],
            DABE_CLEAN_GBSP_PCA_RANK_MODE="fixed",
            DABE_CLEAN_GBSP_FIXED_PCA_RANK=32,
            DABE_CLEAN_DABE_V2_HARD_THRESHOLD=0.58,
            USE_EAOGP=False,
        )
        work_root = Path(__file__).resolve().parents[2] / "workdir"
        with tempfile.TemporaryDirectory(dir=work_root) as temporary:
            cache_path = Path(temporary) / "synthetic.pt"
            torch.save(payload, cache_path)
            result = _load_dabe_clean_dabe_v2_static(
                {"cache_path": str(cache_path)}, "TR-CAMO", "synthetic", cfg
            )
            payload["selected_ranks"] = torch.tensor([8])
            torch.save(payload, cache_path)
            with self.assertRaisesRegex(RuntimeError, "fixed-PCA rank contract"):
                _load_dabe_clean_dabe_v2_static(
                    {"cache_path": str(cache_path)}, "TR-CAMO", "synthetic", cfg
                )
        expected_soft = F.interpolate(
            response.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False
        ).squeeze(0)
        self.assertTrue(torch.equal(result["dabe_v2_soft_68"], expected_soft))
        self.assertTrue(
            torch.equal(result["dabe_v2_hard_68"], (expected_soft > 0.58).float())
        )

    def test_r32_lcic_d_full_soft_dice_config(self):
        cfg = load_config(LCIC_R32_D_FULL_SOFT_DICE_CONFIG_PATH)
        report = validate_gbsp_lcic_config(cfg)
        static_report = validate_dabev2hard_static_only_config(cfg)
        self.assertEqual(report["status"], "PASS", report.get("errors"))
        self.assertEqual(static_report["status"], "PASS", static_report.get("errors"))
        self.assertEqual(report["schema"], "gbsp_r32_lcic_decoder_config_audit_v1")
        self.assertEqual(cfg.EXP_NAME, "25-gbsp-r32-lcic-d-full-dice005-t058")
        self.assertEqual(cfg.HEAD_TYPE, "lcic")
        self.assertEqual(cfg.LCIC_VARIANT, "d_full")
        self.assertTrue(cfg.LCIC_USE_CONSENSUS)
        self.assertTrue(cfg.LCIC_USE_INNOVATION)
        self.assertTrue(cfg.LCIC_SOFT_DICE_VARIANT)
        self.assertEqual(cfg.LCIC_SOFT_DICE_WEIGHT, 0.05)
        self.assertEqual(cfg.DABE_CLEAN_GBSP_FIXED_PCA_RANK, 32)
        self.assertEqual(report["contract"]["pca_rank_mode"], "fixed")
        self.assertEqual(report["contract"]["fixed_pca_rank"], 32)

    def test_r32_lcic_soft_dice_dataset_threshold_config(self):
        cfg = load_config(
            LCIC_R32_D_FULL_SOFT_DICE_DATASET_THRESHOLD_CONFIG_PATH
        )
        report = validate_gbsp_lcic_config(cfg)
        static_report = validate_dabev2hard_static_only_config(cfg)
        self.assertEqual(report["status"], "PASS", report.get("errors"))
        self.assertEqual(static_report["status"], "PASS", static_report.get("errors"))
        self.assertEqual(
            cfg.EXP_NAME,
            "26-gbsp-r32-lcic-d-full-dice005-camo050-cod058",
        )
        self.assertEqual(cfg.HEAD_TYPE, "lcic")
        self.assertTrue(cfg.LCIC_R32_VARIANT)
        self.assertTrue(cfg.LCIC_SOFT_DICE_VARIANT)
        self.assertEqual(cfg.LCIC_SOFT_DICE_WEIGHT, 0.05)
        self.assertTrue(cfg.LCIC_DATASET_THRESHOLD_VARIANT)
        self.assertEqual(
            cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET,
            {"TR-CAMO": 0.50, "TR-COD10K": 0.58},
        )
        self.assertTrue(cfg.DABE_CLEAN_DABE_V2_REQUIRE_DATASET_THRESHOLD)
        self.assertEqual(cfg.DABE_CLEAN_GBSP_FIXED_PCA_RANK, 32)
        self.assertEqual(
            report["contract"]["target"],
            "GBSP_R32_abs_minmax_strict_gt_CAMO_0.50_COD10K_0.58",
        )
        self.assertEqual(
            report["contract"]["loss"],
            "mean_BCEWithLogits_plus_0.05_soft_Dice",
        )

    def test_r32_lcic_soft_dice_uniform_t050_config(self):
        cfg = load_config(LCIC_R32_D_FULL_SOFT_DICE_T050_CONFIG_PATH)
        report = validate_gbsp_lcic_config(cfg)
        static_report = validate_dabev2hard_static_only_config(cfg)
        self.assertEqual(report["status"], "PASS", report.get("errors"))
        self.assertEqual(static_report["status"], "PASS", static_report.get("errors"))
        self.assertEqual(
            cfg.EXP_NAME,
            "26-gbsp-r32-lcic-d-full-dice005-uniform050",
        )
        self.assertTrue(cfg.LCIC_UNIFORM_T050_VARIANT)
        self.assertEqual(
            cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET,
            {"TR-CAMO": 0.50, "TR-COD10K": 0.50},
        )
        self.assertEqual(
            report["contract"]["target"],
            "GBSP_R32_abs_minmax_strict_gt_uniform_0.50",
        )
        self.assertEqual(
            report["contract"]["loss"],
            "mean_BCEWithLogits_plus_0.05_soft_Dice",
        )

    def test_r32_lcic_soft_dice_uniform_t050_lr0003_config(self):
        cfg = load_config(LCIC_R32_D_FULL_SOFT_DICE_T050_LR0003_CONFIG_PATH)
        report = validate_gbsp_lcic_config(cfg)
        static_report = validate_dabev2hard_static_only_config(cfg)
        self.assertEqual(report["status"], "PASS", report.get("errors"))
        self.assertEqual(static_report["status"], "PASS", static_report.get("errors"))
        self.assertEqual(
            cfg.EXP_NAME,
            "31-gbsp-r32-lcic-d-full-dice005-t050-lr0003",
        )
        self.assertTrue(cfg.LCIC_LR_0003_VARIANT)
        self.assertEqual(cfg.LR, 3e-4)
        self.assertEqual(cfg.DINO["lr"], 3e-4)
        self.assertEqual(report["contract"]["learning_rate"], 3e-4)
        self.assertEqual(
            cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET,
            {"TR-CAMO": 0.50, "TR-COD10K": 0.50},
        )

    def test_r32_lcic_soft_dice_010_uniform_t050_config(self):
        cfg = load_config(LCIC_R32_D_FULL_SOFT_DICE_010_T050_CONFIG_PATH)
        report = validate_gbsp_lcic_config(cfg)
        static_report = validate_dabev2hard_static_only_config(cfg)
        self.assertEqual(report["status"], "PASS", report.get("errors"))
        self.assertEqual(static_report["status"], "PASS", static_report.get("errors"))
        self.assertEqual(
            cfg.EXP_NAME,
            "32-gbsp-r32-lcic-d-full-dice010-t050",
        )
        self.assertTrue(cfg.LCIC_SOFT_DICE_VARIANT)
        self.assertTrue(cfg.LCIC_SOFT_DICE_010_VARIANT)
        self.assertEqual(cfg.LCIC_SOFT_DICE_WEIGHT, 0.10)
        self.assertEqual(
            report["contract"]["loss"],
            "mean_BCEWithLogits_plus_0.10_soft_Dice",
        )
        self.assertEqual(
            cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET,
            {"TR-CAMO": 0.50, "TR-COD10K": 0.50},
        )

    def test_r32_lcic_soft_dice_010_uniform_t050_seed2027_config(self):
        cfg = load_config(
            LCIC_R32_D_FULL_SOFT_DICE_010_T050_SEED2027_CONFIG_PATH
        )
        report = validate_gbsp_lcic_config(cfg)
        static_report = validate_dabev2hard_static_only_config(cfg)
        self.assertEqual(report["status"], "PASS", report.get("errors"))
        self.assertEqual(static_report["status"], "PASS", static_report.get("errors"))
        self.assertEqual(
            cfg.EXP_NAME,
            "32-gbsp-r32-lcic-d-full-dice010-t050-seed2027",
        )
        self.assertTrue(cfg.LCIC_SOFT_DICE_010_VARIANT)
        self.assertTrue(cfg.LCIC_SEED_VARIANT)
        self.assertEqual(cfg.LCIC_SOFT_DICE_WEIGHT, 0.10)
        self.assertEqual(cfg.SEED, 2027)
        self.assertEqual(
            report["contract"]["loss"],
            "mean_BCEWithLogits_plus_0.10_soft_Dice",
        )

    def test_r32_lcic_soft_dice_uniform_t050_seed_repeats(self):
        for seed, path in LCIC_R32_D_FULL_SOFT_DICE_T050_SEED_CONFIG_PATHS.items():
            with self.subTest(seed=seed):
                cfg = load_config(path)
                report = validate_gbsp_lcic_config(cfg)
                static_report = validate_dabev2hard_static_only_config(cfg)
                self.assertEqual(report["status"], "PASS", report.get("errors"))
                self.assertEqual(
                    static_report["status"],
                    "PASS",
                    static_report.get("errors"),
                )
                self.assertEqual(cfg.SEED, seed)
                self.assertTrue(cfg.LCIC_SEED_VARIANT)
                self.assertEqual(
                    cfg.EXP_NAME,
                    "26-gbsp-r32-lcic-d-full-dice005-uniform050-"
                    f"seed{seed}",
                )
                self.assertEqual(
                    cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET,
                    {"TR-CAMO": 0.50, "TR-COD10K": 0.50},
                )
                self.assertEqual(
                    report["contract"]["target"],
                    "GBSP_R32_abs_minmax_strict_gt_uniform_0.50",
                )

    def test_r32_lcic_uniform_t050_no_dice_seed2027(self):
        cfg = load_config(LCIC_R32_D_FULL_T050_NO_DICE_SEED2027_CONFIG_PATH)
        report = validate_gbsp_lcic_config(cfg)
        static_report = validate_dabev2hard_static_only_config(cfg)
        self.assertEqual(report["status"], "PASS", report.get("errors"))
        self.assertEqual(static_report["status"], "PASS", static_report.get("errors"))
        self.assertEqual(
            cfg.EXP_NAME,
            "26-gbsp-r32-lcic-d-full-nodice-uniform050-seed2027",
        )
        self.assertEqual(cfg.SEED, 2027)
        self.assertTrue(cfg.LCIC_NO_SOFT_DICE_VARIANT)
        self.assertFalse(cfg.LCIC_SOFT_DICE_VARIANT)
        self.assertEqual(cfg.LCIC_SOFT_DICE_WEIGHT, 0.0)
        self.assertEqual(
            cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET,
            {"TR-CAMO": 0.50, "TR-COD10K": 0.50},
        )
        self.assertEqual(
            report["contract"]["loss"],
            "unchanged_single_mean_BCEWithLogits",
        )


if __name__ == "__main__":
    unittest.main()
