#!/usr/bin/env python3
"""Audit the DABE-v2-hard ECST/no-ECST single-variable control."""

import argparse
import json
import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dabev2hard_noecst_control import (  # noqa: E402
    BASELINE_CONFIG_PATH,
    CONTROL_CONFIG_PATH,
    build_control_audit_report,
)
from common.teacher_routing import (  # noqa: E402
    teacher_routing_uses_ecst,
    validate_teacher_routing_config,
)
from common.utils import load_config  # noqa: E402


DEFAULT_OUTPUT = (
    MAIN_ROOT / "analysis/dabev2hard_ecst_vs_noecst_config_diff.txt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare every resolved config field and verify that the only "
            "conceptual change is disabling ECST teacher routing."
        )
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=BASELINE_CONFIG_PATH,
    )
    parser.add_argument(
        "--control-config",
        type=Path,
        default=CONTROL_CONFIG_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing audit text file.",
    )
    return parser.parse_args()


def format_report(report):
    lines = [
        "DABE-v2 Hard ECST vs no-ECST configuration audit",
        f"status = {report['status']}",
        f"baseline_config = {report['baseline_config']}",
        f"control_config = {report['control_config']}",
        "compared_uppercase_and_lowercase_field_count = "
        f"{report['compared_uppercase_and_lowercase_field_count']}",
        f"allowed_differences = {report['allowed_differences']}",
        "required_effective_differences = "
        f"{report['required_effective_differences']}",
        "",
        "[Effective Differences]",
    ]
    for record in report["actual_differences"]:
        lines.extend(
            [
                f"field = {record['field']}",
                f"  baseline = {record['baseline']!r}",
                f"  control = {record['control']!r}",
            ]
        )
    lines.extend(["", "[Protected Key Fields]"])
    for name, values in report["key_fields"].items():
        lines.append(
            f"{name}: baseline={values['baseline']!r} | "
            f"control={values['control']!r} | matches={values['matches']}"
        )
    lines.extend(["", "[Safety]"])
    for name, value in report["safety"].items():
        lines.append(f"{name} = {value}")
    lines.extend(["", "[Errors]"])
    if report["errors"]:
        lines.extend(f"- {message}" for message in report["errors"])
    else:
        lines.append("none")
    lines.extend(["", "[JSON]", json.dumps(report, ensure_ascii=False, indent=2)])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    baseline_path = args.baseline_config.resolve()
    control_path = args.control_config.resolve()
    output_path = args.output.resolve()
    baseline_cfg = load_config(baseline_path)
    control_cfg = load_config(control_path)
    report = build_control_audit_report(baseline_cfg, control_cfg)
    report["baseline_config"] = str(baseline_path)
    report["control_config"] = str(control_path)

    try:
        routing_mode = validate_teacher_routing_config(control_cfg)
        routing_uses_ecst = teacher_routing_uses_ecst(control_cfg)
        report["teacher_routing_validation"] = {
            "mode": routing_mode,
            "uses_ecst": routing_uses_ecst,
        }
        if routing_mode != "none" or routing_uses_ecst:
            report["errors"].append(
                "control teacher routing is not the exact no-ECST identity path"
            )
    except Exception as error:  # Preserve the full audit report on failure.
        report["teacher_routing_validation"] = {
            "error": f"{type(error).__name__}: {error}"
        }
        report["errors"].append(
            "teacher routing validation failed: "
            f"{type(error).__name__}: {error}"
        )
    report["status"] = "PASS" if not report["errors"] else "FAIL"

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Audit output already exists; pass --overwrite to replace: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = format_report(report)
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"audit_output = {output_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
