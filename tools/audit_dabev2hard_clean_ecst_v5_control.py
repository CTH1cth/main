#!/usr/bin/env python3
"""Audit DABE-v2 Hard static BCE against unchanged A1 Clean-ECST v5."""

import argparse
import json
import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dabev2hard_clean_ecst_v5_control import (  # noqa: E402
    CONTROL_CONFIG_PATH,
    PARENT_CONFIG_PATH,
    build_control_audit_report,
)
from common.utils import load_config  # noqa: E402


DEFAULT_OUTPUT = (
    MAIN_ROOT / "analysis/dabev2hard_clean_ecst_v5_config_diff.txt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the A1 Clean-ECST v5 parent and DABE-v2 Hard child, "
            "then verify the exact five-field static-target-only change."
        )
    )
    parser.add_argument("--parent-config", type=Path, default=PARENT_CONFIG_PATH)
    parser.add_argument("--control-config", type=Path, default=CONTROL_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing audit text file.",
    )
    return parser.parse_args()


def format_report(report):
    lines = [
        "DABE-v2 Hard + A1 Clean-ECST v5 configuration audit",
        f"status = {report['status']}",
        f"parent_config = {report['parent_config']}",
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
                f"  parent = {record['parent']!r}",
                f"  control = {record['control']!r}",
            ]
        )

    lines.extend(["", "[Supervision Semantics]"])
    for name, value in report["supervision_semantics"].items():
        lines.append(f"{name} = {value!r}")

    lines.extend(["", "[Protected Fields]"])
    for name, values in report["protected_fields"].items():
        lines.append(
            f"{name}: parent={values['parent']!r} | "
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
    parent_path = args.parent_config.resolve()
    control_path = args.control_config.resolve()
    output_path = args.output.resolve()

    parent_cfg = load_config(parent_path)
    control_cfg = load_config(control_path)
    report = build_control_audit_report(parent_cfg, control_cfg)
    report["parent_config"] = str(parent_path)
    report["control_config"] = str(control_path)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            "Audit output already exists; pass --overwrite to replace: "
            f"{output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = format_report(report)
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"audit_output = {output_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
