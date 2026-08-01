#!/usr/bin/env python3
"""Audit the DABE-v2 Hard static-only configuration without running a model."""

import argparse
import json
import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dabev2hard_static_only import (  # noqa: E402
    PARENT_CONFIG_PATH,
    STATIC_ONLY_CONFIG_PATH,
    build_static_only_audit_report,
)
from common.utils import load_config  # noqa: E402


DEFAULT_OUTPUT = MAIN_ROOT / "analysis/dabev2hard_static_only_config_diff.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the no-ECST parent and static-only child, then verify "
            "that DABE-v2 Hard is the only effective supervision source."
        )
    )
    parser.add_argument("--parent-config", type=Path, default=PARENT_CONFIG_PATH)
    parser.add_argument(
        "--static-only-config", type=Path, default=STATIC_ONLY_CONFIG_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing static audit text file.",
    )
    return parser.parse_args()


def format_report(report):
    lines = [
        "DABE-v2 Hard static-only configuration audit",
        f"status = {report['status']}",
        f"parent_config = {report['parent_config']}",
        f"static_only_config = {report['static_only_config']}",
        "compared_uppercase_and_lowercase_field_count = "
        f"{report['compared_uppercase_and_lowercase_field_count']}",
        f"allowed_differences = {report['allowed_differences']}",
        "required_effective_differences = "
        f"{report['required_effective_differences']}",
        f"static_only_endpoints = {report['static_only_endpoints']}",
        "",
        "[Effective Differences]",
    ]
    for record in report["actual_differences"]:
        lines.extend(
            [
                f"field = {record['field']}",
                f"  parent = {record['parent']!r}",
                f"  static_only = {record['static_only']!r}",
            ]
        )

    lines.extend(["", "[Teacher Loss Switches]"])
    for name, values in report["teacher_loss_switches"].items():
        lines.append(
            f"{name}: parent={values['parent']!r} | "
            f"static_only={values['static_only']!r} | "
            f"disabled={values['disabled']}"
        )

    lines.extend(["", "[Protected Key Fields]"])
    for name, values in report["protected_key_fields"].items():
        lines.append(
            f"{name}: parent={values['parent']!r} | "
            f"static_only={values['static_only']!r} | "
            f"matches={values['matches']}"
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
    static_only_path = args.static_only_config.resolve()
    output_path = args.output.resolve()

    parent_cfg = load_config(parent_path)
    static_only_cfg = load_config(static_only_path)
    report = build_static_only_audit_report(parent_cfg, static_only_cfg)
    report["parent_config"] = str(parent_path)
    report["static_only_config"] = str(static_only_path)

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
