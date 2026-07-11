#!/usr/bin/env python3
"""Check whether the repository is safe and complete for a public GitHub release."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json", ".csv", ".cff", ".sh", ".ps1", ".bat"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "results", "data", "datasets", "work_dirs", "smoke_work"}
REQUIRED = [
    ".gitignore",
    "README.md",
    "README_zh.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "pyproject.toml",
    "environment.yml",
    "environment-commonroad.yml",
    "docs/REPRODUCE_PAPER.md",
    "configs/nc_v110/nc_v110_commonroad_full_fixed_taxonomy.yaml",
    "configs/nc_v111/nc_v111_decoupling_full.yaml",
    "configs/nc_v112/nc_v112_field_baselines_full_10k.yaml",
    "scripts/nc_v110/09_stratum_boundary_analysis.py",
    "scripts/nc_v111/02_decoupling_audit_full.py",
    "scripts/nc_v112/02_evaluate_extended_label_baselines_strict.py",
    "figure_tools/plot_nc_v11_figures_4_5_final_v2_2.py",
    "figure_tools/plot_supplementary_figures.py",
]
BANNED_NAMES = {".idea", ".agents", "dist", "github_readiness_report.md"}
BANNED_GLOBS = ["*_bak*.py", "*.pyc", "*.pyo", "*.zip", "*.7z", "*.tar", "*.tar.gz"]
ABSOLUTE_PATTERNS = {
    "windows_drive": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?![<>])"),
    "linux_home": re.compile(r"/(?:home|Users)/[^\s'\"`]+"),
    "sandbox_path": re.compile(r"/mnt/data/"),
}
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
}


def under_skipped(path: Path, root: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.relative_to(root).parts[:-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-file-mb", type=float, default=20.0)
    parser.add_argument("--report", default="github_readiness_report.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, list[dict[str, object]]] = {"absolute_paths": [], "secrets": [], "large_files": [], "banned": []}

    for rel in REQUIRED:
        if not (root / rel).exists():
            errors.append(f"Missing required path: {rel}")

    for name in BANNED_NAMES:
        for path in root.rglob(name):
            if path.resolve() == (root / args.report).resolve():
                continue
            details["banned"].append({"path": path.relative_to(root).as_posix()})
    for pattern in BANNED_GLOBS:
        for path in root.rglob(pattern):
            if path.is_file() and not under_skipped(path, root):
                details["banned"].append({"path": path.relative_to(root).as_posix()})
    if details["banned"]:
        errors.append(f"Found banned release artifacts: {len(details['banned'])}")

    max_bytes = int(args.max_file_mb * 1024 * 1024)
    for path in root.rglob("*"):
        if not path.is_file() or under_skipped(path, root):
            continue
        rel = path.relative_to(root).as_posix()
        if rel == args.report or rel == "scripts/99_check_github_readiness.py":
            continue
        size = path.stat().st_size
        if size > max_bytes:
            details["large_files"].append({"path": rel, "size_mb": round(size / 1024 / 1024, 3)})
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, regex in ABSOLUTE_PATTERNS.items():
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    details["absolute_paths"].append({"path": rel, "line": i, "pattern": name, "text": line.strip()[:200]})
        for name, regex in SECRET_PATTERNS.items():
            if regex.search(text):
                details["secrets"].append({"path": rel, "pattern": name})

    if details["absolute_paths"]:
        errors.append(f"Found machine-specific absolute paths: {len(details['absolute_paths'])}")
    if details["secrets"]:
        errors.append(f"Found possible secrets: {len(details['secrets'])}")
    if details["large_files"]:
        warnings.append(f"Found files larger than {args.max_file_mb:g} MB: {len(details['large_files'])}")

    report = {
        "root": "<REPOSITORY_ROOT>",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        **details,
    }
    report_path = root / args.report
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"status: {report['status']}")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")
    print(f"report: {report_path}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
