#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys


def check_module(name: str, strict: bool) -> None:
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, "__version__", "OK"))
    except Exception as exc:
        print(name, "MISSING", type(exc).__name__, str(exc))
        if strict:
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--modules", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--check-commonroad-reader", action="store_true")
    parser.add_argument("--pip-list", action="store_true")
    args = parser.parse_args()
    print(f"ENV={args.env_name}")
    print(sys.executable)
    for module in [m.strip() for m in args.modules.split(",") if m.strip()]:
        check_module(module, strict=bool(args.strict and module != "commonroad_io"))
    if args.check_commonroad_reader:
        try:
            from commonroad.common.file_reader import CommonRoadFileReader

            print("CommonRoadFileReader", "OK", CommonRoadFileReader)
        except Exception as exc:
            print("CommonRoadFileReader", "MISSING", type(exc).__name__, str(exc))
            if args.strict:
                raise
    if args.pip_list:
        cp = subprocess.run([sys.executable, "-m", "pip", "list"], text=True, capture_output=True)
        print("PIP_LIST_BEGIN")
        print(cp.stdout)
        if cp.stderr:
            print("PIP_LIST_STDERR_BEGIN")
            print(cp.stderr)


if __name__ == "__main__":
    main()
