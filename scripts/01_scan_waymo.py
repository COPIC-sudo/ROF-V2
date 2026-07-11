from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from _bootstrap import ROOT
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir, find_tfrecord_files
from rtbev.tfrecord_reader import iter_tfrecord_records
from rtbev.waymo_reader import quick_scan_scenario


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default=None, choices=["training", "validation", "testing"])
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-scenarios", type=int, default=None)
    ap.add_argument("--out-name", default="scan")
    args = ap.parse_args()

    cfg = load_config(args.config)
    split = args.split or cfg["dataset"].get("split", "validation")
    work_dir = Path(cfg["project"]["work_dir"])
    out_dir = ensure_dir(work_dir / "manifests")
    files = find_tfrecord_files(Path(cfg["dataset"]["scenario_root"]), split)
    if args.max_files is not None:
        files = files[: args.max_files]

    rows, total, kept = [], 0, 0
    for path in tqdm(files, desc=f"scan:{split}"):
        for rec_idx, raw in iter_tfrecord_records(path):
            total += 1
            info = quick_scan_scenario(raw, cfg)
            info["file_path"] = str(path)
            info["record_index"] = rec_idx
            rows.append(info)
            kept += int(bool(info["keep"]))
            if args.max_scenarios is not None and total >= args.max_scenarios:
                break
        if args.max_scenarios is not None and total >= args.max_scenarios:
            break
    df = pd.DataFrame(rows)
    out_csv = out_dir / f"{args.out_name}_candidates.csv"
    keep_csv = out_dir / f"{args.out_name}_candidates_kept.csv"
    df.to_csv(out_csv, index=False)
    df[df["keep"] == True].to_csv(keep_csv, index=False)  # noqa:E712
    print(f"[scan] total scenarios = {total}")
    print(f"[scan] kept scenarios  = {kept}")
    print(f"[scan] wrote {out_csv}")
    print(f"[scan] wrote {keep_csv}")

if __name__ == "__main__":
    main()
