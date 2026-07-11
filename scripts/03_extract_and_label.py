from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from _bootstrap import ROOT
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir, write_gzip_pickle
from rtbev.tfrecord_reader import iter_tfrecord_records
from rtbev.waymo_reader import scenario_bytes_to_sample
from rtbev.labels import assign_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--candidate-csv", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-name", default="labels.csv", help="labels 输出文件名；用于 train/validation 分开保存")
    args = ap.parse_args()
    cfg = load_config(args.config)
    work_dir = Path(cfg["project"]["work_dir"])
    samples_dir = ensure_dir(work_dir / "samples")
    labels_dir = ensure_dir(work_dir / "labels")

    cand = pd.read_csv(args.candidate_csv)
    if "keep" in cand.columns:
        cand = cand[cand["keep"] == True].copy()  # noqa:E712
    if args.limit is not None:
        cand = cand.iloc[: args.limit].copy()
    if cand.empty:
        raise SystemExit("candidate csv 中没有样本")

    file_to_indices = defaultdict(dict)
    for _, row in cand.iterrows():
        file_to_indices[str(row["file_path"])][int(row["record_index"])] = row.to_dict()

    label_rows = []
    for file_path, wanted in tqdm(file_to_indices.items(), desc="extract+label files"):
        for rec_idx, raw in iter_tfrecord_records(Path(file_path)):
            if rec_idx not in wanted:
                continue
            sample = scenario_bytes_to_sample(raw, cfg)
            if sample is None:
                continue
            metrics = assign_label(sample, cfg)
            sample["label"] = metrics
            write_gzip_pickle(samples_dir / f"{sample['sample_id']}.pkl.gz", sample)
            label_rows.append({
                "sample_id": sample["sample_id"],
                "scenario_id": sample["scenario_id"],
                "file_path": file_path,
                "record_index": rec_idx,
                "agent_count": sample["agent_count"],
                "ego_track_id": sample["ego_track_id"],
                **metrics,
            })
    df = pd.DataFrame(label_rows).sort_values("sample_id")
    out_csv = labels_dir / args.out_name
    df.to_csv(out_csv, index=False)
    print(f"[labels] wrote {out_csv}")
    if not df.empty:
        print(df["label_name"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    main()
