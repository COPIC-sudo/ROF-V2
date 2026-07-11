from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir, write_gzip_pickle
from rtbev.tfrecord_reader import iter_tfrecord_records
from rtbev.waymo_reader import scenario_bytes_to_sample
from rtbev.labels import assign_label
from waymo_open_dataset.protos import scenario_pb2


def _scenario_meta(raw: bytes) -> tuple[int, int, float]:
    s = scenario_pb2.Scenario()
    s.ParseFromString(raw)
    dt = 0.1
    if len(s.timestamps_seconds) >= 2:
        dt = float(np.median(np.diff(np.asarray(s.timestamps_seconds, dtype=float))))
    return int(s.current_time_index), int(len(s.timestamps_seconds)), dt


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract rolling Waymo samples for event-onset lead-time analysis.")
    ap.add_argument("--config", default="configs/nc_minimal.yaml")
    ap.add_argument("--candidate-csv", required=True)
    ap.add_argument("--out-name", default="labels_rolling_full.csv")
    ap.add_argument("--limit-scenarios", type=int, default=None)
    ap.add_argument("--stride", type=int, default=2, help="time-index stride; Waymo is 10 Hz, so stride=2 gives 0.2 s")
    ap.add_argument("--start-index", type=int, default=0, help="absolute current_time_index start for full scan")
    ap.add_argument("--end-index", type=int, default=-1, help="absolute inclusive end; -1 means latest valid index with enough future")
    ap.add_argument("--offset-mode", action="store_true", help="legacy: use offsets from Waymo current_time_index instead of full scenario scan")
    ap.add_argument("--offset-start", type=int, default=-50, help="legacy offset-mode start")
    ap.add_argument("--offset-end", type=int, default=10, help="legacy offset-mode end")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work_dir = Path(cfg["project"]["work_dir"])
    samples_dir = ensure_dir(work_dir / "samples")
    labels_dir = ensure_dir(work_dir / "labels")
    cand = pd.read_csv(args.candidate_csv)
    if "keep" in cand.columns:
        cand = cand[cand["keep"] == True].copy()  # noqa:E712
    if args.limit_scenarios is not None:
        cand = cand.iloc[: args.limit_scenarios].copy()
    file_to_indices = defaultdict(dict)
    for _, row in cand.iterrows():
        file_to_indices[str(row["file_path"])][int(row["record_index"])] = row.to_dict()

    rows = []
    future_steps = int(cfg["dataset"].get("max_future_steps", 30))
    stride = max(1, int(args.stride))
    for file_path, wanted in tqdm(file_to_indices.items(), desc="rolling extract files"):
        for rec_idx, raw in iter_tfrecord_records(Path(file_path)):
            if rec_idx not in wanted:
                continue
            base_cur, n_ts, dt = _scenario_meta(raw)
            latest_valid = n_ts - future_steps - 1
            if latest_valid < 0:
                continue
            if args.offset_mode:
                cur_indices = [base_cur + off for off in range(int(args.offset_start), int(args.offset_end) + 1, stride)]
                cur_indices = [i for i in cur_indices if 0 <= i <= latest_valid]
            else:
                end = latest_valid if int(args.end_index) < 0 else min(int(args.end_index), latest_valid)
                start = max(0, int(args.start_index))
                cur_indices = list(range(start, end + 1, stride))
            for cur_idx in cur_indices:
                sample = scenario_bytes_to_sample(raw, cfg, current_time_index=cur_idx, sample_id_suffix=f"t{cur_idx:03d}")
                if sample is None:
                    continue
                metrics = assign_label(sample, cfg)
                sample["label"] = metrics
                write_gzip_pickle(samples_dir / f"{sample['sample_id']}.pkl.gz", sample)
                rows.append({
                    "sample_id": sample["sample_id"],
                    "scenario_id": sample["scenario_id"],
                    "root_scenario_id": sample["scenario_id"],
                    "file_path": file_path,
                    "record_index": rec_idx,
                    "base_current_time_index": int(base_cur),
                    "current_time_index": int(cur_idx),
                    "current_time_s": float(cur_idx * dt),
                    "relative_to_base_time_s": float((cur_idx - base_cur) * dt),
                    "agent_count": sample["agent_count"],
                    "ego_track_id": sample["ego_track_id"],
                    **metrics,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["scenario_id", "current_time_index", "sample_id"])
    out_csv = labels_dir / args.out_name
    df.to_csv(out_csv, index=False)
    print(f"[rolling-labels-v1.1] wrote {out_csv}; rows={len(df)}")
    if not df.empty:
        print(df["label_name"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
