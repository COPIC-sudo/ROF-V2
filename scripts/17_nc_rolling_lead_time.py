from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ROOT  # noqa: F401
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir
from rtbev.nc_eval import score_definitions, y_for_task


def _scenario_split(keys: np.ndarray, calibration_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    keys = np.asarray(sorted(set(map(str, keys))))
    rng = np.random.default_rng(seed)
    keys = keys.copy(); rng.shuffle(keys)
    n_cal = max(1, int(round(len(keys) * calibration_fraction))) if len(keys) > 1 else len(keys)
    cal = set(keys[:n_cal]); test = set(keys[n_cal:])
    if not test and len(keys) > 1:
        test = set(keys[n_cal-1:]); cal = set(keys[:n_cal-1])
    return cal, test


def _threshold_for_fpr(y: np.ndarray, score: np.ndarray, fpr: float) -> float:
    neg = score[y == 0]
    neg = neg[np.isfinite(neg)]
    if len(neg) == 0:
        return np.nan
    return float(np.quantile(neg, 1.0 - float(fpr)))


def _event_onset_for_group(times: np.ndarray, y: np.ndarray, min_pre_s: float) -> tuple[bool, float, int, float]:
    # First negative -> positive transition.  Events already positive at the first
    # frame are excluded from lead-time summaries because no pre-event window exists.
    if len(y) < 2:
        return False, np.nan, -1, 0.0
    for i in range(1, len(y)):
        if y[i-1] == 0 and y[i] == 1:
            pre = float(times[i] - times[0])
            if pre >= float(min_pre_s):
                return True, float(times[i]), int(i), pre
            return False, np.nan, -1, pre
    return False, np.nan, -1, float(times[-1] - times[0]) if len(times) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="NC v1.1 rolling lead-time analysis with event-onset and calibration/test split.")
    ap.add_argument("--config", default="configs/nc_minimal.yaml")
    ap.add_argument("--features-csv", required=True, help="e.g. <work>/features/rof_features_rolling.csv")
    ap.add_argument("--task", choices=["warning_or_above", "emergency_only", "safe_vs_risky"], default="warning_or_above")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg["project"]["work_dir"])
    out = ensure_dir(work / "results" / "nc_rolling_lead_time")
    df = pd.read_csv(args.features_csv).copy()
    if "current_time_s" not in df.columns and "current_time_index" in df.columns:
        df["current_time_s"] = pd.to_numeric(df["current_time_index"], errors="coerce") * float(cfg["labels"].get("dt_s", 0.1))
    if "current_time_s" not in df.columns:
        raise SystemExit("Rolling feature CSV must contain current_time_s or current_time_index. Use 03b_extract_rolling_waymo.py --scan-all-valid + 04_generate_rof_features.py --out-name rof_features_rolling.csv")
    df["label_id"] = df["label_id"].astype(int)
    df = df.sort_values(["scenario_id", "current_time_s", "sample_id"]).reset_index(drop=True)
    y_all = y_for_task(df["label_id"].to_numpy(), args.task)
    score_map = score_definitions(df)
    ncfg = cfg.get("nc_experiments", {})
    fprs = [float(x) for x in ncfg.get("fixed_fpr_levels", [0.01, 0.05])]
    min_pre_s = float(ncfg.get("lead_time_min_pre_event_s", 2.0))
    cal_frac = float(ncfg.get("lead_time_calibration_fraction", 0.35))
    cal_keys, test_keys = _scenario_split(df["scenario_id"].astype(str).unique(), cal_frac, int(args.seed))
    cal_mask = df["scenario_id"].astype(str).isin(cal_keys).to_numpy()
    test_mask = df["scenario_id"].astype(str).isin(test_keys).to_numpy()

    onset_rows=[]
    scenario_info={}
    for sid, sub_idx in df.groupby("scenario_id", sort=False).groups.items():
        idx=np.asarray(list(sub_idx), dtype=int)
        sub=df.iloc[idx].sort_values("current_time_s")
        idx=sub.index.to_numpy(int)
        tt=pd.to_numeric(sub["current_time_s"], errors="coerce").to_numpy(float)
        yy=y_all[idx]
        ok,event_t,event_i,pre_dur=_event_onset_for_group(tt, yy, min_pre_s)
        scenario_info[str(sid)]={"has_valid_onset":ok,"event_time_s":event_t,"event_index_local":event_i,"pre_event_duration_s":pre_dur,"has_any_event":bool(np.any(yy==1)),"n_frames":len(idx)}
        onset_rows.append({"scenario_id":sid,"task":args.task,**scenario_info[str(sid)]})
    pd.DataFrame(onset_rows).to_csv(out / f"event_onsets_{args.task}.csv", index=False)

    rows=[]; traj_rows=[]
    for score_name, score in score_map.items():
        for fpr in fprs:
            thr = _threshold_for_fpr(y_all[cal_mask], score[cal_mask], fpr)
            if not np.isfinite(thr):
                continue
            for sid, sub_idx in df.loc[test_mask].groupby("scenario_id", sort=False).groups.items():
                # groupby on filtered df gives original index labels.
                sub=df.loc[list(sub_idx)].sort_values("current_time_s")
                idx=sub.index.to_numpy(int)
                tt=pd.to_numeric(sub["current_time_s"], errors="coerce").to_numpy(float)
                yy=y_all[idx]
                ss=score[idx]
                info=scenario_info[str(sid)]
                alert=ss >= thr
                if info["has_valid_onset"]:
                    event_t=float(info["event_time_s"])
                    pre=tt < event_t
                    pre_alert=alert & pre
                    first_pre=float(tt[pre_alert][0]) if np.any(pre_alert) else np.nan
                    lead=float(event_t-first_pre) if np.isfinite(first_pre) else np.nan
                    rows.append({"task":args.task,"score_name":score_name,"target_fpr":fpr,"threshold":thr,"split":"test","scenario_id":sid,"has_event":True,"has_valid_onset":True,"event_time_s":event_t,"pre_event_duration_s":float(info["pre_event_duration_s"]),"first_pre_event_alert_time_s":first_pre,"lead_time_s":lead,"missed_event":not np.any(pre_alert),"false_alert_non_event":False})
                    rel=tt-event_t
                    for tr,sv,yyv in zip(rel, ss, yy):
                        if -5.0 <= tr <= 1.0:
                            traj_rows.append({"scenario_id":sid,"score_name":score_name,"target_fpr":fpr,"time_to_event_s":float(tr),"score":float(sv),"y":int(yyv)})
                else:
                    # Non-event or invalid onset scenario.  Use it to estimate false alerts,
                    # but keep invalid-onset events separate from clean non-events.
                    is_non_event=not bool(info["has_any_event"])
                    rows.append({"task":args.task,"score_name":score_name,"target_fpr":fpr,"threshold":thr,"split":"test","scenario_id":sid,"has_event":bool(info["has_any_event"]),"has_valid_onset":False,"event_time_s":np.nan,"pre_event_duration_s":float(info["pre_event_duration_s"]),"first_pre_event_alert_time_s":np.nan,"lead_time_s":np.nan,"missed_event":False,"false_alert_non_event":bool(is_non_event and np.any(alert))})

    res=pd.DataFrame(rows)
    res.to_csv(out / f"lead_time_events_{args.task}.csv", index=False)
    pd.DataFrame(traj_rows).to_csv(out / f"pre_event_score_trajectories_{args.task}.csv", index=False)
    summary=[]
    if not res.empty:
        for (score_name,fpr), sub in res.groupby(["score_name","target_fpr"]):
            ev=sub[(sub["has_valid_onset"]==True)]
            nonev=sub[(sub["has_event"]==False)]
            invalid_ev=sub[(sub["has_event"]==True)&(sub["has_valid_onset"]==False)]
            summary.append({"task":args.task,"score_name":score_name,"target_fpr":float(fpr),"n_event_scenarios":int(len(ev)),"n_invalid_event_scenarios":int(len(invalid_ev)),"n_non_event_scenarios":int(len(nonev)),"median_lead_time_s":float(pd.to_numeric(ev["lead_time_s"],errors="coerce").median()) if len(ev) else np.nan,"mean_lead_time_s":float(pd.to_numeric(ev["lead_time_s"],errors="coerce").mean()) if len(ev) else np.nan,"missed_event_rate":float(ev["missed_event"].mean()) if len(ev) else np.nan,"false_alert_non_event_rate":float(nonev["false_alert_non_event"].mean()) if len(nonev) else np.nan})
    pd.DataFrame(summary).to_csv(out / f"lead_time_summary_{args.task}.csv", index=False)
    print(f"[nc-leadtime-v1.1] wrote {out}")


if __name__ == "__main__":
    main()
