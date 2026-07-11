from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.ops import unary_union

from ..utils import nearest_value, ensure_dir, wrap_to_pi, transform_points_global_to_local
from .geometry_bins import alpha_shape_polygon
from .par_parse import derive_params
from .rom_models import VehicleParams, simulate_bicycle
from .stability import stability_filter



def _classify_action_family(delta_rad: float, throttle: float, brake: float) -> str:
    """Coarse action family used by NC-minimal actionability metrics.

    The fallback primitive generator samples constant steering/brake/throttle
    commands.  This helper assigns each primitive to an interpretable family so
    downstream features can report left/right/brake survival rather than only a
    single MSR scalar.
    """
    delta = float(delta_rad)
    th = float(throttle)
    br = float(brake)
    steer_thr = 0.05
    brake_thr = 0.05
    hard_brake_thr = 0.6
    if abs(delta) <= steer_thr and br <= brake_thr:
        return "accelerate" if th > 0.05 else "keep"
    if abs(delta) <= steer_thr and br > brake_thr:
        return "hard_brake" if br >= hard_brake_thr else "brake"
    side = "left" if delta > 0 else "right"
    if br > brake_thr:
        return f"brake_{side}"
    return side


def _primitive_action_cost(delta_rad: float, throttle: float, brake: float, cfg: dict) -> float:
    fb = cfg.get("tube", {}).get("fallback", {})
    steering_values = [abs(float(x)) for x in fb.get("steering_values_rad", [0.35])]
    brake_values = [abs(float(x)) for x in fb.get("brake_values", [1.0])]
    throttle_values = [abs(float(x)) for x in fb.get("throttle_values", [1.0])]
    max_delta = max(max(steering_values), 1e-6)
    max_brake = max(max(brake_values), 1e-6)
    max_throttle = max(max(throttle_values), 1e-6)
    # Low cost means easy/natural action; high cost means aggressive maneuver.
    return float(0.50 * abs(float(delta_rad)) / max_delta + 0.35 * abs(float(brake)) / max_brake + 0.15 * abs(float(throttle)) / max_throttle)

@dataclass
class TubeUnionRecord:
    mu: float
    v0_kph: int
    query_dt_s: float
    horizon_s: float
    time_s: list[float]
    slice_wkts: list[str | None]
    _geom_cache: dict[int, object] = field(default_factory=dict, init=False, repr=False)
    _time_array_cache: np.ndarray | None = field(default=None, init=False, repr=False)

    def to_json_dict(self) -> dict:
        return {
            "mu": self.mu,
            "v0_kph": self.v0_kph,
            "query_dt_s": self.query_dt_s,
            "horizon_s": self.horizon_s,
            "time_s": self.time_s,
            "slice_wkts": self.slice_wkts,
        }

    @staticmethod
    def from_json_dict(d: dict) -> "TubeUnionRecord":
        return TubeUnionRecord(
            float(d["mu"]),
            int(d["v0_kph"]),
            float(d.get("query_dt_s", 0.1)),
            float(d.get("horizon_s", 2.0)),
            [float(x) for x in d["time_s"]],
            list(d["slice_wkts"]),
        )

    def _time_array(self) -> np.ndarray:
        if self._time_array_cache is None:
            self._time_array_cache = np.asarray(self.time_s, dtype=np.float64)
        return self._time_array_cache

    def slice_index_at_time(self, target_time_s: float) -> int | None:
        arr = self._time_array()
        if arr.size == 0:
            return None
        return int(np.argmin(np.abs(arr - float(target_time_s))))

    def get_slice_at_time(self, target_time_s: float):
        """Return a cached shapely geometry for the nearest physical time slice.

        The old implementation parsed the WKT string on every agent/time query.
        Feature generation repeatedly queries the same (mu, speed, time) slices, so
        this cache removes thousands of WKT parse calls without changing results.
        """
        idx = self.slice_index_at_time(target_time_s)
        if idx is None or idx < 0 or idx >= len(self.slice_wkts):
            return None
        if idx in self._geom_cache:
            return self._geom_cache[idx]
        w = self.slice_wkts[idx]
        geom = None if (w is None or w == "") else shapely_wkt.loads(w)
        self._geom_cache[idx] = geom
        return geom


@dataclass
class PrimitiveRecord:
    mu: float
    v0_kph: int
    time_s: np.ndarray           # [K0]
    x: np.ndarray                # [M,K0]
    y: np.ndarray                # [M,K0]
    heading: np.ndarray          # [M,K0]
    weight: np.ndarray           # [M]
    primitive_id: np.ndarray     # [M]
    action_family: np.ndarray    # [M], e.g. keep/brake/left/right/brake_left
    action_cost: np.ndarray      # [M], normalized maneuver effort cost in [0, +inf)

    @property
    def count(self) -> int:
        return int(self.x.shape[0])


class TubeLibrary:
    def __init__(self, records: dict[tuple[float, int], TubeUnionRecord]):
        self.records = records

    @classmethod
    def from_workdir(cls, lib_dir: Path) -> "TubeLibrary":
        recs: dict[tuple[float, int], TubeUnionRecord] = {}
        for p in sorted(lib_dir.glob("tube_union_mu*_v*.json")):
            obj = json.loads(p.read_text(encoding="utf-8"))
            rec = TubeUnionRecord.from_json_dict(obj)
            recs[(float(rec.mu), int(rec.v0_kph))] = rec
        if not recs:
            raise FileNotFoundError(f"no tube_union_*.json found under {lib_dir}")
        return cls(recs)

    def select(self, mu: float, speed_kph: float, mu_bins: list[float], speed_bins_kph: list[int]) -> TubeUnionRecord:
        mu_key = float(nearest_value(mu, mu_bins))
        v_key = int(round(nearest_value(speed_kph, speed_bins_kph)))
        key = (mu_key, v_key)
        if key not in self.records:
            raise KeyError(f"tube record not found for key={key}; available={sorted(self.records)}")
        return self.records[key]


class PrimitiveLibrary:
    def __init__(self, records: dict[tuple[float, int], PrimitiveRecord]):
        self.records = records

    @classmethod
    def from_workdir(cls, lib_dir: Path) -> "PrimitiveLibrary":
        recs: dict[tuple[float, int], PrimitiveRecord] = {}
        for p in sorted(lib_dir.glob("primitive_mu*_v*.npz")):
            data = np.load(p, allow_pickle=True)
            mu = float(data["mu"])
            v0 = int(data["v0_kph"])
            M = data["x"].shape[0]
            action_family = data["action_family"] if "action_family" in data.files else np.asarray(["unknown"] * M)
            action_cost = data["action_cost"] if "action_cost" in data.files else np.ones(M, dtype=np.float64)
            recs[(mu, v0)] = PrimitiveRecord(
                mu=mu,
                v0_kph=v0,
                time_s=np.asarray(data["time_s"], dtype=np.float64),
                x=np.asarray(data["x"], dtype=np.float64),
                y=np.asarray(data["y"], dtype=np.float64),
                heading=np.asarray(data["heading"], dtype=np.float64),
                weight=np.asarray(data.get("weight", np.ones(M)), dtype=np.float64),
                primitive_id=np.asarray(data.get("primitive_id", np.arange(M)), dtype=np.int64),
                action_family=np.asarray(action_family).astype(str),
                action_cost=np.asarray(action_cost, dtype=np.float64),
            )
        return cls(recs)

    @property
    def available(self) -> bool:
        return bool(self.records)

    def select(self, mu: float, speed_kph: float, mu_bins: list[float], speed_bins_kph: list[int]) -> PrimitiveRecord | None:
        if not self.records:
            return None
        mu_key = float(nearest_value(mu, mu_bins))
        v_key = int(round(nearest_value(speed_kph, speed_bins_kph)))
        return self.records.get((mu_key, v_key))


def _load_union_polys_from_record(rec: TubeUnionRecord):
    out = []
    for idx, w in enumerate(rec.slice_wkts):
        if idx in rec._geom_cache:
            out.append(rec._geom_cache[idx])
            continue
        geom = None if (w is None or w == "") else shapely_wkt.loads(w)
        rec._geom_cache[idx] = geom
        out.append(geom)
    return out


def load_union_polys_at_times(rec: TubeUnionRecord, query_times_s) -> list:
    """Return tube polygons aligned by physical time, not by array index.

    This is important when old manifests are stored at 100 Hz/10 ms while the
    ROF pipeline queries 10 Hz/0.1 s slices.
    """
    polys = _load_union_polys_from_record(rec)
    times = np.asarray(rec.time_s, dtype=float)
    out = []
    for t in np.asarray(query_times_s, dtype=float):
        if len(times) == 0:
            out.append(None)
            continue
        idx = int(np.argmin(np.abs(times - float(t))))
        out.append(polys[idx] if 0 <= idx < len(polys) else None)
    return out


def get_record_slice_at_time(rec: TubeUnionRecord, target_time_s: float):
    """Return the cached slice polygon nearest to a physical query time.

    This fixes the common error of indexing old 100 Hz tube manifests by 0.1 s
    query indices.  Old tube_layered manifests may have dt=0.01, while ROF uses
    0.1 s Waymo slices.  Always query by physical time, not by list index.
    """
    return rec.get_slice_at_time(target_time_s)


def get_primitive_aligned(rec: PrimitiveRecord, target_times_s: Iterable[float]):
    times = np.asarray(list(target_times_s), dtype=np.float64)
    src = np.asarray(rec.time_s, dtype=np.float64)
    idx = np.asarray([int(np.argmin(np.abs(src - t))) for t in times], dtype=np.int64)
    return {
        "x": rec.x[:, idx],
        "y": rec.y[:, idx],
        "heading": rec.heading[:, idx],
        "weight": rec.weight,
        "primitive_id": rec.primitive_id,
        "action_family": rec.action_family,
        "action_cost": rec.action_cost,
        "times_s": times,
    }


def import_existing_layered_manifests(src_dir: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    n = 0
    for p in sorted(src_dir.glob("tube_layered_*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        mu = float(obj["mu"])
        v0 = int(obj["v0"])
        dt = float(obj["dt"])
        horizon = float(obj["horizon"])
        layered = obj["layered"]
        time_s: list[float] = []
        slice_wkts: list[str | None] = []
        for sl in layered:
            k = int(sl["time_index"])
            # old builder used k from 0; physical time should be (k+1)*dt
            time_s.append(round((k + 1) * dt, 6))
            polys = []
            for b in sl.get("bins", []):
                w = b.get("wkt")
                if not w:
                    continue
                try:
                    polys.append(shapely_wkt.loads(w))
                except Exception:
                    continue
            if not polys:
                slice_wkts.append(None)
                continue
            u = unary_union(polys)
            try:
                if not u.is_valid:
                    u = u.buffer(0)
            except Exception:
                pass
            slice_wkts.append(u.wkt if not u.is_empty else None)
        rec = TubeUnionRecord(mu=mu, v0_kph=v0, query_dt_s=dt, horizon_s=horizon, time_s=time_s, slice_wkts=slice_wkts)
        out = out_dir / f"tube_union_mu{mu}_v{v0}.json"
        out.write_text(json.dumps(rec.to_json_dict(), ensure_ascii=False), encoding="utf-8")
        n += 1
    if n == 0:
        raise FileNotFoundError(f"no tube_layered_*.json found under {src_dir}")


def _make_vehicle_cfg(cfg: dict, par_path: Path | None) -> dict:
    veh = dict(cfg["vehicle"])
    if par_path is not None and par_path.exists():
        veh.update(derive_params(str(par_path)))
    return veh


def _simulate_constant_primitives(mu: float, v0_kph: int, cfg: dict, veh_cfg: dict):
    fb = cfg["tube"]["fallback"]
    sim_dt = float(fb["sim_dt_s"])
    horizon = float(cfg["tube"]["horizon_s"])
    t = np.arange(0.0, horizon + 1e-9, sim_dt)
    steering_values = [float(x) for x in fb["steering_values_rad"]]
    throttle_values = [float(x) for x in fb["throttle_values"]]
    brake_values = [float(x) for x in fb["brake_values"]]
    ax_coeff = dict(fb.get("longitudinal_model", {}))

    vp = VehicleParams(
        float(veh_cfg["m_kg"]), float(veh_cfg["Iz_kgm2"]), float(veh_cfg["lf_m"]), float(veh_cfg["lr_m"]),
        float(veh_cfg["track_m"]), float(veh_cfg["h_cg_m"]),
        float(veh_cfg["Cf0_Nprad_front"]) * float(mu), float(veh_cfg["Cr0_Nprad_rear"]) * float(mu),
    )

    trajs = []
    primitive_ids = []
    action_families = []
    action_costs = []
    pid = 0
    for delta0 in steering_values:
        for th in throttle_values:
            for br in brake_values:
                if th > 0.0 and br > 0.0:
                    continue
                delta = np.full_like(t, delta0, dtype=float)
                throttle = np.full_like(t, th, dtype=float)
                brake = np.full_like(t, br, dtype=float)
                sim = simulate_bicycle(t, delta, throttle, brake, float(mu), 0.0, 0.0, 0.0, float(v0_kph) / 3.6, vp, ax_coeff=ax_coeff)
                tr = {"v_mps": sim["v_total"], "beta_rad": sim["beta"], "r_rps": sim["r"], "ax": sim["ax_body"], "ay": sim["ay_body"], "yaw_rad": sim["psi"]}
                ok = stability_filter(tr, cfg, veh_cfg, mu)["ok"]
                if ok:
                    trajs.append(sim)
                    primitive_ids.append(pid)
                    action_families.append(_classify_action_family(delta0, th, br))
                    action_costs.append(_primitive_action_cost(delta0, th, br, cfg))
                pid += 1
    return t, trajs, primitive_ids, action_families, action_costs


def build_fallback_union_and_primitive_library(cfg: dict, out_dir: Path, par_path: Path | None) -> None:
    out_dir = ensure_dir(out_dir)
    veh_cfg = _make_vehicle_cfg(cfg, par_path)
    alpha_cfg = cfg["tube"]["alpha"]
    query_dt_s = float(cfg["tube"]["query_dt_s"])
    horizon_s = float(cfg["tube"]["horizon_s"])
    query_times = np.arange(query_dt_s, horizon_s + 1e-9, query_dt_s)

    for mu in [float(x) for x in cfg["tube"]["mu_bins"]]:
        for v0 in [int(x) for x in cfg["tube"]["speed_bins_kph"]]:
            t_sim, trajs, primitive_ids, action_families, action_costs = _simulate_constant_primitives(mu, v0, cfg, veh_cfg)
            if not trajs:
                raise RuntimeError(f"no stable fallback trajectories for mu={mu}, v0={v0}")
            slice_wkts = []
            prim_x = []
            prim_y = []
            prim_h = []
            for tr in trajs:
                idxs = [int(np.argmin(np.abs(t_sim - qt))) for qt in query_times]
                prim_x.append(np.asarray(tr["x"], dtype=float)[idxs])
                prim_y.append(np.asarray(tr["y"], dtype=float)[idxs])
                prim_h.append(np.asarray(tr["psi"], dtype=float)[idxs])
            for j, qt in enumerate(query_times):
                pts = np.array([[px[j], py[j]] for px, py in zip(prim_x, prim_y)], dtype=np.float64)
                shp = alpha_shape_polygon(pts, alpha_cfg)
                slice_wkts.append(None if shp is None else shp.wkt)
            rec = TubeUnionRecord(mu=mu, v0_kph=v0, query_dt_s=query_dt_s, horizon_s=horizon_s, time_s=[float(x) for x in query_times], slice_wkts=slice_wkts)
            (out_dir / f"tube_union_mu{mu}_v{v0}.json").write_text(json.dumps(rec.to_json_dict(), ensure_ascii=False), encoding="utf-8")
            np.savez_compressed(
                out_dir / f"primitive_mu{mu}_v{v0}.npz",
                mu=np.asarray(mu), v0_kph=np.asarray(v0), time_s=query_times.astype(np.float32),
                x=np.asarray(prim_x, dtype=np.float32), y=np.asarray(prim_y, dtype=np.float32), heading=np.asarray(prim_h, dtype=np.float32),
                weight=np.ones(len(prim_x), dtype=np.float32), primitive_id=np.asarray(primitive_ids, dtype=np.int64),
                action_family=np.asarray(action_families).astype("U32"), action_cost=np.asarray(action_costs, dtype=np.float32),
            )


# Backward-compatible name used by older script.
def build_fallback_union_library(cfg: dict, out_dir: Path, par_path: Path | None) -> None:
    build_fallback_union_and_primitive_library(cfg, out_dir, par_path)


def _read_old_run_csv(path: Path, horizon_s: float = 2.0):
    df = pd.read_csv(path)
    tcol = "Time" if "Time" in df.columns else "T_Stamp" if "T_Stamp" in df.columns else None
    if tcol is None:
        dt = 0.01
        n = int(horizon_s / dt) + 1
        df = df.iloc[:n].copy()
        df["Time"] = np.arange(len(df)) * dt
        tcol = "Time"
    else:
        t0 = float(df[tcol].iloc[0])
        df = df[(df[tcol] - t0) <= horizon_s + 1e-9].copy()
        df[tcol] = df[tcol] - t0

    def get(*names):
        for c in names:
            if c in df.columns:
                return df[c].to_numpy(dtype=float)
        return None

    x = get("Xcg_SM", "X", "x")
    y = get("Ycg_SM", "Y", "y")
    if x is None or y is None:
        raise RuntimeError(f"Missing X/Y columns in {path}")
    yaw_deg = get("Yaw", "Yaw (deg)")
    yaw_rad = get("yaw_rad", "psi_rad")
    if yaw_rad is None:
        yaw_rad = np.deg2rad(yaw_deg) if yaw_deg is not None else np.zeros_like(x)
    t = df[tcol].to_numpy(dtype=float)
    pts = np.stack([x, y], axis=1)
    local = transform_points_global_to_local(pts, np.asarray([x[0], y[0]], dtype=float), float(yaw_rad[0]))
    heading = wrap_to_pi(yaw_rad - float(yaw_rad[0]))
    return t, local[:, 0], local[:, 1], heading


def _parse_mu_v_from_name(name: str):
    m = re.search(r"mu([0-9.]+)_v([0-9]+)", name)
    if not m:
        return None
    mu = float(m.group(1).rstrip("."))
    v = int(m.group(2))
    return mu, v


def build_primitive_library_from_stable_runs(stable_dir: Path, out_dir: Path, cfg: dict, raw_root: Path | None = None) -> None:
    """Build primitive_mu*_v*.npz from old stable_runs_mu*_v*.csv files.

    The old tube project stores stable primitive file paths in stable_runs CSVs.
    This function reads those raw run CSVs and exports primitive trajectories
    required by MSR. If a stored run path is stale, raw_root is used as a fallback
    lookup by basename.
    """
    out_dir = ensure_dir(out_dir)
    horizon = float(cfg["tube"].get("horizon_s", 2.0))
    qdt = float(cfg["tube"].get("query_dt_s", 0.1))
    query_times = np.arange(qdt, horizon + 1e-9, qdt)
    count_files = 0
    for csv_path in sorted(stable_dir.glob("stable_runs_mu*_v*.csv")):
        parsed = _parse_mu_v_from_name(csv_path.name)
        if parsed is None:
            continue
        mu, v0 = parsed
        runs = pd.read_csv(csv_path)
        col = "run" if "run" in runs.columns else runs.columns[0]
        xs, ys, hs, weights, ids, families, costs = [], [], [], [], [], [], []
        missing = 0
        for pid, run_str in enumerate(runs[col].dropna().astype(str).tolist()):
            p = Path(run_str)
            if not p.exists():
                candidates = []
                if raw_root is not None:
                    candidates.extend(raw_root.rglob(p.name))
                candidates.extend(stable_dir.parent.parent.rglob(p.name))
                p = next((c for c in candidates if c.exists()), p)
            if not p.exists():
                missing += 1
                continue
            try:
                t, x, y, h = _read_old_run_csv(p, horizon_s=horizon)
            except Exception:
                missing += 1
                continue
            idx = [int(np.argmin(np.abs(t - qt))) for qt in query_times]
            xs.append(np.asarray(x)[idx])
            ys.append(np.asarray(y)[idx])
            hs.append(np.asarray(h)[idx])
            weights.append(1.0)
            ids.append(pid)
            families.append("unknown")
            costs.append(1.0)
        if not xs:
            print(f"[primitive] no usable primitive from {csv_path} (missing={missing})")
            continue
        np.savez_compressed(
            out_dir / f"primitive_mu{mu}_v{v0}.npz",
            mu=np.asarray(mu), v0_kph=np.asarray(v0), time_s=query_times.astype(np.float32),
            x=np.asarray(xs, dtype=np.float32), y=np.asarray(ys, dtype=np.float32), heading=np.asarray(hs, dtype=np.float32),
            weight=np.asarray(weights, dtype=np.float32), primitive_id=np.asarray(ids, dtype=np.int64),
            action_family=np.asarray(families).astype("U32"), action_cost=np.asarray(costs, dtype=np.float32),
        )
        print(f"[primitive] exported mu={mu}, v={v0}, primitives={len(xs)}, missing={missing}")
        count_files += 1
    if count_files == 0:
        raise FileNotFoundError(f"no primitive files exported from stable_runs under {stable_dir}")
