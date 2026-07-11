from __future__ import annotations

import numpy as np


def _f(x, default=None):
    try:
        return float(x)
    except Exception:
        if default is not None:
            return float(default)
        raise


def stability_filter(traj: dict, cfg: dict, veh: dict, mu: float) -> dict:
    stab_cfg = cfg.get("stability", cfg)
    eta_beta = _f(stab_cfg.get("eta_beta", 0.95))
    eta_r = _f(stab_cfg.get("eta_r", 0.95))
    eta_ltr = _f(stab_cfg.get("eta_ltr", 0.95))
    eta_ax = _f(stab_cfg.get("eta_ax", 0.95))
    eps_v = _f(stab_cfg.get("eps_v_mps", 0.5))
    mu = _f(mu)
    g = 9.81

    N = len(traj["v_mps"])
    ok_mask = np.ones(N, dtype=bool)
    reasons = {}

    beta_max = eta_beta * np.arctan(mu)
    s1 = np.abs(traj["beta_rad"]) < beta_max
    ok_mask &= s1
    reasons["S1_beta"] = bool(np.all(s1))

    s4 = np.abs(traj["ax"]) <= (eta_ax * mu * g)
    ok_mask &= s4
    reasons["S4_ax"] = bool(np.all(s4))

    v_eff = np.maximum(traj["v_mps"], eps_v)
    r_lim = eta_r * mu * g / v_eff
    s2 = np.abs(traj["r_rps"]) < r_lim
    ok_mask &= s2
    reasons["S2_r"] = bool(np.all(s2))

    use_load = all(k in traj and traj[k] is not None for k in ["Fz_L1", "Fz_L2", "Fz_R1", "Fz_R2"])
    if use_load:
        FzL = traj["Fz_L1"] + traj["Fz_L2"]
        FzR = traj["Fz_R1"] + traj["Fz_R2"]
        denom = np.maximum(FzL + FzR, 1e-3)
        ltr = (FzL - FzR) / denom
        ltr_max_abs = _f(stab_cfg.get("ltr_max_abs", 0.95))
        s3 = np.abs(ltr) < ltr_max_abs
    else:
        track_m = _f(veh.get("track_m", 1.55))
        h_cg_m = _f(veh.get("h_cg_m", 0.55))
        s3 = np.abs(h_cg_m * traj["ay"]) < (0.5 * track_m * g * eta_ltr)
    ok_mask &= s3
    reasons["S3_ltr"] = bool(np.all(s3))

    Cf0 = _f(veh["Cf0_Nprad_front"])
    Cr0 = _f(veh["Cr0_Nprad_rear"])
    Cf = mu * Cf0
    Cr = mu * Cr0
    m = _f(veh["m_kg"])
    Iz = _f(veh["Iz_kgm2"])
    a = _f(veh["lf_m"])
    b = _f(veh["lr_m"])

    v = np.maximum(traj["v_mps"], eps_v)
    a11 = -(Cf + Cr) / (m * v)
    a12 = -1.0 - (a * Cf - b * Cr) / (m * (v * v) + 1e-9)
    a21 = -(a * Cf - b * Cr) / Iz
    a22 = -(a * a * Cf + b * b * Cr) / (Iz * v)

    trm = a11 + a22
    det = a11 * a22 - a12 * a21
    s5 = (trm < 0.0) & (det > 0.0)
    ok_mask &= s5
    reasons["S5_spec"] = bool(np.all(s5))

    return {"ok": bool(np.all(ok_mask)), "mask": ok_mask, "reasons": reasons}
