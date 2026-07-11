from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class VehicleParams:
    m_kg: float
    Iz_kgm2: float
    lf_m: float
    lr_m: float
    track_m: float
    h_cg_m: float
    Cf_Nprad: float
    Cr_Nprad: float
    g_mps2: float = 9.81


def ax_from_model(throttle: float, brake: float, coeff: Dict[str, float], mu: float, g: float) -> float:
    a0 = float(coeff.get("a0", 0.0))
    a_th = float(coeff.get("a_th", 0.0))
    a_br = float(coeff.get("a_br", 0.0))
    ax = a0 + a_th * float(throttle) - a_br * float(brake)
    ax = float(np.clip(ax, -abs(mu) * g, abs(mu) * g))
    return ax


def simulate_bicycle(time: np.ndarray, delta: np.ndarray, throttle: np.ndarray, brake: np.ndarray, mu: float, x0: float, y0: float, psi0: float, u0: float, params: VehicleParams, ax_coeff: Optional[Dict[str, float]] = None) -> dict:
    t = np.asarray(time, dtype=float).reshape(-1)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.05
    K = len(t)
    delta = np.asarray(delta, dtype=float).reshape(-1)
    throttle = np.asarray(throttle, dtype=float).reshape(-1)
    brake = np.asarray(brake, dtype=float).reshape(-1)

    x = np.zeros(K, float); y = np.zeros(K, float); psi = np.zeros(K, float)
    u = np.zeros(K, float); v = np.zeros(K, float); r = np.zeros(K, float)
    ax_body = np.zeros(K, float); ay_body = np.zeros(K, float)

    x[0], y[0], psi[0], u[0] = float(x0), float(y0), float(psi0), max(float(u0), 0.1)

    m = params.m_kg; Iz = params.Iz_kgm2; lf = params.lf_m; lr = params.lr_m
    Cf = params.Cf_Nprad; Cr = params.Cr_Nprad; g = params.g_mps2
    L = lf + lr; Fzf = m * g * lr / L; Fzr = m * g * lf / L
    ax_coeff = ax_coeff or {"a0": 0.0, "a_th": 2.0, "a_br": 8.0}

    for k in range(K - 1):
        uk = max(u[k], 0.1); vk = v[k]; rk = r[k]; psik = psi[k]; dk = float(delta[k])
        alpha_f = np.arctan2(vk + lf * rk, uk) - dk
        alpha_r = np.arctan2(vk - lr * rk, uk)
        Fy_f = -Cf * alpha_f; Fy_r = -Cr * alpha_r
        Fy_f = float(np.clip(Fy_f, -abs(mu) * Fzf, abs(mu) * Fzf))
        Fy_r = float(np.clip(Fy_r, -abs(mu) * Fzr, abs(mu) * Fzr))
        ax = ax_from_model(float(throttle[k]), float(brake[k]), ax_coeff, mu=float(mu), g=g)
        u_dot = ax + vk * rk; v_dot = (Fy_f + Fy_r) / m - uk * rk; r_dot = (lf * Fy_f - lr * Fy_r) / Iz
        x_dot = uk * np.cos(psik) - vk * np.sin(psik)
        y_dot = uk * np.sin(psik) + vk * np.cos(psik)
        psi_dot = rk
        x[k + 1] = x[k] + dt * x_dot; y[k + 1] = y[k] + dt * y_dot; psi[k + 1] = psi[k] + dt * psi_dot
        u[k + 1] = max(0.0, u[k] + dt * u_dot); v[k + 1] = v[k] + dt * v_dot; r[k + 1] = r[k] + dt * r_dot
        ax_body[k] = ax; ay_body[k] = v_dot + uk * rk

    psi = wrap_to_pi(psi); beta = np.arctan2(v, np.maximum(u, 0.1)); v_total = np.sqrt(u * u + v * v)
    ax_body[-1] = ax_body[-2] if K > 1 else 0.0; ay_body[-1] = ay_body[-2] if K > 1 else 0.0
    return {"time": t, "x": x, "y": y, "psi": psi, "u": u, "v_lat": v, "r": r, "v_total": v_total, "beta": beta, "ax_body": ax_body, "ay_body": ay_body}
