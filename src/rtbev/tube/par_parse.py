from __future__ import annotations

import re
import yaml
from pathlib import Path


def _find(txt: str, key: str):
    m = re.search(r"\b" + re.escape(key) + r"\s+([\-0-9\.]+)", txt)
    return float(m.group(1)) if m else None


def parse_vehicle_from_par(par_path: str) -> dict:
    txt = Path(par_path).read_text(encoding="utf-8", errors="ignore")
    vals = {}
    vals["LX_AXLE_mm"] = _find(txt, "LX_AXLE")
    vals["LX_CG_SU_mm"] = _find(txt, "LX_CG_SU")
    vals["M_SU_kg"] = _find(txt, "M_SU")
    vals["IZZ_SU"] = _find(txt, "IZZ_SU")
    vals["H_CG_SU_mm"] = _find(txt, "H_CG_SU")
    m = re.search(r"Front Track\s*=\s*([0-9\.]+)\s*mm", txt)
    if m:
        vals["FrontTrack_mm"] = float(m.group(1))
    occ = list(re.finditer(r"\n\s*M_US\s+([0-9\.]+)", txt))
    if len(occ) >= 1:
        vals["M_US_front"] = float(occ[0].group(1))
    if len(occ) >= 2:
        vals["M_US_rear"] = float(occ[1].group(1))
    return vals


def derive_params(par_path: str) -> dict:
    info = parse_vehicle_from_par(par_path)
    for k in ["LX_AXLE_mm", "LX_CG_SU_mm", "M_SU_kg", "IZZ_SU", "H_CG_SU_mm"]:
        if info.get(k) is None:
            raise RuntimeError(f"Missing {k} in .par")
    L = info["LX_AXLE_mm"] / 1000.0
    lf = info["LX_CG_SU_mm"] / 1000.0
    lr = L - lf
    t_w = (info.get("FrontTrack_mm", 1540.0)) / 1000.0
    h_cg = info["H_CG_SU_mm"] / 1000.0
    m_su = info["M_SU_kg"]
    m_us_f = info.get("M_US_front", 0.0)
    m_us_r = info.get("M_US_rear", 0.0)
    m_total = m_su + m_us_f + m_us_r
    Iz = info["IZZ_SU"] + 0.5 * (m_us_f + m_us_r) * (t_w / 2.0) ** 2
    return {
        "m_kg": round(m_total, 3),
        "Iz_kgm2": round(Iz, 3),
        "lf_m": round(lf, 4),
        "lr_m": round(lr, 4),
        "track_m": round(t_w, 4),
        "h_cg_m": round(h_cg, 4),
    }


def save_yaml(params: dict, out_yaml: str):
    out = Path(out_yaml)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"vehicle": params}, f, sort_keys=False, allow_unicode=True)
