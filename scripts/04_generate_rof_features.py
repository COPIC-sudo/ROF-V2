from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from _bootstrap import ROOT
from rtbev.config import load_config
from rtbev.io_utils import ensure_dir, read_gzip_pickle
from rtbev.tube.rt_library import TubeLibrary, PrimitiveLibrary
from rtbev.pipeline import sample_to_bev_tensor_and_features


def _is_cuda_device(device: str | None) -> bool:
    return str(device or "").lower().startswith("cuda")


def _print_runtime_banner(device: str, cfg: dict) -> None:
    print(f"[rof] requested device: {device}")
    if _is_cuda_device(device):
        if torch is None:
            raise RuntimeError("--device cuda was requested, but PyTorch is not installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False. Install CUDA-enabled PyTorch.")
        dev = torch.device(device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"[rof] CUDA enabled: {props.name}, capability={props.major}.{props.minor}, total_mem={props.total_memory / (1024**3):.1f} GiB")
        print(f"[rof] gpu_dtype={cfg.get('runtime', {}).get('gpu_dtype', 'float32')}, msr_gpu_chunk_size={cfg.get('runtime', {}).get('msr_gpu_chunk_size', 256)}")
        print("[rof] GPU acceleration path active: RT polygon rasterization, CV rasterization, and MSR primitive survival use torch/CUDA.")
    else:
        print("[rof] CPU path active. For GPU acceleration use --device cuda:0 with a CUDA-enabled PyTorch install.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--no-tensors", action="store_true", help="只输出 features.csv，不保存 npz 张量")
    ap.add_argument("--out-name", default="rof_features.csv", help="features 输出文件名；rolling/INTERACTION 可用 rof_features_rolling.csv 或 interaction_rof_features.csv")
    ap.add_argument("--gpu-dtype", choices=["float32", "float64"], default=None, help="CUDA rasterization dtype；默认 float64 以尽量贴近原 CPU 双精度边界行为；如需更快可显式设为 float32")
    ap.add_argument("--msr-gpu-chunk-size", type=int, default=None, help="MSR CUDA 分块大小；显存不足时降到 16，显存充足时可升到 64/128")
    ap.add_argument("--gpu-polygon-edge-chunk", type=int, default=None, help="CUDA polygon point-in-ring 边分块大小，默认 256")
    ap.add_argument("--save-uncompressed-tensors", action="store_true", help="用 np.savez 替代 np.savez_compressed；数值不变但写盘更快、文件更大")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("runtime", {})
    device = args.device or cfg["runtime"].get("device", "cpu")
    cfg["runtime"]["device"] = device
    if args.gpu_dtype is not None:
        cfg["runtime"]["gpu_dtype"] = args.gpu_dtype
    else:
        cfg["runtime"].setdefault("gpu_dtype", "float64")
    if args.msr_gpu_chunk_size is not None:
        cfg["runtime"]["msr_gpu_chunk_size"] = int(args.msr_gpu_chunk_size)
    else:
        cfg["runtime"].setdefault("msr_gpu_chunk_size", 256)
    # Alias used by some internal box kernels.  MSR uses msr_gpu_chunk_size first.
    cfg["runtime"].setdefault("gpu_box_batch_size", int(cfg["runtime"]["msr_gpu_chunk_size"]))
    if args.gpu_polygon_edge_chunk is not None:
        cfg["runtime"]["gpu_polygon_edge_chunk"] = int(args.gpu_polygon_edge_chunk)
    else:
        cfg["runtime"].setdefault("gpu_polygon_edge_chunk", 256)

    _print_runtime_banner(device, cfg)

    work_dir = Path(cfg["project"]["work_dir"])
    tensor_dir = ensure_dir(work_dir / "bev_tensors")
    feat_dir = ensure_dir(work_dir / "features")
    labels = pd.read_csv(args.labels_csv)
    if args.max_samples is not None:
        labels = labels.iloc[: args.max_samples].copy()

    lib = TubeLibrary.from_workdir(work_dir / "tube_library")
    prim_lib = PrimitiveLibrary.from_workdir(work_dir / "tube_library")
    if not prim_lib.available:
        print("[warn] primitive_mu*_v*.npz not found; MSR/REDI_full will be NaN. Run 02b_prepare_primitive_library.py or fallback mode.")

    feat_rows = []
    t0 = time.perf_counter()
    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="rof+features"):
        sample_id = row["sample_id"]
        sample = read_gzip_pickle(work_dir / "samples" / f"{sample_id}.pkl.gz")
        tensors, feats = sample_to_bev_tensor_and_features(sample, lib, cfg, device=device, primitive_lib=prim_lib, return_tensors=(not args.no_tensors))
        if not args.no_tensors:
            out_npz = tensor_dir / f"{sample_id}.npz"
            if args.save_uncompressed_tensors:
                np.savez(out_npz, **tensors)
            else:
                np.savez_compressed(out_npz, **tensors)
        meta = {
            "sample_id": sample_id,
            "scenario_id": row.get("scenario_id", sample_id),
            **feats,
            "label_id": int(row["label_id"]),
            "label_name": row["label_name"],
            "risk_score": float(row.get("risk_score", np.nan)),
            "dmin_future_m": float(row.get("dmin_future_m", np.nan)),
            "t_at_dmin_s": float(row.get("t_at_dmin_s", np.nan)),
            "collision_any": bool(row.get("collision_any", False)),
        }
        for extra_col in ["root_scenario_id", "base_current_time_index", "current_time_index", "current_time_s", "current_time_s_global", "relative_to_base_time_s", "source_dataset", "source_file", "case_id", "frame_id"]:
            if extra_col in row:
                meta[extra_col] = row.get(extra_col)
        feat_rows.append(meta)

    if _is_cuda_device(device) and torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()

    feat_df = pd.DataFrame(feat_rows)
    out_csv = feat_dir / args.out_name
    feat_df.to_csv(out_csv, index=False)
    # backward-compatible filenames for the legacy pipeline only.
    if args.out_name == "rof_features.csv":
        feat_df.to_csv(feat_dir / "rt_features.csv", index=False)
    elapsed = time.perf_counter() - t0
    n = max(len(labels), 1)
    if not args.no_tensors:
        print(f"[rof] wrote tensors under {tensor_dir}")
    else:
        print("[rof] tensor saving disabled (--no-tensors)")
    print(f"[rof] wrote features to {out_csv}")
    print(f"[rof] elapsed={elapsed:.1f}s, avg={elapsed / n:.3f}s/sample")


if __name__ == "__main__":
    main()
