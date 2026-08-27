# ROF Actionability v1.1.0

本仓库是论文公开代码，用于量化自动驾驶紧急交互中的**可行动作空间坍缩（feasible-action collapse）**。代码覆盖 actionability endpoint、Waymo 五折 OOF 验证、CommonRoad 10k planner-facing 外部验证、严格固定 FPR 指标、强基线、扩展 lattice 动作库敏感性、label-feature decoupling 审计以及论文数据图生成。

## 论文主线

距离和 TTC 回答“交互有多近”，actionability 回答“ego 还剩多少可行响应”。本仓库将 actionability 操作化为短时域 candidate-action feasibility endpoint，并与 distance、TTC、CommonRoad-CriMe-style、RSS-style、drivability 和 deterministic forecast-risk 基线进行比较。

代码支撑的主张边界为：

- actionability 捕捉 proximity 无法完全覆盖的 feasible-response 风险维度；
- 该信号在 Waymo 内部验证和 CommonRoad lattice-planner 外部验证中成立；
- 结论对扩展 lattice 动作库保持稳定；
- decoupling audit 降低但不能完全排除 label-feature coupling；
- 低速 collision-heavy 场景中 distance/TTC 仍可能更有竞争力。

本仓库不声称 closed-loop safety guarantee、native-planner robustness、crash prediction 或所有分层均统一优越。

## 目录结构

```text
configs/          论文分析使用的冻结公开配置
src/rtbev/        ROF、actionability、外部验证和基线核心代码
scripts/          Waymo/CommonRoad 管线及版本化论文分析
figure_tools/     最终主图和补充图绘图脚本
tests/            合成回归测试与 smoke tests
docs/             复现说明、evidence lock 与 claim boundaries
source_data/      独立结果归档的接入说明
```

论文分析模块：

```text
nc_v090–nc_v097   Waymo confirmatory、robustness 和 aligned-feature 分析
nc_v110           CommonRoad 10k、fixed taxonomy、strict-FPR 和 boundary 分析
nc_v111           non-action、temporal 和 mismatch decoupling 审计
nc_v112           field baselines 和 extended-label 评估
```

## 安装

锁定的 Python 主版本为 3.10。

Waymo/核心环境：

```powershell
conda env create -f environment.yml
conda activate rof-actionability
```

CommonRoad 建议使用独立环境：

```powershell
conda env create -f environment-commonroad.yml
conda activate rof-actionability-commonroad
```

Pip：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

CommonRoad 依赖：

```powershell
pip install -e .[dev,commonroad]
```

## 数据路径

本仓库不包含 Waymo 或 CommonRoad 原始数据。请通过官方渠道下载，并遵守相应条款。

PowerShell 中设置：

```powershell
$env:ROF_WORK_DIR="<ROF_WORK_DIR>"
$env:WAYMO_SCENARIO_ROOT="<WAYMO_SCENARIO_ROOT>"
$env:COMMONROAD_SCENARIO_ROOT="<COMMONROAD_SCENARIO_ROOT>"
```

论文派生结果单独归档为：

```text
ROF_results_v1_1_integrated_evidence_lock.zip
```

详见 `source_data/README.md` 和 `docs/REPRODUCE_PAPER.md`。

## 快速检查

```powershell
python -m compileall -q src scripts tests figure_tools
pytest -q
python scripts/99_smoke_test.py
python scripts/99_check_github_readiness.py --root .
```

Smoke test 使用合成数据，不需要下载 Waymo/CommonRoad。

## 从 evidence lock 重画论文图

先解压 `ROF_results_v1_1_integrated_evidence_lock.zip`，然后执行：

```powershell
python figure_tools/make_rof_figures_2_to_6.py `
  --input "<EVIDENCE_ROOT>/99_cleanup_QA/legacy_v100_reference" `
  --out figures/generated/v100 `
  --figures 2 3

python figure_tools/plot_nc_v11_figures_4_5_final_v2_2.py `
  --root "<EVIDENCE_ROOT>" `
  --outdir figures/generated/v11 `
  --skip-supplementary

python figure_tools/make_supplementary_figures_v100.py `
  --data "<EVIDENCE_ROOT>/99_cleanup_QA/legacy_v100_reference" `
  --out figures/generated/supp_v100 `
  --formats pdf png svg

python figure_tools/plot_supplementary_figures.py `
  --evidence-lock "ROF_results_v1_1_integrated_evidence_lock.zip" `
  --output-dir figures/generated/supp_v11 `
  --dpi 600
```

Figure 1 是 information-access 概念图，其 panel source tables 已包含在 evidence lock 中，最终矢量图随论文工程维护。

## 完整复现

完整原始数据复现需要第三方数据和较多计算资源。标准顺序为：

1. Waymo 扫描、样本导出；
2. proximity/actionability label 构建；
3. ROF/current-state/CV feature 生成；
4. `nc_v090–nc_v097` Waymo 分析；
5. `nc_v110` CommonRoad outcome-blind cohort、lattice-base labels；
6. lattice-extended、strict-FPR 和 boundary analyses；
7. `nc_v112` field baselines；
8. `nc_v111` decoupling audits。

详细命令见 `docs/REPRODUCE_PAPER.md`。

## 许可与第三方内容

除特别说明外，本仓库使用 Apache License 2.0。`src/waymo_open_dataset/` 下的 Waymo protobuf 生成模块在 `THIRD_PARTY_NOTICES.md` 中说明。Waymo 与 CommonRoad 原始数据不随仓库分发。


