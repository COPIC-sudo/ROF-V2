# Figure generation

The public release retains only the final or canonical paper plotters.

| Script | Purpose | Input |
|---|---|---|
| `make_rof_figures_2_to_6.py` | Integrated-evidence compatible Waymo/main figures; use Figures 2–3 for the v1.1 manuscript | Extracted `legacy_v100_reference` directory |
| `plot_v100_redesigned_figures_refined.py` | Exact refined legacy artwork when the optional redesigned plot-ready bundle is available | `03_main_figure_source_data_v100_redesigned` bundle |
| `make_supplementary_figures_v100.py` | Supplementary Figures S1–S5; legacy CommonRoad pilot is excluded by default | Extracted `legacy_v100_reference` or v100 evidence lock |
| `plot_nc_v11_figures_4_5_final_v2_2.py` | Main Figures 4–5 | Extracted v1.1 integrated evidence lock |
| `plot_supplementary_figures.py` | Supplementary Figures S6–S9 | v1.1 evidence-lock ZIP or extracted directory |

All plotting scripts write PDF/SVG vector output and high-resolution PNG output. They do not modify the evidence lock.
