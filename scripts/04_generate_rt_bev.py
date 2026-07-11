from __future__ import annotations
from pathlib import Path
import runpy
from _bootstrap import ROOT
runpy.run_path(str(Path(__file__).with_name("04_generate_rof_features.py")), run_name="__main__")
