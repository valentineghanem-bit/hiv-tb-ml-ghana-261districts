"""Regenerate the poster through the Bespoke HI-EI visual generator."""
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts" / "build_hiei_visuals.py"), run_name="__main__")
