# Quantum IFM — Overlap, Dose Reduction & Sensitivity

This repo contains Python scripts for analyzing and visualizing interaction-free measurement (IFM) behavior: overlap regions between dose-reduction conditions, category maps, and sensitivity comparisons.

## Contents

- `overlap_region_optimal.py` — standalone script that computes the overlap of three conditions (two dose sources + threshold) and plots categories; includes a zero-contour for `η_dose - η1 = 0`, square aspect, and dynamic legend.
- `overlap_regions_and_dose_reduction.py` — Tkinter GUI to explore overlap categories and heatmaps on the (t₀, φ) plane; A4 font controls, batch save, accurate minimal t₀* tick, and square saved plots. Run with `python overlap_regions_and_dose_reduction.py --grid 180 --nT 201`.
- `sensitivity_analysis.py` — Tkinter GUI for sensitivity maps: s(μ,ν), coherent references, ratios, source maps, and |11⟩ with fixed/optimal T. Includes stable colorbar axes and fixed plot geometry for publication-quality exports.

## Quickstart

```bash
pip install -r requirements.txt
python overlap_region_optimal.py
python overlap_regions_and_dose_reduction.py --grid 180 --nT 201
python sensitivity_analysis.py
```

## Requirements
- Python ≥ 3.9
- NumPy, Matplotlib, Tkinter (preinstalled with most Python installations)


