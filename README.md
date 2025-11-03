# Two-Photon Interaction-Free Measurement (IFM)

This repository contains all simulation, visualization, and analytical resources used for a study of **two-photon interaction-free measurement (IFM)** schemes in a Mach–Zehnder interferometer.  
It combines **Mathematica notebooks** for theoretical derivations with **Python scripts** for numerical analysis and visualization.

---

## 📁 Repository Structure

```
two-photon-IFM/
├─ Mathematica notebooks/
│  ├─ IFM_Calculations.nb
│  └─ Coherent_phase_sensitivity.nb
│
├─ Python scripts/
│  ├─ overlap_region_optimal.py
│  ├─ overlap_regions_and_dose_reduction.py
│  ├─ sensitivity_analysis.py
│  ├─ README.md
│  ├─ requirements.txt
│  └─ .gitignore
```

### ▸ `Mathematica notebooks/`
Contains **Wolfram Mathematica** notebooks for:
- deriving single- and two-photon probability amplitudes,
- computing IFM efficiency and dose-reduction metrics,
- analyzing phase sensitivity and quantum advantage regions.

### ▸ `Python scripts/`
Contains the **Python implementation and visualization GUI** for reproducing and exploring IFM behavior numerically.

---

## ⚙️ Installation

### Requirements
- Python ≥ 3.9  
- NumPy, Matplotlib, Tkinter  
- Mathematica ≥ 13.0 (for `.nb` files)

### Setup
```bash
# clone the repository
git clone https://github.com/Itamar8826/two-photon-IFM.git
cd two-photon-IFM/IFM_Project

# install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the standalone overlap analyzer
```bash
python overlap_region_optimal.py
```

### Launch the GUI for dose-reduction & overlap maps
```bash
python overlap_regions_and_dose_reduction.py --grid 180 --nT 201
```

### Launch the GUI for phase sensitivity visualization
```bash
python sensitivity_analysis.py
```

### Open Mathematica notebooks
In Mathematica:
```wolfram
NotebookOpen["notebooks/IFM_Calculations.nb"]
```

---

## 📊 Project Description

The project investigates **how multi-photon quantum inputs can enhance interaction-free measurements** by reducing absorption (dose) while improving phase sensitivity.  
The Python tools reproduce contour maps, identify joint-advantage regions where
\[
\eta^{\text{dose}} > 0 \quad \text{and} \quad S_\varphi^2 < S_{\varphi,\mathrm{coh}}^2,
\]
and visualize optimal interferometer settings \((t_0,\varphi)\).

---

## 🧩 Citation
If you reference or reuse this code in a report or paper, please cite as:
> Itamar Horovitz, *Two-Photon Interaction-Free Measurement – Numerical and Analytical Exploration*, 2025.

---

## 📄 License
MIT License © 2025 Itamar Horovitz

---

## 🔗 Links
- **Project page:** [https://github.com/Itamar8826/two-photon-IFM](https://github.com/Itamar8826/two-photon-IFM)
- **Related report:** *Summary and Conclusions*, Section 4.7 (attached in final PDF)
