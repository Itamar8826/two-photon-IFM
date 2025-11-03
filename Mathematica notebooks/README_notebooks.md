# Mathematica Notebooks — Two-Photon IFM

This folder contains Wolfram Mathematica notebooks used to derive analytical expressions and verify symbolic results for the **two-photon interaction-free measurement (IFM)** project.

---

## 📘 Contents

| Notebook | Purpose |
|-----------|----------|
| `IFM_Calculations.nb` | Full symbolic derivations of Mach–Zehnder interferometer output states, IFM efficiency \(\eta\), and dose-reduction metric \(\eta^{\text{dose}}\). |
| `General_phase_sensitivity.nb` | Analytical and numerical evaluation of phase sensitivity and comparison to coherent references. Includes plots of \(S_{\varphi}^2\) vs. object parameters. |

---

## ⚙️ Requirements

- Wolfram Mathematica ≥ 13.0  
- No external packages required

---

## ▶️ Usage

1. Open the notebook in Mathematica:
   ```wolfram
   NotebookOpen["notebooks/IFM_Calculations.nb"]
   ```
2. Evaluate all sections sequentially (`Shift + Enter`) or use **Evaluation → Evaluate Notebook**.
3. For numerical visualization or export, connect to the Python scripts in `../IFM_Project/` if desired.

---

## 📄 Description

The notebooks provide the **analytical foundation** for the Python visualization tools.  
They derive exact output probabilities for one- and two-photon inputs, calculate detection and absorption probabilities, and define both:
- IFM efficiency \(\eta = \frac{P(\text{detect})}{P(\text{detect})+P(\text{absorb})}\)  
- Dose-reduction metric \(\eta^{\text{dose}}\)

These quantities are compared to their coherent-state counterparts and exported for use in numerical simulations.

---

## 🧩 Citation
If referenced in a paper, please cite:
> Itamar Horovitz, *Two-Photon Interaction-Free Measurement – Analytical Derivations in Mathematica*, 2025.

---

## 📄 License
MIT License © 2025 Itamar Horovitz
