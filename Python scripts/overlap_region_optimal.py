#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script to find and plot the overlap of three conditions:
1. Dose Reduction Source (from code 1)
2. Min Sensitivity Source (from code 2)
3. Dose Reduction > Eta1 (from code 1)

**Updates**:
- Added contour line for eta_dose - eta1 = 0
- Ensures a square data plot aspect ratio
- 4-category asymmetric match logic
- 10-category plotting
- Zoomed in to t0 > 0.5 and phi < pi/2
- **NEW: Legend is now dynamic, only shows labels for colors present in the zoomed view.**
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# --- Grid Parameters ---
GRID_SIZE = 200  # Resolution for the (t0, phi) plane
NT_POINTS = 201  # Resolution for T-optimization (for f4 and T_opt)


# =============================================================================
# HELPERS (COMMON)
# =============================================================================

def _safe_div(num: np.ndarray, den: np.ndarray, fill=np.nan) -> np.ndarray:
    """Vectorized safe division (from Code 1)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        out = num / den
    out[~np.isfinite(out)] = fill
    return out


def _sin2(phi):
    """Vectorized sin^2(phi) (from Code 2)"""
    s2 = np.sin(phi) ** 2
    return np.maximum(s2, 1e-12)


def _sqrt_pos(x):
    """Vectorized sqrt(max(x, 0)) (from Code 2)"""
    return np.sqrt(np.maximum(x, 0.0))


def _safe_over(num, den, tiny=1e-300):
    """Vectorized safe division (from Code 2)"""
    den_safe = np.where(np.abs(den) < tiny, np.sign(den) * tiny + (den == 0) * tiny, den)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        out = num / den_safe
    return out


def _finite_pos(x):
    """Returns x where x is finite and > 0, else nan (from Code 2)"""
    x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
    return x


# =============================================================================
# MATH FUNCTIONS (FROM CODE 1: DOSE REDUCTION)
# =============================================================================

def eta1max(t0_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    """
    eta1max = (1 + t^2 - 2 t cos(phi)) / (2 - 2 t cos(phi))
    (Needed for Condition 3)
    """
    c = np.cos(phi_grid)
    num = 1.0 + t0_grid ** 2 - 2.0 * t0_grid * c
    den = 2.0 - 2.0 * t0_grid * c
    return _safe_div(num, den)


def dosea_symmetric_20(t0_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    """Dose Function (AS, 20/02)"""
    c = np.cos(phi_grid)
    num = 4.0 * t0_grid ** 2 - 4.0 * t0_grid * c
    den = 4.0 - 4.0 * t0_grid * c
    return _safe_div(num, den)


def dose_symmetric_20(t0_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    """Dose Function (S, 20/02)"""
    c = np.cos(phi_grid)
    c2 = np.cos(2.0 * phi_grid)
    num = 13.0 - 20.0 * t0_grid ** 2 - 3.0 * t0_grid ** 4 - 4.0 * (t0_grid + t0_grid ** 3) * c + 2.0 * t0_grid ** 2 * c2
    den = -15.0 + 4.0 * t0_grid ** 2 + t0_grid ** 4 - 4.0 * (t0_grid + t0_grid ** 3) * c + 2.0 * t0_grid ** 2 * c2
    return _safe_div(num, den)


def dose_symmetric_11(t0_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    """Dose Function (S, 11)"""
    c2 = np.cos(2.0 * phi_grid)
    num = -3.0 + 4.0 * t0_grid ** 2 + t0_grid ** 4 - 2.0 * t0_grid ** 2 * c2
    den = -3.0 + t0_grid ** 4 + 2.0 * t0_grid ** 2 * c2
    return -_safe_div(num, den)


def f11_max_over_T(T_points: np.ndarray, t0_grid: np.ndarray, phi_grid: np.ndarray):
    """Dose Function (AS, 11)"""
    # Broadcast to (nT, Nphi, Nt)
    T = T_points[:, None, None]
    t = t0_grid[None, :, :]
    phi = phi_grid[None, :, :]
    c = np.cos(phi)

    term1 = (
            -1.0
            + T * (2.0 + T * (-6.0 - 4.0 * (-2.0 + T) * T))
            + (1.0 + T * (4.0 - 4.0 * T * (3.0 - 4.0 * T + 2.0 * T ** 2))) * t ** 2
            + T * (2.0 + T * (-6.0 - 4.0 * (-2.0 + T) * T)) * t ** 4
            + 4.0 * (-1.0 + T) * T * t * c * (((1.0 - 2.0 * T) ** 2) * (1.0 + t ** 2)
                                              - 4.0 * (-1.0 + T) * T * t * c)
    )

    term2 = (
            1.0 - t ** 2
            + T * (
                    8.0 * t ** 2
                    + 8.0 * T ** 2 * (1.0 + t ** 2) ** 2
                    - 4.0 * T ** 3 * (1.0 + t ** 2) ** 2
                    - 4.0 * T * (1.0 + 4.0 * t ** 2 + t ** 4)
            )
            + 4.0 * (-1.0 + T) * T * t * c * (((1.0 - 2.0 * T) ** 2) * (1.0 + t ** 2)
                                              - 4.0 * (-1.0 + T) * T * t * c)
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        values = term1 / term2
    values[~np.isfinite(values)] = -np.inf  # invalid/inf never win the max

    # Max over T axis (axis=0)
    idx = np.nanargmax(values, axis=0)  # (Nphi, Nt)
    fmax = np.take_along_axis(values, idx[None, :, :], axis=0)[0]
    T_star = T_points[idx]
    return fmax, T_star


# =============================================================================
# MATH FUNCTIONS (FROM CODE 2: SENSITIVITY)
# =============================================================================

def SensCoherentN12(T, mu, nu, tO, phi):
    """Sensitivity function 1 (from s_min_of_three)"""
    sin2 = _sin2(phi)
    sqrt_term = _sqrt_pos((1.0 - T) * T * mu * nu)
    num1 = (-2.0 * (-1.0 + tO ** 2) * sqrt_term
            + T * (tO ** 2 * mu + nu)
            - (-1.0 + T) * (mu + tO ** 2 * nu))
    num = _safe_over(num1, sin2)
    den_term = ((-1.0 + T) * T * mu - (-1.0 + T) * T * nu + (1.0 - 2.0 * T) * sqrt_term)
    den = 16.0 * tO ** 2 * den_term ** 2
    val = _safe_over(num, den)
    return _finite_pos(val)


def SensCoNb1(T, mu, nu, tO, phi):
    """Sensitivity function 2 (from s_min_of_three)"""
    sin2 = _sin2(phi)
    sTT = (1.0 - T) * T
    sqrt_p = _sqrt_pos(sTT * mu * nu)
    num1 = ((1.0 - T) ** 2 * mu + T ** 2 * tO ** 2 * mu
            + (1.0 - T) * T * (nu + tO ** 2 * nu)
            + 2.0 * (1.0 - T) * sqrt_p - 2.0 * T * tO ** 2 * sqrt_p
            - 2.0 * tO * (-(sTT) * mu + (sTT) * nu + (1.0 - 2.0 * T) * sqrt_p) * np.cos(phi))
    num = _safe_over(num1, sin2)
    den = 4.0 * tO ** 2 * ((sTT) * mu - (sTT) * nu + (-1.0 + 2.0 * T) * sqrt_p) ** 2
    val = _safe_over(num, den)
    return _finite_pos(val)


def SensCoNb2(T, mu, nu, tO, phi):
    """Sensitivity function 3 (from s_min_of_three)"""
    sin2 = _sin2(phi)
    sqrt_term = _sqrt_pos((1.0 - T) * T * mu * nu)
    num1 = (-T ** 2 * (1.0 + tO ** 2) * (mu - nu)
            + tO ** 2 * (nu - 2.0 * sqrt_term)
            + T * ((1.0 + tO ** 2) * mu + 2.0 * (sqrt_term + tO ** 2 * (-nu + sqrt_term)))
            - 2.0 * tO * (-sqrt_term + T ** 2 * (-mu + nu) + T * (mu - nu + 2.0 * sqrt_term)) * np.cos(phi))
    num = _safe_over(num1, sin2)
    den_term = ((-1.0 + T) * T * mu - (-1.0 + T) * T * nu + sqrt_term - 2.0 * T * sqrt_term)
    den = 4.0 * tO ** 2 * den_term ** 2
    val = _safe_over(num, den)
    return _finite_pos(val)


def dose11(T, tO, phi):
    """Dose function for |11> (from Code 2, needed for T_opt)"""
    T2, T3, T4 = T ** 2, T ** 3, T ** 4
    t2, t4 = tO ** 2, tO ** 4
    num = (
            (-1 + T * (2 + T * (-6 - 4 * (-2 + T) * T)))
            + (1 + T * (4 - 4 * T * (3 - 4 * T + 2 * T2))) * t2
            + T * (2 + T * (-6 - 4 * (-2 + T) * T)) * t4
            + 4 * (-1.0 + T) * T * tO * np.cos(phi)
            * ((1 - 2 * T) ** 2 * (1 + t2) - 4 * (-1.0 + T) * T * tO * np.cos(phi))
    )
    den = (
            (1 - t2)
            + T * (2.22045e-16 + 8 * t2 + 8 * T2 * (1 + t2) ** 2 - 4 * T3 * (1 + t2) ** 2 - 4 * T * (
            1 + 4 * t2 + t4))
            + 4 * (-1.0 + T) * T * tO * np.cos(phi)
            * ((1 - 2 * T) ** 2 * (1 + t2) - 4 * (-1.0 + T) * T * tO * np.cos(phi))
    )
    return _safe_over(num, den)


def s11(T, tO, phi):
    """Sensitivity function for |11> (from Code 2)"""
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    csc_phi = 1.0 / (sin_phi + 1e-12)
    term1 = (-4 * T ** 2 + 8 * T ** 3 + 12 * T ** 4 - 64 * T ** 5 + 96 * T ** 6 - 64 * T ** 7 + 16 * T ** 8)
    term2 = (-tO ** 2 + 8 * T * tO ** 2 - 8 * T ** 2 * tO ** 2 - 128 * T ** 3 * tO ** 2 + 640 * T ** 4 * tO ** 2 -
             1408 * T ** 5 * tO ** 2 + 1664 * T ** 6 * tO ** 2 - 1024 * T ** 7 * tO ** 2 + 256 * T ** 8 * tO ** 2)
    term3 = (
            tO ** 4 - 16 * T * tO ** 4 + 124 * T ** 2 * tO ** 4 - 600 * T ** 3 * tO ** 4 + 1836 * T ** 4 * tO ** 4 -
            3456 * T ** 5 * tO ** 4 + 3840 * T ** 6 * tO ** 4 - 2304 * T ** 7 * tO ** 4 + 576 * T ** 8 * tO ** 4)
    term4 = (16 * T ** 2 * tO ** 6 - 160 * T ** 3 * tO ** 6 + 656 * T ** 4 * tO ** 6 - 1408 * T ** 5 * tO ** 6 +
             1664 * T ** 6 * tO ** 6 - 1024 * T ** 7 * tO ** 6 + 256 * T ** 8 * tO ** 6)
    term5 = (
            16 * T ** 4 * tO ** 8 - 64 * T ** 5 * tO ** 8 + 96 * T ** 6 * tO ** 8 - 64 * T ** 7 * tO ** 8 + 16 * T ** 8 * tO ** 8)
    cos_term = (4 * (1 - 2 * T) ** 2 * (-1.0 + T) * T * tO * (1 + tO ** 2) *
                (-1 + 2 * tO ** 2 - 16 * T * tO ** 2 - 16 * T ** 3 * (1 + 5 * tO ** 2 + tO ** 4) +
                 8 * T ** 4 * (1 + 5 * tO ** 2 + tO ** 4) + 8 * T ** 2 * (1 + 7 * tO ** 2 + tO ** 4)) * cos_phi)
    cos2_term = (8 * (-1.0 + T) ** 2 * T ** 2 * tO ** 2 *
                 (tO ** 2 * (4 + tO ** 2) + 32 * T ** 2 * (1 + 3 * tO ** 2 + tO ** 4) -
                  8 * T * (1 + 4 * tO ** 2 + tO ** 4) - 16 * T ** 3 * (3 + 8 * tO ** 2 + 3 * tO ** 4) +
                  8 * T ** 4 * (3 + 8 * tO ** 2 + 3 * tO ** 4)) * np.cos(2 * phi))
    cos3_term = ((32 * T ** 3 * tO ** 3 - 224 * T ** 4 * tO ** 3 + 608 * T ** 5 * tO ** 3 - 800 * T ** 6 * tO ** 3 +
                  512 * T ** 7 * tO ** 3 - 128 * T ** 8 * tO ** 3 + 32 * T ** 3 * tO ** 5 - 224 * T ** 4 * tO ** 5 +
                  608 * T ** 5 * tO ** 5 - 800 * T ** 6 * tO ** 5 + 512 * T ** 7 * tO ** 5 - 128 * T ** 8 * tO ** 5) *
                 np.cos(3 * phi))
    cos4_term = ((32 * T ** 4 * tO ** 4 - 128 * T ** 5 * tO ** 4 + 192 * T ** 6 * tO ** 4 - 128 * T ** 7 * tO ** 4 +
                  32 * T ** 8 * tO ** 4) * np.cos(4 * phi))
    numerator = -(
            term1 + term2 + term3 + term4 + term5 - cos_term + cos2_term + cos3_term + cos4_term) * csc_phi ** 2
    denominator = (16 * (-1.0 + T) ** 2 * T ** 2 * tO ** 2 * (
            (1 - 2 * T) ** 2 * (1 + tO ** 2) - 8 * (-1.0 + T) * T * tO * cos_phi) ** 2)
    return np.abs(numerator / (np.abs(denominator) + 1e-12))


# =============================================================================
# NEW COMPUTATION FUNCTIONS (ADAPTED FROM CODE 2)
# =============================================================================

def compute_Topt_map_11(t0_grid, phi_grid, nT=101):
    """
    Finds the optimal T for each (t0, phi) point that maximizes dose11.
    (Adapted from Code 2's _compute_Topt_map_11)
    """
    Ts = np.linspace(1e-6, 0.5 - 1e-6, nT).reshape((-1, 1, 1))
    dose_vals = dose11(Ts, t0_grid[None, ...], phi_grid[None, ...])  # (nT, Ny, Nx)
    k = np.argmax(dose_vals, axis=0)
    return Ts.reshape(-1)[k]


def _s_min_of_three(T, mu, nu, t0_grid, phi_grid):
    """
    Computes the min sensitivity for (mu, nu) input at fixed T.
    (Adapted from Code 2's _s_min_of_three)
    """
    f1 = SensCoherentN12(T, mu, nu, t0_grid, phi_grid)
    f2 = SensCoNb1(T, mu, nu, t0_grid, phi_grid)
    f3 = SensCoNb2(T, mu, nu, t0_grid, phi_grid)
    stack = np.stack([f1, f2, f3], axis=-1)
    return np.nanmin(stack, axis=-1)


# =============================================================================
# MAIN SCRIPT LOGIC
# =============================================================================

def main():
    print(f"Creating grid (t0, phi) with size {GRID_SIZE}x{GRID_SIZE}...")
    # Full data range
    t0_vals = np.linspace(0.01, 0.99, GRID_SIZE)
    phi_vals = np.linspace(0.01, np.pi - 0.01, GRID_SIZE)
    t0_grid, phi_grid = np.meshgrid(t0_vals, phi_vals)
    T_points = np.linspace(0.001, 0.999, NT_POINTS)

    # --- 1. Compute Condition 1 (Dose Source Map) ---
    print("Computing Condition 1 (Dose Source Map from Code 1)...")
    f1_dose = dosea_symmetric_20(t0_grid, phi_grid)  # Idx 0
    f2_dose = dose_symmetric_20(t0_grid, phi_grid)  # Idx 1
    f3_dose = dose_symmetric_11(t0_grid, phi_grid)  # Idx 2
    f4_dose, _ = f11_max_over_T(T_points, t0_grid, phi_grid)  # Idx 3

    stack_dose = np.stack([f1_dose, f2_dose, f3_dose, f4_dose], axis=0)
    max_dose_vals = np.nanmax(stack_dose, axis=0)
    source_dose = np.nanargmax(stack_dose, axis=0)  # Indices 0-3
    source_dose[max_dose_vals <= 0.0] = -1  # -1 for "none positive"

    # --- 4-CATEGORY GROUPING for Dose Map ---
    # Cat 1: (AS, 20/02)
    # Cat 2: (S, 20/02)
    # Cat 3: (S, 11)
    # Cat 4: (AS, 11)
    dose_origin_map = np.full_like(source_dose, 0)  # 0 = None
    dose_origin_map[source_dose == 0] = 1
    dose_origin_map[source_dose == 1] = 2
    dose_origin_map[source_dose == 2] = 3
    dose_origin_map[source_dose == 3] = 4

    # --- 2. Compute Condition 3 (Dose > Eta1) ---
    print("Computing Condition 3 (Dose > Eta1 from Code 1)...")
    eta1_max = eta1max(t0_grid, phi_grid)
    # This mask is True where (max_dose_vals - eta1_max) > 0.0
    cond3_mask = (max_dose_vals - eta1_max) > 0.0
    # We also need the difference data for the contour plot
    dose_minus_eta1_diff = max_dose_vals - eta1_max

    # --- 3. Compute Condition 2 (Sensitivity Source Map) ---
    print("Computing Condition 2 (Global Min Sensitivity Source from Code 2)...")
    T_map_opt11 = compute_Topt_map_11(t0_grid, phi_grid, nT=NT_POINTS)

    # These are the 4 functions for the "Global s-min" plot in Code 2
    s1_sens = _s_min_of_three(1.0, 0, 2, t0_grid, phi_grid)  # Idx 0: (AS, 20/02)
    s2_sens = _s_min_of_three(0.5, 2, 0, t0_grid, phi_grid)  # Idx 1: (S, 20/02)
    s3_sens = s11(0.5, t0_grid, phi_grid)  # Idx 2: (S, 11)
    s4_sens = s11(T_map_opt11, t0_grid, phi_grid)  # Idx 3: (AS, 11)

    stack_sens = np.stack([s1_sens, s2_sens, s3_sens, s4_sens], axis=0)
    stack_sens_safe = np.where(np.isfinite(stack_sens), stack_sens, np.inf)
    min_sens_vals = np.min(stack_sens_safe, axis=0)
    source_sens = np.argmin(stack_sens_safe, axis=0)  # Indices 0-3

    all_inf = ~np.isfinite(min_sens_vals) | (min_sens_vals == np.inf)
    source_sens[all_inf] = -1  # -1 for "none finite"

    # --- 4-CATEGORY GROUPING for Sens Map (with remapping rule) ---
    # "treat 11 optimal T [idx 3] like 11 T=0.5 [idx 2]"
    # This means (AS, 11) source is remapped to (S, 11) category.
    sens_origin_map = np.full_like(source_sens, 0)  # 0 = None
    sens_origin_map[source_sens == 0] = 1  # (AS, 20/02)
    sens_origin_map[source_sens == 1] = 2  # (S, 20/02)
    sens_origin_map[source_sens == 2] = 3  # (S, 11)
    sens_origin_map[source_sens == 3] = 3  # (AS, 11) is REMAPPED to 3

    # --- 4. Find Overlaps ---
    print("Finding overlap regions...")

    # Define the 4 types of origin match, based on the asymmetric rule
    is_AS_20_02_overlap = (dose_origin_map == 1) & (sens_origin_map == 1)
    is_S_20_02_overlap = (dose_origin_map == 2) & (sens_origin_map == 2)
    is_S_11_overlap = (dose_origin_map == 3) & (sens_origin_map == 3)
    is_AS_11_special_overlap = (dose_origin_map == 4) & (sens_origin_map == 3)

    # --- 5. Build Final Plot Map ---
    # We create a map with 10 categories
    plot_map = np.zeros_like(t0_grid, dtype=int)

    # Cat 1: Dose > Eta1 ONLY
    plot_map[cond3_mask] = 1

    # Cats 2-5: The "Match (only)" regions (old gray area)
    # These are where a match happens AND cond3 is FALSE
    plot_map[is_AS_20_02_overlap & ~cond3_mask] = 2
    plot_map[is_S_20_02_overlap & ~cond3_mask] = 3
    plot_map[is_S_11_overlap & ~cond3_mask] = 4
    plot_map[is_AS_11_special_overlap & ~cond3_mask] = 5

    # Cats 6-9: The "Triple Overlap" regions
    # These are where a match happens AND cond3 is TRUE
    # This will overwrite the '1's from the cond3_mask assignment
    plot_map[is_AS_20_02_overlap & cond3_mask] = 6
    plot_map[is_S_20_02_overlap & cond3_mask] = 7
    plot_map[is_S_11_overlap & cond3_mask] = 8
    plot_map[is_AS_11_special_overlap & cond3_mask] = 9

    # --- NEW: Find which categories are present IN THE ZOOMED VIEW ---
    t0_zoom_mask = t0_vals > 0.5
    phi_zoom_mask = phi_vals < (np.pi / 2)

    # Need to apply 1D masks to 2D plot_map[phi_idx, t0_idx]
    # This creates a 2D slice of the map
    zoomed_plot_map_slice = plot_map[phi_zoom_mask, :][:, t0_zoom_mask]

    # Find the unique category indices present in this slice
    present_categories = np.unique(zoomed_plot_map_slice)
    print(f"Categories present in zoomed view: {present_categories}")

    # --- 6. Plot the results ---
    print("Plotting results...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Define all 10 colors and labels
    all_colors = [
        # Cat 0:
        '#DDDDDD',  # None
        # Cat 1:
        '#0072B2',  # Dose > Eta1 only (Blue)
        # Cats 2-5 (The old gray area, now split):
        '#B0C4DE',  # 2: Match (AS, 20/02) (only) (Light Steel Blue)
        '#DDA0DD',  # 3: Match (S, 20/02) (only) (Plum)
        '#98FB98',  # 4: Match (S, 11) (only) (Pale Green)
        '#F08080',  # 5: Match (AS, 11)D/(S, 11)S (only) (Light Coral)
        # Cats 6-9 (The triple overlaps):
        '#F0E442',  # 6: OVERLAP (AS, 20/02) + (Dose > Eta1) (Yellow)
        '#E69F00',  # 7: OVERLAP (S, 20/02) + (Dose > Eta1) (Orange)
        '#009E73',  # 8: OVERLAP (S, 11) + (Dose > Eta1) (Green)
        '#D55E00',  # 9: OVERLAP (AS, 11)D/(S, 11)S + (Dose > Eta1) (Vermillion)
    ]
    all_labels = [
        "No explicit optimal, $\eta^{dose}_{2 photon,\max} < \eta_{1 photon,\max}$",
        r" $\eta^{dose}_{2 photon,\max} > \eta_{1 photon,\max}$",

        "Match: (AS, 20/02) (only)",
        "|02⟩,|20⟩ T =0.5 &  $\eta^{dose}_{2 photon,\max} < \eta_{1 photon,\max}$",
        "Match: (S, 11) (only)",
        r"Match: (AS, 11)$_D$ / (S, 11)$_S$ (only)",

        r"**Overlap: (AS, 20/02) + $\eta_{dose} > \eta_1$**",
        r"|02⟩,|20⟩ T =0.5 & $\eta^{dose}_{2 photon,\max} > \eta_{1 photon,\max}$",
        r"**Overlap: (S, 11) + $\eta_{dose} > \eta_1$**",
        r"**Overlap: (AS, 11)$_D$ / (S, 11)$_S$ + $\eta_{dose} > \eta_1$**"
    ]

    # We plot with all colors, so the mapping is correct
    cmap = ListedColormap(all_colors)
    boundaries = np.arange(-0.5, 10.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    im = ax.pcolormesh(
        t0_grid, phi_grid, plot_map,
        cmap=cmap, norm=norm, shading='auto'
    )

    # --- Add the contour line for eta_dose - eta1 = 0 ---
    try:
        ax.contour(
            t0_grid, phi_grid, dose_minus_eta1_diff,
            levels=[0.0],
            colors='black',
            linewidths=2.0,
            linestyles='--'
        )
    except Exception as e:
        print(f"Note: Contour plotting may have issues if data is all NaN/Inf. {e}")

    # --- NEW: Create DYNAMIC legend ---
    patches = []
    for idx in present_categories:
        # idx is an integer from 0 to 9
        patches.append(mpatches.Patch(color=all_colors[idx], label=all_labels[idx]))

    # Add an entry for the contour line (it's always visible in this view)
    from matplotlib.lines import Line2D
    contour_legend_entry = Line2D([0], [0],
                                  color='black',
                                  lw=2.0,
                                  ls='--',
                                  label=r'$\eta^{dose}_{2 photon,\max} = \eta_{1 photon,\max}$ ')
    patches.append(contour_legend_entry)
    # --- Legend (dynamic) with bigger text and no title ---
    ax.legend(
        handles=patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title=None,  # <-- no legend title
        fontsize=13,  # <-- bigger legend labels
        handlelength=1.6
    )

    # Axis labels (bigger) and ticks
    ax.set_xlabel(r"$t_0$", fontsize=15)
    ax.set_ylabel(r"$\varphi$", fontsize=15)
    ax.tick_params(axis='both', labelsize=13)

    # Cleaner, shorter title
    ax.set_title(
        r"Optimal configurations for both Dose reduction and Phase Sensitivity "
        ,fontsize=15, pad=8
    )

    # --- Set axis limits for the zoomed-in view ---
    ax.set_xlim(0.5, 0.99)
    ax.set_ylim(0.01, np.pi / 2)

    # --- Ensure the data region itself is perfectly square ---
    ax.set_xlim(0.5, 0.99)
    ax.set_ylim(0.01, np.pi / 2)

    # Match data range ratio to axes box
    ax.set_aspect(
        (ax.get_xlim()[1] - ax.get_xlim()[0]) /
        (ax.get_ylim()[1] - ax.get_ylim()[0])
    )



    fig.tight_layout()
    # plt.savefig("overlap_analysis_plot_10cat_ZOOMED_dynamic_legend.png", dpi=300, bbox_inches='tight')
    # print("Plot saved as 'overlap_analysis_plot_10cat_ZOOMED_dynamic_legend.png'")
    plt.show() # Commented out for non-interactive environments


if __name__ == "__main__":
    main()