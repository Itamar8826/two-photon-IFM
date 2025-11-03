#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI to explore overlap categories and heatmaps for dose-reduction functions
on the (t0, phi) plane.

Updates in this version:
- Font controls for A4: set Title / Axis Label / Tick font sizes (right pane).
- Title box to set custom plot title dynamically.
- Save… button (PNG/PDF/SVG/EPS) with DPI control (default 600).
- Categories plot uses a fixed high-contrast palette.
- Saved plots force a square data axes box (independent of window).
- Heatmaps show a white vertical line at the *accurate* minimal t0 where the
  function first becomes non-negative for some φ, and put that t0* as a major tick.
  Default ticks are 1 decimal; the special t0* tick shows 3 decimals.

Run (PyCharm or CLI):
    python dose_overlap_and_heatmaps_gui.py --grid 180 --nT 201
"""

import argparse
import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

# Embed TrueType in EPS/PDF (journal-friendly)
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42

matplotlib.use("TkAgg")

# -------------------------- Math helpers (vectorized) --------------------------
EPS = 1e-12

CAT_COLORS = [
    "#999999",  # Gray

    "#0072B2",  # Blue


    "#D55E00",  # Vermillion
    "#CC79A7",  # Magenta
    "#117733",  # Dark Green
    "#56B4E9",  # Sky Blue
    "#E69F00",  # Orange
    "#000000",  # Black
    "#009E73",  # Green
    "#882255",  # Wine
    "#44AA99",  # Teal

    "#DDCC77",  # Sand
    "#332288",  # Navy
    "#F0E442",  # Yellow
    "#AA4499",  # Purple
    "#88CCEE"   # Light Cyan
]


def _add_xtick(ax, x, tol=1e-9):
    """Add x as a major tick (if not already present)."""
    lo, hi = ax.get_xlim()
    if not (lo <= x <= hi) or not np.isfinite(x):
        return
    ticks = np.array(ax.get_xticks(), dtype=float)
    if ticks.size and np.any(np.abs(ticks - x) < tol):
        return
    new_ticks = np.unique(np.append(ticks, x))
    ax.set_xticks(new_ticks)

def _safe_div(num: np.ndarray, den: np.ndarray, fill=np.nan) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        out = num / den
    out[~np.isfinite(out)] = fill
    return out


# -------------------------- Dose functions (vectorized) --------------------------

# -------------------------- Extra (standalone) function --------------------------
def eta1min(t0_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    """
    eta1min = - (1 + t^2 - 2 t cos(phi)) / (-3 + t^2 + 2 t cos(phi))
    Standalone: not included in any other calculations.
    """
    c = np.cos(phi_grid)
    num = 1.0 + t0_grid**2 - 2.0 * t0_grid * c
    den = -3.0 + t0_grid**2 + 2.0 * t0_grid * c
    return -_safe_div(num, den)

def eta1max(t0_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    """
    eta1max = (1 + t^2 - 2 t cos(phi)) / (2 - 2 t cos(phi))
    Standalone: not included in any other calculations.
    """
    c = np.cos(phi_grid)
    num = 1.0 + t0_grid**2 - 2.0 * t0_grid * c
    den = 2.0 - 2.0 * t0_grid * c
    return _safe_div(num, den)

def dosea_symmetric_20(t0_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    c = np.cos(phi_grid)
    num = 4.0 * t0_grid**2 - 4.0 * t0_grid * c
    den = 4.0 - 4.0 * t0_grid * c
    return _safe_div(num, den)

def dose_symmetric_20(t0_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    c = np.cos(phi_grid)
    c2 = np.cos(2.0 * phi_grid)
    num = 13.0 - 20.0 * t0_grid**2 - 3.0 * t0_grid**4 - 4.0 * (t0_grid + t0_grid**3) * c + 2.0 * t0_grid**2 * c2
    den = -15.0 + 4.0 * t0_grid**2 + t0_grid**4 - 4.0 * (t0_grid + t0_grid**3) * c + 2.0 * t0_grid**2 * c2
    return _safe_div(num, den)

def dose_symmetric_11(t0_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    c2 = np.cos(2.0 * phi_grid)
    num = -3.0 + 4.0 * t0_grid**2 + t0_grid**4 - 2.0 * t0_grid**2 * c2
    den = -3.0 + t0_grid**4 + 2.0 * t0_grid**2 * c2
    return -_safe_div(num, den)

def f11_max_over_T(T_points: np.ndarray, t0_grid: np.ndarray, phi_grid: np.ndarray):
    # Broadcast to (nT, Nphi, Nt)
    T = T_points[:, None, None]
    t = t0_grid[None, :, :]
    phi = phi_grid[None, :, :]
    c = np.cos(phi)

    term1 = (
        -1.0
        + T * (2.0 + T * (-6.0 - 4.0 * (-2.0 + T) * T))
        + (1.0 + T * (4.0 - 4.0 * T * (3.0 - 4.0 * T + 2.0 * T**2))) * t**2
        + T * (2.0 + T * (-6.0 - 4.0 * (-2.0 + T) * T)) * t**4
        + 4.0 * (-1.0 + T) * T * t * c * (((1.0 - 2.0 * T) ** 2) * (1.0 + t**2)
        - 4.0 * (-1.0 + T) * T * t * c)
    )

    term2 = (
        1.0 - t**2
        + T * (
            8.0 * t**2
            + 8.0 * T**2 * (1.0 + t**2) ** 2
            - 4.0 * T**3 * (1.0 + t**2) ** 2
            - 4.0 * T * (1.0 + 4.0 * t**2 + t**4)
        )
        + 4.0 * (-1.0 + T) * T * t * c * (((1.0 - 2.0 * T) ** 2) * (1.0 + t**2)
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


# -------------------------- Category computation (vectorized) --------------------------
def compute_all_maps(grid: int, nT: int,
                     t_min=0.01, t_max=0.99, phi_min=0.01, phi_max=math.pi - 0.01,
                     T_lo=0.001, T_hi=0.999):
    # Axes and grid
    t_vals = np.linspace(t_min, t_max, grid + 1)
    phi_vals = np.linspace(phi_min, phi_max, grid + 1)
    T_grid, P_grid = np.meshgrid(t_vals, phi_vals)  # shapes (Nphi, Nt)

    # Base functions
    f1 = dosea_symmetric_20(T_grid, P_grid)
    f2 = dose_symmetric_20(T_grid, P_grid)
    f3 = dose_symmetric_11(T_grid, P_grid)

    # f11 max over T
    T_points = np.linspace(T_lo, T_hi, nT)
    f4, T_star = f11_max_over_T(T_points, T_grid, P_grid)

    # Positivity masks
    m1 = f1 > 0.0
    m2 = f2 > 0.0
    m3 = f3 > 0.0
    m4 = f4 > 0.0

    # Category map (matches your previous mapping)
    cats = np.zeros_like(f1, dtype=int)
    cats = np.where(m1 & m2 & m3 & m4, 15, cats)
    cats = np.where(m1 & m2 & m3 & ~m4, 7, cats)
    cats = np.where(m1 & m2 & ~m3 & m4, 11, cats)
    cats = np.where(m1 & ~m2 & m3 & m4, 13, cats)
    cats = np.where(~m1 & m2 & m3 & m4, 14, cats)
    cats = np.where(m1 & m2 & ~m3 & ~m4, 4, cats)
    cats = np.where(m1 & ~m2 & m3 & ~m4, 5, cats)
    cats = np.where(m1 & ~m2 & ~m3 & m4, 9, cats)
    cats = np.where(~m1 & m2 & m3 & ~m4, 6, cats)
    cats = np.where(~m1 & m2 & ~m3 & m4, 10, cats)
    cats = np.where(~m1 & ~m2 & m3 & m4, 12, cats)
    cats = np.where(m1 & ~m2 & ~m3 & ~m4, 1, cats)
    cats = np.where(~m1 & m2 & ~m3 & ~m4, 2, cats)
    cats = np.where(~m1 & ~m2 & m3 & ~m4, 3, cats)
    cats = np.where(~m1 & ~m2 & ~m3 & m4, 8, cats)
    # else remains 0

    # Max-of-all and Source Map
    stack = np.stack([f1, f2, f3, f4], axis=0)  # (4, Nphi, Nt)
    max_vals = np.nanmax(stack, axis=0)
    argmax_idx = np.nanargmax(stack, axis=0)  # 0..3
    source = argmax_idx + 1
    source[max_vals <= 0.0] = 0

    return {
        't_vals': t_vals,
        'phi_vals': phi_vals,
        'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4,
        'T_star': T_star, 'T_points': T_points,
        'cats': cats,
        'max_vals': max_vals,
        'source': source,
    }


# -------------------------- Accurate t0* helpers --------------------------
def _max_over_phi_f1(t0: float, phi_vals: np.ndarray) -> float:
    tcol = np.full_like(phi_vals, fill_value=t0, dtype=float)
    return np.nanmax(dosea_symmetric_20(tcol, phi_vals))

def _max_over_phi_f2(t0: float, phi_vals: np.ndarray) -> float:
    tcol = np.full_like(phi_vals, fill_value=t0, dtype=float)
    return np.nanmax(dose_symmetric_20(tcol, phi_vals))

def _max_over_phi_f3(t0: float, phi_vals: np.ndarray) -> float:
    tcol = np.full_like(phi_vals, fill_value=t0, dtype=float)
    return np.nanmax(dose_symmetric_11(tcol, phi_vals))

def _max_over_phi_f4(t0: float, phi_vals: np.ndarray, T_points: np.ndarray) -> float:
    t_grid = np.full((phi_vals.shape[0], 1), t0, dtype=float)
    phi_grid = phi_vals[:, None]
    fmax, _ = f11_max_over_T(T_points, t_grid, phi_grid)  # (Nphi,1)
    return np.nanmax(fmax[:, 0])

def _bracket_first_nonneg(grid2d: np.ndarray):
    """
    Given data (Nphi, Nt), find first t-index j where any value >= 0.
    Returns (j_neg, j_nonneg) as indices that bracket the threshold.
    If no nonneg column found -> None.
    If first column already nonneg -> (None, 0).
    """
    pos_any = np.any(np.isfinite(grid2d) & (grid2d >= 0.0), axis=0)
    if not np.any(pos_any):
        return None
    j_nonneg = int(np.argmax(pos_any))
    j_neg = None if j_nonneg == 0 else j_nonneg - 1
    return (j_neg, j_nonneg)

def _bisection_min_t0_nonneg(phi_vals: np.ndarray, t_low: float, t_high: float,
                             eval_max_over_phi, tol: float = 5e-5, max_iter: int = 60) -> float:
    """
    Find minimal t0 in [t_low, t_high] with max_phi f(t0,phi) >= 0.
    """
    lo, hi = float(t_low), float(t_high)
    # Bisection on g(t) = max_phi f(t,phi) crossing 0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = eval_max_over_phi(mid)
        if not np.isfinite(f_mid):
            mid = np.nextafter(mid, hi)
            f_mid = eval_max_over_phi(mid)
        if f_mid >= 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi  # first non-negative


# -------------------------- Plotting helpers --------------------------
CAT_LABELS = [
    "None (all ≤ 0)",
    r"$\eta_{20,02}^{\mathrm{AS}}$ only",
    r"$\eta_{20,02}^{\mathrm{S}}$ only",
    r"$\eta_{11}^{\mathrm{S}}$ only",
    r"$\eta_{20,02}^{\mathrm{AS}}$ + $\eta_{20,02}^{\mathrm{S}}$",
    r"$\eta_{20,02}^{\mathrm{AS}}$ + $\eta_{11}^{\mathrm{S}}$",
    r"$\eta_{20,02}^{\mathrm{S}}$ + $\eta_{11}^{\mathrm{S}}$",
    r"$\eta_{20,02}^{\mathrm{AS}}$ + $\eta_{20,02}^{\mathrm{S}}$ + $\eta_{11}^{\mathrm{S}}$",
    r"$\eta_{11}^{\mathrm{AS},\,\mathrm{opt}\,T}$ only",
    r"$\eta_{20,02}^{\mathrm{AS}}$ + $\eta_{11}^{\mathrm{AS},\,\mathrm{opt}\,T}$",
    r"$\eta_{20,02}^{\mathrm{S}}$ + $\eta_{11}^{\mathrm{AS},\,\mathrm{opt}\,T}$",
    r"$\eta_{20,02}^{\mathrm{AS}}$ + $\eta_{20,02}^{\mathrm{S}}$ + $\eta_{11}^{\mathrm{AS},\,\mathrm{opt}\,T}$",
    r"$\eta_{11}^{\mathrm{S}}$ + $\eta_{11}^{\mathrm{AS},\,\mathrm{opt}\,T}$",
    r"$\eta_{20,02}^{\mathrm{AS}}$ + $\eta_{11}^{\mathrm{S}}$ + $\eta_{11}^{\mathrm{AS},\,\mathrm{opt}\,T}$",
    r"$\eta_{20,02}^{\mathrm{S}}$ + $\eta_{11}^{\mathrm{S}}$ + $\eta_{11}^{\mathrm{AS},\,\mathrm{opt}\,T}$",
    "All",
]

SOURCE_LABELS = [
    "None positive",
    r"$\eta_{20,02}^{\mathrm{AS}}$",
    r"$\eta_{20,02}^{\mathrm{S}}$",
    r"$\eta_{11}^{\mathrm{S}}$",
    r"$\eta_{11}^{\mathrm{AS},\,\mathrm{opt}\,T}$",
]

def _discrete_plasma(N=16):
    base = plt.get_cmap('plasma', N)
    return base, [base(i) for i in range(N)]

def draw_category(ax, data, t_vals, phi_vals, show_borders: bool,
                  label_size=18, tick_size=16, legend_size=14, legend_title_size=15, title=""):
    """
    Dynamic categorical plot:
    - Uses colors only for categories present in 'data'
    - Legend shows only present categories
    - Keeps category->color consistent with CAT_COLORS index (stable across runs)
    """
    import numpy as np
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

    # Which categories are present?
    present = np.unique(data[np.isfinite(data)]).astype(int)
    # Safety: restrict to valid 0..15 range
    present = present[(present >= 0) & (present < len(CAT_COLORS))]

    if present.size == 0:
        # Nothing positive on the map; just show the raw image with a neutral colormap
        im = ax.imshow(
            data, origin="lower",
            extent=[t_vals[0], t_vals[-1], phi_vals[0], phi_vals[-1]],
            aspect="auto", interpolation="nearest", cmap="Greys"
        )
        ax.set_xlabel(r"$t_0$", fontsize=label_size)
        ax.set_ylabel(r"$\varphi$", fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        if title:
            ax.set_title(title, fontsize=label_size + 2)
        return im, None

    # Map original category values -> compact indices 0..K-1 (for imshow)
    cat_to_idx = {c: i for i, c in enumerate(present)}
    mapped = np.full_like(data, -1, dtype=int)
    for c, i in cat_to_idx.items():
        mapped[data == c] = i

    # Colors: pick from CAT_COLORS by original category value (stable identity)
    # If any category index is out of range, fall back to tab20
    fallback = plt.get_cmap("tab20").colors
    colors = []
    for c in present:
        if 0 <= c < len(CAT_COLORS):
            colors.append(CAT_COLORS[c])
        else:
            colors.append(fallback[c % len(fallback)])

    cmap = ListedColormap(colors, name="cat_present")
    boundaries = np.arange(-0.5, len(present) + 0.5, 1)
    norm = BoundaryNorm(boundaries, len(present))

    im = ax.imshow(
        mapped,
        origin="lower",
        extent=[t_vals[0], t_vals[-1], phi_vals[0], phi_vals[-1]],
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    # Optional white borders between categories
    if show_borders and mapped.size:
        Tm, Pm = np.meshgrid(t_vals, phi_vals)
        ax.contour(
            Tm, Pm, mapped,
            levels=np.arange(len(present)) + 0.5,
            colors='white', linewidths=0.9, antialiased=True
        )

    # Build legend only for present categories
    patches = [
        mpatches.Patch(color=colors[i], label=CAT_LABELS[c])
        for i, c in enumerate(present)
    ]
    # Use 1–2 columns depending on how many items we have
    ncols = 1 if len(patches) <= 8 else 2
    leg = ax.legend(
        handles=patches, loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=True, framealpha=1.0, title="Category", ncol=ncols, borderaxespad=0.5
    )
    for txt in leg.get_texts():
        txt.set_fontsize(legend_size)
    leg.get_title().set_fontsize(legend_title_size)

    ax.set_xlabel(r"$t_0$", fontsize=label_size)
    ax.set_ylabel(r"$\varphi$", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    if title:
        ax.set_title(title, fontsize=label_size + 2)

    return im, leg


def _make_axes_square(ax):
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)  # square axes box
    else:
        ax.set_aspect('equal', adjustable='box')

def draw_heatmap(ax, data, t_vals, phi_vals, title: str, show_zero: bool,
                 label_size=18, tick_size=16, title_size=20, draw_min_tpos=True,
                 which_func=None, T_points=None, cbar_label=None):


    """
    Returns (im, cbar, tpos) where tpos is the accurate minimal t0 with max_phi f >= 0.
    """
    im = ax.imshow(
        data,
        origin="lower",
        extent=[t_vals[0], t_vals[-1], phi_vals[0], phi_vals[-1]],
        aspect="auto",
        interpolation="bilinear",
        cmap='plasma',
    )

    # optional zero contour
    if show_zero:
        Tm, Pm = np.meshgrid(t_vals, phi_vals)
        ax.contour(Tm, Pm, data, levels=[0.0], colors='k', linewidths=1.6)
        ax.contour(Tm, Pm, data, levels=[0.0], colors='white', linewidths=0.9)

    # Accurate tpos (minimal t0 with max_phi >= 0): bracket on grid, then bisection
    tpos = None
    if draw_min_tpos and which_func is not None:
        br = _bracket_first_nonneg(data)
        if br is not None:
            j_neg, j_nonneg = br
            if j_neg is None:
                # already nonnegative at the first column
                tpos = float(t_vals[0])
            else:
                t_low, t_high = float(t_vals[j_neg]), float(t_vals[j_nonneg])
                if which_func == "f1":
                    eval_fun = lambda t0: _max_over_phi_f1(t0, phi_vals)
                elif which_func == "f2":
                    eval_fun = lambda t0: _max_over_phi_f2(t0, phi_vals)
                elif which_func == "f3":
                    eval_fun = lambda t0: _max_over_phi_f3(t0, phi_vals)
                elif which_func == "f4":
                    eval_fun = lambda t0: _max_over_phi_f4(t0, phi_vals, T_points)
                else:
                    eval_fun = None
                if eval_fun is not None:
                    tpos = _bisection_min_t0_nonneg(phi_vals, t_low, t_high, eval_fun, tol=5e-5, max_iter=60)

        if tpos is not None and np.isfinite(tpos):
            ax.axvline(tpos, color='white', linewidth=1.2, linestyle=(0, (6, 3)))
            _add_xtick(ax, tpos)


    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label if cbar_label is not None else "", fontsize=label_size)

    cbar.ax.tick_params(labelsize=tick_size)

    ax.set_xlabel(r"$t_0$", fontsize=label_size)
    ax.set_ylabel(r"$\varphi$", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    if title:
        ax.set_title(title, fontsize=title_size)

    return im, cbar, tpos

def draw_source_map(ax, source, t_vals, phi_vals, show_borders: bool,
                    label_size=18, tick_size=16, legend_size=14, legend_title_size=15, title=""):
    src_colors = ListedColormap([
        (0.9, 0.9, 0.9, 1.0),   # 0: none positive (light gray)
        (0.14, 0.0, 0.5, 1.0),  # 1
        (0.4, 0.0, 0.85, 1.0),  # 2
        (0.9, 0.2, 0.4, 1.0),   # 3
        (0.99, 0.9, 0.15, 1.0), # 4
    ])
    boundaries = np.arange(-0.5, 5.5, 1)
    norm = BoundaryNorm(boundaries, src_colors.N)
    im = ax.imshow(
        source,
        origin="lower",
        extent=[t_vals[0], t_vals[-1], phi_vals[0], phi_vals[-1]],
        aspect="auto",
        interpolation="nearest",
        cmap=src_colors,
        norm=norm,
    )
    if show_borders:
        Tm, Pm = np.meshgrid(t_vals, phi_vals)
        ax.contour(Tm, Pm, source, levels=np.arange(5) + 0.5, colors='white', linewidths=0.9, antialiased=True)

    patches = [mpatches.Patch(color=src_colors(i), label=SOURCE_LABELS[i]) for i in range(src_colors.N)]
    leg = ax.legend(handles=patches, loc="center left", bbox_to_anchor=(1.01, 0.5),
                    frameon=True, title="Source", ncol=1, borderaxespad=0.5)
    for txt in leg.get_texts():
        txt.set_fontsize(legend_size)
    leg.get_title().set_fontsize(legend_title_size)

    ax.set_xlabel(r"$t_0$", fontsize=label_size)
    ax.set_ylabel(r"$\varphi$", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    if title:
        ax.set_title(title, fontsize=label_size + 2)

    return im, leg


# -------------------------- GUI --------------------------
class DoseGUI:
    def __init__(self, master, grid: int, nT: int):
        self.master = master
        self.master.title("Dose Reduction: Overlap & Heatmaps")
        self.master.geometry("1280x860")

        # Core controls
        self.grid_points = tk.IntVar(value=grid)
        self.nT_points = tk.IntVar(value=nT)
        self.show_borders = tk.BooleanVar(value=True)
        self.plot_choice = tk.StringVar(value="Categories")

        # Font controls (A4-friendly defaults)
        self.title_size = tk.IntVar(value=20)
        self.label_size = tk.IntVar(value=18)
        self.tick_size = tk.IntVar(value=16)
        self.legend_size = tk.IntVar(value=14)
        self.legend_title_size = tk.IntVar(value=15)

        # Title text
        self.custom_title = tk.StringVar(value="")

        # Save controls
        self.save_dpi = tk.IntVar(value=600)
        self.save_format = tk.StringVar(value="png")  # png/pdf/svg/eps

        # Track last legend/colorbar so we can remove them when switching plots
        self._last_legend = None
        self._last_cbar = None

        # Track current special tpos value for tick formatting
        self._tpos_value = None

        self.data = None  # computed maps

        # Layout
        main = ttk.Frame(master)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        controls = ttk.Frame(main)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        plot_frame = ttk.Frame(main)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Controls ---
        ttk.Label(controls, text="Plot").pack(anchor='w')
        plot_opts = [
            "Categories",
            "Heatmap: eta20 AS",
            "Heatmap: eta20 S",
            "Heatmap: eta11 S",
            "Heatmap: eta11 AS (maxT)",
            "Heatmap: Max of all",
            "Heatmap: eta1 max (coh, single-port)",  # existing NEW (from before)
            "Heatmap: (Max of all) - eta1 max",  # existing NEW (from before)
            "Heatmap: eta1 min (coh, single-port)",  # <-- NEW
            "Heatmap: eta1 max - eta1 min",  # <-- NEW
            "Source Map (argmax)",
        ]

        self.plot_menu = ttk.Combobox(controls, textvariable=self.plot_choice, values=plot_opts, state="readonly")
        self.plot_menu.pack(fill=tk.X)
        self.plot_menu.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        ttk.Separator(controls, orient='horizontal').pack(fill=tk.X, pady=6)

        # Grid / nT
        row = ttk.Frame(controls); row.pack(fill=tk.X)
        ttk.Label(row, text="Grid").pack(side=tk.LEFT)
        self.grid_entry = ttk.Spinbox(row, from_=40, to=1000, increment=20, textvariable=self.grid_points, width=7)
        self.grid_entry.pack(side=tk.RIGHT)

        row = ttk.Frame(controls); row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row, text="nT").pack(side=tk.LEFT)
        self.nT_entry = ttk.Spinbox(row, from_=41, to=1201, increment=20, textvariable=self.nT_points, width=7)
        self.nT_entry.pack(side=tk.RIGHT)

        self.border_chk = ttk.Checkbutton(controls, text="Show overlay (borders/zero)", variable=self.show_borders,
                                          command=self.update_plot)
        self.border_chk.pack(anchor='w', pady=(6, 6))

        # Title controls
        ttk.Label(controls, text="Graph title").pack(anchor='w')
        title_entry = ttk.Entry(controls, textvariable=self.custom_title)
        title_entry.pack(fill=tk.X)
        title_entry.bind("<Return>", lambda e: self.update_plot())
        ttk.Button(controls, text="Apply Title", command=self.update_plot).pack(fill=tk.X, pady=(4, 8))

        # Font size controls
        ttk.Label(controls, text="Font sizes (A4-friendly)").pack(anchor='w')
        f = ttk.Frame(controls); f.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(f, text="Title").pack(side=tk.LEFT); ttk.Spinbox(f, from_=10, to=48, textvariable=self.title_size, width=5, command=self.update_plot).pack(side=tk.RIGHT)

        f = ttk.Frame(controls); f.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(f, text="Axis labels").pack(side=tk.LEFT); ttk.Spinbox(f, from_=10, to=36, textvariable=self.label_size, width=5, command=self.update_plot).pack(side=tk.RIGHT)

        f = ttk.Frame(controls); f.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(f, text="Ticks").pack(side=tk.LEFT); ttk.Spinbox(f, from_=8, to=30, textvariable=self.tick_size, width=5, command=self.update_plot).pack(side=tk.RIGHT)

        f = ttk.Frame(controls); f.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(f, text="Legend").pack(side=tk.LEFT); ttk.Spinbox(f, from_=8, to=28, textvariable=self.legend_size, width=5, command=self.update_plot).pack(side=tk.RIGHT)

        f = ttk.Frame(controls); f.pack(fill=tk.X, pady=(2, 6))
        ttk.Label(f, text="Legend title").pack(side=tk.LEFT); ttk.Spinbox(f, from_=8, to=28, textvariable=self.legend_title_size, width=5, command=self.update_plot).pack(side=tk.RIGHT)

        # Recompute + Save
        self.btn_recompute = ttk.Button(controls, text="Recompute", command=self.recompute)
        self.btn_recompute.pack(fill=tk.X, pady=(4, 0))

        # Save options row
        ttk.Separator(controls, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Label(controls, text="Export").pack(anchor='w')
        sf = ttk.Frame(controls); sf.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(sf, text="Format").pack(side=tk.LEFT)
        ttk.Combobox(sf, textvariable=self.save_format,
                     values=["png", "pdf", "svg", "eps"],
                     width=6, state="readonly").pack(side=tk.RIGHT)

        sf2 = ttk.Frame(controls); sf2.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(sf2, text="DPI").pack(side=tk.LEFT)
        ttk.Spinbox(sf2, from_=150, to=1200, increment=50, textvariable=self.save_dpi, width=6).pack(side=tk.RIGHT)

        self.btn_save = ttk.Button(controls, text="Save…", command=self.save_current_plot)
        self.btn_save.pack(fill=tk.X, pady=(6, 0))

        self.status = ttk.Label(controls, text="Ready", foreground="#555")
        self.status.pack(anchor='w', pady=(8, 0))

        # Figure/Canvas
        self.fig, self.ax = plt.subplots(figsize=(8.6, 6.2))
        _make_axes_square(self.ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial compute & draw
        self.recompute()

    def set_status(self, msg: str):
        self.status.config(text=msg)
        self.status.update_idletasks()

    def recompute(self):
        grid = int(self.grid_points.get())
        nT = int(self.nT_points.get())
        if grid < 20 or nT < 21:
            messagebox.showwarning("Parameters too small", "Increase Grid >= 20 and nT >= 21.")
            return
        self.set_status(f"Computing (grid={grid}, nT={nT}) …")
        self.master.config(cursor="watch")
        try:
            self.data = compute_all_maps(grid=grid, nT=nT)
        except Exception as e:
            messagebox.showerror("Computation error", str(e))
            self.master.config(cursor="")
            self.set_status("Error")
            return
        self.master.config(cursor="")
        self.set_status("Done")
        self.update_plot()

    def _clear_prev_decor(self):
        # Remove any previous legend/colorbar so they don't accumulate
        if getattr(self, "_last_legend", None) is not None:
            try:
                self._last_legend.remove()
            except Exception:
                pass
            self._last_legend = None
        if getattr(self, "_last_cbar", None) is not None:
            try:
                if hasattr(self._last_cbar, "remove"):
                    self._last_cbar.remove()
                elif hasattr(self._last_cbar, "ax"):
                    self._last_cbar.ax.remove()
            except Exception:
                pass
            self._last_cbar = None

    def update_plot(self):
        if self.data is None:
            return

        choice = self.plot_choice.get()
        show_b = bool(self.show_borders.get())
        title_text = self.custom_title.get().strip()

        self._clear_prev_decor()
        self.ax.clear()
        self._tpos_value = None  # reset unless set by a heatmap branch

        t_vals = self.data['t_vals']
        phi_vals = self.data['phi_vals']
        cats = self.data['cats']

        # Font sizes
        lbl_sz = int(self.label_size.get())
        tck_sz = int(self.tick_size.get())
        ttl_sz = int(self.title_size.get())
        leg_sz = int(self.legend_size.get())
        leg_t_sz = int(self.legend_title_size.get())

        # Plot according to choice
        leg_or_cbar = None
        if choice == "Categories":
            default_title = "Positive Regions of Dose Reduction (categorical)"
            im, leg_or_cbar = draw_category(
                self.ax, cats, t_vals, phi_vals, show_b,
                label_size=lbl_sz, tick_size=tck_sz,
                legend_size=leg_sz, legend_title_size=leg_t_sz,
                title=(title_text or default_title)
            )


        elif choice == "Heatmap: eta20 AS":

            default_title = r"Heatmap: $\eta_{20,02}^{\mathrm{AS}}$"

            im, leg_or_cbar, tpos = draw_heatmap(

                self.ax, self.data['f1'], t_vals, phi_vals,

                title=(title_text or default_title), show_zero=show_b,

                label_size=lbl_sz, tick_size=tck_sz, title_size=ttl_sz,

                draw_min_tpos=True, which_func="f1",

                cbar_label=r"$\eta_{20,02}^{\mathrm{AS}}$"

            )

            self._tpos_value = tpos


        elif choice == "Heatmap: eta20 S":

            default_title = r"Heatmap: $\eta_{20,02}^{\mathrm{S}}$"

            im, leg_or_cbar, tpos = draw_heatmap(

                self.ax, self.data['f2'], t_vals, phi_vals,

                title=(title_text or default_title), show_zero=show_b,

                label_size=lbl_sz, tick_size=tck_sz, title_size=ttl_sz,

                draw_min_tpos=True, which_func="f2",

                cbar_label=r"$\eta_{20,02}^{\mathrm{S}}$"

            )

            self._tpos_value = tpos


        elif choice == "Heatmap: eta11 S":

            default_title = r"Heatmap: $\eta_{11}^{\mathrm{S}}$"

            im, leg_or_cbar, tpos = draw_heatmap(

                self.ax, self.data['f3'], t_vals, phi_vals,

                title=(title_text or default_title), show_zero=show_b,

                label_size=lbl_sz, tick_size=tck_sz, title_size=ttl_sz,

                draw_min_tpos=True, which_func="f3",

                cbar_label=r"$\eta_{11}^{\mathrm{S}}$"

            )

            self._tpos_value = tpos


        elif choice == "Heatmap: eta11 AS (maxT)":

            default_title = r"Heatmap: $\eta_{11}^{\mathrm{AS},\,\mathrm{opt}\,T}$"

            im, leg_or_cbar, tpos = draw_heatmap(

                self.ax, self.data['f4'], t_vals, phi_vals,

                title=(title_text or default_title), show_zero=show_b,

                label_size=lbl_sz, tick_size=tck_sz, title_size=ttl_sz,

                draw_min_tpos=True, which_func="f4", T_points=self.data['T_points'],

                cbar_label=r"$\eta_{11}^{\mathrm{AS},\,\mathrm{opt}\,T}$"

            )

            self._tpos_value = tpos


        elif choice == "Heatmap: Max of all":
            default_title = r"Heatmap: $\max\{f_1,f_2,f_3,f_4\}$"
            im, leg_or_cbar, _ = draw_heatmap(
                self.ax, self.data['max_vals'], t_vals, phi_vals,
                title=(title_text or default_title),
                show_zero=show_b,
                label_size=lbl_sz, tick_size=tck_sz, title_size=ttl_sz,
                draw_min_tpos=False  # no special tpos for composite map
            )

        elif choice == "Source Map (argmax)":
            default_title = "Source Map (function attaining the maximum)"
            im, leg_or_cbar = draw_source_map(
                self.ax, self.data['source'], t_vals, phi_vals, show_b,
                label_size=lbl_sz, tick_size=tck_sz,
                legend_size=leg_sz, legend_title_size=leg_t_sz,
                title=(title_text or default_title)
            )
        elif choice == "Heatmap: eta1 max (coh, single-port)":
            # Build grids on the fly (standalone; not stored in data)
            Tm, Pm = np.meshgrid(t_vals, phi_vals)  # shapes (Nphi, Nt)
            eta = eta1max(Tm, Pm)
            default_title = r" One photon efficiency: $\eta_{1,AS} = \dfrac{1+t_0^2-2t_0\cos\varphi}{2-2t_0\cos\varphi}$"
            im, leg_or_cbar, _ = draw_heatmap(
                self.ax, eta, t_vals, phi_vals,
                title=(title_text or default_title),
                show_zero=show_b,
                label_size=lbl_sz, tick_size=tck_sz, title_size=ttl_sz,
                draw_min_tpos=False,
                cbar_label=None
            )

        elif choice == "Heatmap: (Max of all) - eta1 max":
            # Build eta1max and subtract from the existing 'max of all' (f1..f4)
            Tm, Pm = np.meshgrid(t_vals, phi_vals)  # shapes (Nphi, Nt)
            eta = eta1max(Tm, Pm)
            diff = self.data['max_vals'] - eta  # standalone combination
            default_title = r"$\eta^{dose}_{2 photon,\max}-\eta_{1 photon,\max}$"
            im, leg_or_cbar, _ = draw_heatmap(
                self.ax, diff, t_vals, phi_vals,
                title=(title_text or default_title),
                show_zero=show_b,
                label_size=lbl_sz, tick_size=tck_sz, title_size=ttl_sz,
                draw_min_tpos=False,
                cbar_label=None
            )
        elif choice == "Heatmap: eta1 min (coh, single-port)":
            Tm, Pm = np.meshgrid(t_vals, phi_vals)  # (Nphi, Nt)
            eta = eta1min(Tm, Pm)
            default_title = r"Heatmap: $\eta_{1,\min} = -\dfrac{1+t_0^2-2t_0\cos\varphi}{-3+t_0^2+2t_0\cos\varphi}$"
            im, leg_or_cbar, _ = draw_heatmap(
                self.ax, eta, t_vals, phi_vals,
                title=(title_text or default_title),
                show_zero=show_b,
                label_size=lbl_sz, tick_size=tck_sz, title_size=ttl_sz,
                draw_min_tpos=False,
                cbar_label=r"$\eta_{1,\min}$"
            )

        elif choice == "Heatmap: eta1 max - eta1 min":
            Tm, Pm = np.meshgrid(t_vals, phi_vals)  # (Nphi, Nt)
            emax = eta1max(Tm, Pm)
            emin = eta1min(Tm, Pm)
            diff = emax - emin
            default_title = r"Heatmap: $\eta_{1,\max}-\eta_{1,\min}$"
            im, leg_or_cbar, _ = draw_heatmap(
                self.ax, diff, t_vals, phi_vals,
                title=(title_text or default_title),
                show_zero=show_b,
                label_size=lbl_sz, tick_size=tck_sz, title_size=ttl_sz,
                draw_min_tpos=False,
                cbar_label=r"$\eta_{1,\max}-\eta_{1,\min}$"
            )

        # Remember legend or colorbar to remove next time
        if choice in ("Categories", "Source Map (argmax)"):
            self._last_legend = leg_or_cbar
        else:
            self._last_cbar = leg_or_cbar

        _make_axes_square(self.ax)

        # -------- Tick formatter: default 1 decimal; tpos tick 3 decimals --------
        tpos = self._tpos_value
        def _fmt_x(v, pos, special=tpos, tol=5e-4):
            if special is not None and np.isfinite(special) and abs(v - special) < tol:
                return f"{v:.3f}"
            return f"{v:.1f}"
        self.ax.xaxis.set_major_formatter(FuncFormatter(_fmt_x))

        self.fig.tight_layout()
        self.canvas.draw()

    def save_current_plot(self):
        fmt = self.save_format.get()
        dpi = int(self.save_dpi.get())
        # Default filename suggestion
        initial = f"plot_{self.plot_choice.get().split(':')[0].strip().lower().replace(' ', '_')}.{fmt}"
        fname = filedialog.asksaveasfilename(
            title="Save current plot",
            defaultextension=f".{fmt}",
            initialfile=initial,
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg"), ("EPS", "*.eps"), ("All Files", "*.*")]
        )
        if not fname:
            return
        if not fname.lower().endswith(("png", "pdf", "svg", "eps")):
            fname = fname + f".{fmt}"
        try:
            _make_axes_square(self.ax)
            self.fig.savefig(fname, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
            self.set_status(f"Saved: {fname} (dpi={dpi})")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


# -------------------------- CLI entry --------------------------
def main():
    ap = argparse.ArgumentParser(description="GUI: overlap categories + heatmaps on (t0, phi)")
    ap.add_argument("--grid", type=int, default=160, help="grid size per axis (default: 160)")
    ap.add_argument("--nT", type=int, default=201, help="# of T-samples for f11 max (default: 201)")
    args = ap.parse_args()

    root = tk.Tk()
    app = DoseGUI(root, grid=args.grid, nT=args.nT)
    root.mainloop()

if __name__ == "__main__":
    main()
