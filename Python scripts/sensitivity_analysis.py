import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, LogNorm
import matplotlib.patches as mpatches
import tkinter as tk
from tkinter import ttk, filedialog
import warnings
import os, sys, time
import re

# NEW: stable colorbar placement without shrinking the plot area
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogLocator, LogFormatter, MaxNLocator

warnings.filterwarnings('ignore')

import matplotlib as mpl

mpl.rcParams.update({
    # --- Page & export ---
    "figure.figsize": (5.83, 8.27),  # A5 portrait in inches (half A4)
    "savefig.dpi": 300,  # crisp on paper
    "pdf.fonttype": 42,  # keep text editable in PDFs
    "ps.fonttype": 42,

    # --- Fonts (tuned for A5 @300dpi) ---
    "font.size": 14,  # base
    "axes.titlesize": 22,  # plot title
    "figure.titlesize": 22,  # suptitle
    "axes.labelsize": 20,  # axis labels
    "xtick.labelsize": 13,  # tick labels
    "ytick.labelsize": 13,
    "legend.fontsize": 13,  # legend text
    "legend.title_fontsize": 13,

    # --- Spacing / visibility ---
    "axes.titlepad": 10,
    "axes.labelpad": 6,

    # --- Lines (helps readability) ---
    "lines.linewidth": 1.6,
    "grid.alpha": 0.25,
})


# -------------------- Scrollable controls column --------------------
class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, width=320, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        self.frame = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.frame, anchor="nw")

        self.frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.frame.bind("<Enter>", self._bind_mousewheel)
        self.frame.bind("<Leave>", self._unbind_mousewheel)

        self.update_idletasks()
        self.canvas.itemconfigure(self._window, width=width)

    def _on_frame_configure(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            delta = event.delta
            if delta == 0:
                return
            direction = -1 if delta > 0 else 1
            self.canvas.yview_scroll(direction * 3, "units")

    def _bind_mousewheel(self, _):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-5>", self._on_mousewheel, add="+")

    def _unbind_mousewheel(self, _):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")


class QuantumSensingGUI:
    # -------------------- Small helpers --------------------
    def _braket_label(self) -> str:
        mu, nu = self.mu_nu_options[self.current_mu_nu.get()]
        return f"|{mu}{nu}⟩"

    def _format_T(self) -> str:
        T = float(self.current_T.get())
        if np.isclose(T, 0.0, atol=1e-3):
            return "0"
        if np.isclose(T, 1.0, atol=1e-3):
            return "1"
        if np.isclose(T, 0.5, atol=5e-4):
            return "0.5"
        return f"{T:.3f}".rstrip("0").rstrip(".")

    @staticmethod
    def _sin2(phi):
        s2 = np.sin(phi) ** 2
        return np.maximum(s2, 1e-12)

    @staticmethod
    def _sqrt_pos(x):
        return np.sqrt(np.maximum(x, 0.0))

    @staticmethod
    def _safe_over(num, den, tiny=1e-300):
        den_safe = np.where(np.abs(den) < tiny, np.sign(den) * tiny + (den == 0) * tiny, den)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            out = num / den_safe
        return out

    @staticmethod
    def _finite_pos(x):
        x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
        return x

    @staticmethod
    def _safe_ratio(num, den, clip=(1e-12, 1e300)):
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            r = num / den
        bad = (~np.isfinite(num)) | (~np.isfinite(den)) | (num <= 0) | (den <= 0) | (~np.isfinite(r))
        r[bad] = np.nan
        if clip is not None:
            mask = np.isfinite(r)
            r[mask] = np.clip(r[mask], clip[0], clip[1])
        return r

    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Sensing — s(μ,ν), SensCoherent, Ratios, and Global s-min")
        self.root.geometry("1200x820")

        # ---- Model parameters / state ----
        self.T_values = [0.001, 0.5, 0.999]
        self.mu_nu_options = {
            '01': (0, 1),
            '10': (1, 0),
            '11': (1, 1),
            '20': (2, 0),
            '02': (0, 2)
        }
        self.current_T = tk.DoubleVar(value=0.5)
        self.current_mu_nu = tk.StringVar(value='01')

        # Plot selection
        self.plot_var = tk.StringVar(value="s_mu_nu")
        self.resolution = tk.IntVar(value=100)
        self.use_custom_title = tk.BooleanVar(value=False)
        self.custom_title = tk.StringVar(value="")

        # Aggregation modes
        self.s_mode = tk.StringVar(value='min')  # 'min', 'N12', 'N1', 'N2'
        self.coh_mode = tk.StringVar(value='min')  # 'min', 'CoherentN12', 'Nb1', 'Nb2'

        # |11⟩ special T selection
        self.T11_mode = tk.StringVar(value='fixed')  # 'fixed' or 'dose-opt'
        self.T_map = None
        self.T_map_opt11 = None

        # Tie marking tolerance (relative)
        self.tie_rel = 0.03  # 3% within the min is considered "minor diff" → mark as co-winner

        # ---- Layout ----
        main_frame = ttk.Frame(root);
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_container = ttk.Frame(main_frame)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_scroll = ScrollableFrame(left_container, width=320)
        control_scroll.pack(fill=tk.BOTH, expand=True)
        control_frame = control_scroll.frame

        plot_frame = ttk.Frame(main_frame);
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(control_frame, text="Analysis Controls", font=('Arial', 14, 'bold')).pack(pady=(0, 10))

        # ---- Plot selection ----
        ttk.Label(control_frame, text="Select Plot:").pack(anchor=tk.W)
        plots = [
            ("s(μ,ν) (log scale)", "s_mu_nu"),
            ("SensCoherent (log scale)", "coherent"),
            ("Ratio: Coherent / s(μ,ν)", "ratio"),
            ("Sensitivity Ratio (log colors)", "ratio_log"),
            ("Comparison (s vs Coherent)", "comparison"),
            ("Source map — s(min of 3)", "source_s"),
            ("Source map — coherent(min of 3)", "source_coh"),
            ("Global s-min (restricted)", "global_smin"),
            ("Source map — Global s-min", "source_global_s"),
        ]
        for text, value in plots:
            ttk.Radiobutton(control_frame, text=text, variable=self.plot_var,
                            value=value, command=self.update_plot).pack(anchor=tk.W, pady=2)

        # ---- T selection ----
        ttk.Label(control_frame, text="Temperature T:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(12, 2))
        T_frame = ttk.Frame(control_frame);
        T_frame.pack(anchor=tk.W, pady=2)
        for T in self.T_values:
            ttk.Radiobutton(T_frame, text=str(T), variable=self.current_T,
                            value=T, command=self.recompute_and_plot).pack(side=tk.LEFT, padx=3)

        # ---- μ,ν selection ----
        ttk.Label(control_frame, text="μ, ν Selection:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(12, 2))
        muv_frame = ttk.Frame(control_frame);
        muv_frame.pack(anchor=tk.W, pady=2)
        muv_labels = [('|0⟩,|1⟩', '01'), ('|1⟩,|0⟩', '10'), ('|1⟩,|1⟩', '11'), ('|2⟩,|0⟩', '20'), ('|0⟩,|2⟩', '02')]
        for text, code in muv_labels:
            ttk.Radiobutton(muv_frame, text=text, variable=self.current_mu_nu,
                            value=code, command=self.recompute_and_plot).pack(anchor=tk.W)

        # ---- Aggregation controls ----
        ttk.Label(control_frame, text="Aggregation (Quantum s):", font=('Arial', 10, 'bold')).pack(anchor=tk.W,
                                                                                                   pady=(12, 2))
        s_frame = ttk.Frame(control_frame);
        s_frame.pack(anchor=tk.W, pady=2)
        for text, val in [("min of 3", 'min'), ("N12 only", 'N12'), ("N1 only", 'N1'), ("N2 only", 'N2')]:
            ttk.Radiobutton(s_frame, text=text, variable=self.s_mode, value=val, command=self.recompute_and_plot).pack(
                anchor=tk.W)

        ttk.Label(control_frame, text="Aggregation (Coherent):", font=('Arial', 10, 'bold')).pack(anchor=tk.W,
                                                                                                  pady=(8, 2))
        coh_frame = ttk.Frame(control_frame);
        coh_frame.pack(anchor=tk.W, pady=2)
        for text, val in [("min of 3", 'min'), ("CoherentN12", 'CoherentN12'), ("Nb1", 'Nb1'), ("Nb2", 'Nb2')]:
            ttk.Radiobutton(coh_frame, text=text, variable=self.coh_mode, value=val,
                            command=self.recompute_and_plot).pack(anchor=tk.W)

        # ---- |11⟩ T mode ----
        ttk.Label(control_frame, text="|11⟩ T mode:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(12, 2))
        t11_frame = ttk.Frame(control_frame);
        t11_frame.pack(anchor=tk.W, pady=2)
        for text, val in [("Use fixed T (above)", 'fixed'), ("Optimize T by dose11", 'dose-opt')]:
            ttk.Radiobutton(t11_frame, text=text, variable=self.T11_mode, value=val,
                            command=self.recompute_and_plot).pack(anchor=tk.W)

        # ---- Resolution ----
        ttk.Label(control_frame, text="Resolution:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(12, 5))
        ttk.Scale(control_frame, from_=20, to=200, variable=self.resolution,
                  orient=tk.HORIZONTAL, command=lambda x: self.update_resolution_label()).pack(fill=tk.X, pady=2)
        self.res_label = ttk.Label(control_frame, text="100x100");
        self.res_label.pack(anchor=tk.W)

        # ---- Buttons ----
        ttk.Button(control_frame, text="Update Plot", command=self.update_plot).pack(pady=(12, 5), fill=tk.X)
        ttk.Button(control_frame, text="Save Plot", command=self.save_plot).pack(pady=(0, 10), fill=tk.X)
        ttk.Button(control_frame, text="Save ALL (batch)", command=self.save_all_plots_batch).pack(pady=(0, 10),
                                                                                                   fill=tk.X)

        # ---- Custom title ----
        ttk.Label(control_frame, text="Custom Title:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(6, 2))
        title_entry = ttk.Entry(control_frame, textvariable=self.custom_title, width=30)
        title_entry.pack(fill=tk.X, pady=(0, 5))
        title_entry.bind('<Return>', lambda e: self.update_plot())
        ttk.Checkbutton(control_frame, text="Use Custom Title",
                        variable=self.use_custom_title,
                        command=self.update_plot).pack(anchor=tk.W, pady=(0, 10))

        # ---- Probe controls ----
        self.probe_enabled = tk.BooleanVar(value=True)
        ttk.Label(control_frame, text="Probe (click plot):", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(12, 2))
        ttk.Checkbutton(control_frame, text="Enable probe", variable=self.probe_enabled).pack(anchor=tk.W)
        self.probe_info = ttk.Label(control_frame, text="Click inside plot to inspect point",
                                    justify='left', wraplength=260)
        self.probe_info.pack(anchor=tk.W, pady=(4, 8))

        # ---- Progress / Status ----
        self.progress = ttk.Progressbar(control_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(6, 0))
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(anchor=tk.W, pady=(5, 0))

        # ---- Matplotlib figure ----
        self.fig = Figure(figsize=(10, 8), dpi=100)  # keep figure canvas constant
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._probe_pt = None
        self._probe_ann = None
        try:
            self.cid_click = self.canvas.mpl_connect('button_press_event', self.on_click)
        except Exception:
            self.cid_click = None

        # ---- Data holders ----
        self.colorbars = []
        self.tO_grid = None
        self.phi_grid = None
        self.sens_data = None
        self.coherent_data = None
        self.ratio_data = None

        # stacks for source maps (for co-winner marking)
        self.stack_s = None
        self.stack_c = None

        # Global min s (restricted set)
        self.global_smin_data = None
        self.global_smin_source_idx = None
        self.global_smin_stack = None  # store all candidates for tie marking
        self.global_smin_source_names = [
            "|02⟩  T=1.0 / |20⟩  T=0.0",
            "|20⟩/|20⟩  T=0.5",
            "|11⟩  T=0.5",
            "|11⟩  optimal T",
        ]

        # Fixed axes rectangles for consistent plot area
        # Single plot: [left, bottom, width, height]
        self._AX_SINGLE = [0.12, 0.12, 0.62, 0.76]
        # Two plots side-by-side
        self._AX_LEFT = [0.07, 0.12, 0.38, 0.76]
        self._AX_RIGHT = [0.55, 0.12, 0.38, 0.76]

        # Initial compute & plot
        self.compute_all_data()
        self.update_plot()

    # -------------------- Layout helpers (stable axes & colorbars) --------------------
    def _new_single_axes(self):
        """Create one data axes with a right colorbar axis; keep a square data box."""
        ax = self.fig.add_axes(self._AX_SINGLE)
        ax.set_box_aspect(1)  # square plot area (equal x/y in pixels)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        return ax, cax

    def _new_double_axes(self):
        """Create two data axes with their own right colorbars; both square."""
        ax1 = self.fig.add_axes(self._AX_LEFT)
        ax1.set_box_aspect(1)
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes("right", size="5%", pad=0.08)

        ax2 = self.fig.add_axes(self._AX_RIGHT)
        ax2.set_box_aspect(1)
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes("right", size="5%", pad=0.08)

        return (ax1, cax1), (ax2, cax2)

    def clear_colorbars(self):
        """Remove and forget any existing colorbars before drawing a new plot."""
        if not hasattr(self, "colorbars"):
            self.colorbars = []
            return
        for c in list(self.colorbars):
            try:
                c.remove()
            except Exception:
                pass
        self.colorbars.clear()

    # -------------------- Sensitivity / dose functions (unchanged math) --------------------
    def SensCoherentN12(self, T, mu, nu, tO, phi):
        sin2 = self._sin2(phi)
        sqrt_term = self._sqrt_pos((1.0 - T) * T * mu * nu)
        num1 = (-2.0 * (-1.0 + tO ** 2) * sqrt_term
                + T * (tO ** 2 * mu + nu)
                - (-1.0 + T) * (mu + tO ** 2 * nu))
        num = self._safe_over(num1, sin2)
        den_term = ((-1.0 + T) * T * mu - (-1.0 + T) * T * nu + (1.0 - 2.0 * T) * sqrt_term)
        den = 16.0 * tO ** 2 * den_term ** 2
        val = self._safe_over(num, den)
        return self._finite_pos(val)

    def SensCoNb1(self, T, mu, nu, tO, phi):
        sin2 = self._sin2(phi)
        sTT = (1.0 - T) * T
        sqrt_p = self._sqrt_pos(sTT * mu * nu)
        num1 = ((1.0 - T) ** 2 * mu + T ** 2 * tO ** 2 * mu
                + (1.0 - T) * T * (nu + tO ** 2 * nu)
                + 2.0 * (1.0 - T) * sqrt_p - 2.0 * T * tO ** 2 * sqrt_p
                - 2.0 * tO * (-(sTT) * mu + (sTT) * nu + (1.0 - 2.0 * T) * sqrt_p) * np.cos(phi))
        num = self._safe_over(num1, sin2)
        den = 4.0 * tO ** 2 * ((sTT) * mu - (sTT) * nu + (-1.0 + 2.0 * T) * sqrt_p) ** 2
        val = self._safe_over(num, den)
        return self._finite_pos(val)

    def SensCoNb2(self, T, mu, nu, tO, phi):
        sin2 = self._sin2(phi)
        sqrt_term = self._sqrt_pos((1.0 - T) * T * mu * nu)
        num1 = (-T ** 2 * (1.0 + tO ** 2) * (mu - nu)
                + tO ** 2 * (nu - 2.0 * sqrt_term)
                + T * ((1.0 + tO ** 2) * mu + 2.0 * (sqrt_term + tO ** 2 * (-nu + sqrt_term)))
                - 2.0 * tO * (-sqrt_term + T ** 2 * (-mu + nu) + T * (mu - nu + 2.0 * sqrt_term)) * np.cos(phi))
        num = self._safe_over(num1, sin2)
        den_term = ((-1.0 + T) * T * mu - (-1.0 + T) * T * nu + sqrt_term - 2.0 * T * sqrt_term)
        den = 4.0 * tO ** 2 * den_term ** 2
        val = self._safe_over(num, den)
        return self._finite_pos(val)

    def sN12(self, T, mu, nu, tO, phi):
        sin2 = self._sin2(phi)
        main = (-5 * T * mu + 13 * T ** 2 * mu - 12 * T ** 3 * mu + 4 * T ** 4 * mu
                - 3 * T * tO ** 2 * mu + 18 * T ** 2 * tO ** 2 * mu - 32 * T ** 3 * tO ** 2 * mu + 16 * T ** 4 * tO ** 2 * mu
                + T ** 2 * tO ** 4 * mu - 4 * T ** 3 * tO ** 4 * mu + 4 * T ** 4 * tO ** 4 * mu
                - T * nu + T ** 2 * nu - 4 * T ** 3 * nu + 4 * T ** 4 * nu
                - tO ** 2 * nu - T * tO ** 2 * nu + 18 * T ** 2 * tO ** 2 * nu - 32 * T ** 3 * tO ** 2 * nu + 16 * T ** 4 * tO ** 2 * nu
                + tO ** 4 * nu - 6 * T * tO ** 4 * nu + 13 * T ** 2 * tO ** 4 * nu - 12 * T ** 3 * tO ** 4 * nu + 4 * T ** 4 * tO ** 4 * nu
                - 2 * T * mu * nu + 10 * T ** 2 * mu * nu - 16 * T ** 3 * mu * nu + 8 * T ** 4 * mu * nu
                - 12 * T * tO ** 2 * mu * nu + 44 * T ** 2 * tO ** 2 * mu * nu - 64 * T ** 3 * tO ** 2 * mu * nu + 32 * T ** 4 * tO ** 2 * mu * nu
                - 2 * T * tO ** 4 * mu * nu + 10 * T ** 2 * tO ** 4 * mu * nu - 16 * T ** 3 * tO ** 4 * mu * nu + 8 * T ** 4 * tO ** 4 * mu * nu)
        cos1 = -8.0 * T * (1.0 - 3.0 * T + 2.0 * T ** 2) * tO * (
                (T - tO ** 2 + T * tO ** 2) * nu
                + mu * (-1.0 - (1.0 + tO ** 2) * nu + T * (1.0 + tO ** 2) * (1.0 + 2.0 * nu))
        ) * np.cos(phi)
        cos2 = 8.0 * (-1.0 + T) ** 2 * T ** 2 * tO ** 2 * (mu + nu + 2.0 * mu * nu) * np.cos(2.0 * phi)
        num = self._safe_over(main + cos1 + cos2, sin2)
        den = 16.0 * (-1.0 + T) ** 2 * T ** 2 * tO ** 2 * ((mu - nu) ** 2)
        val = self._safe_over(-num, den)
        return self._finite_pos(val)

    def sN1(self, T, mu, nu, tO, phi):
        sin2 = self._sin2(phi)
        main = (-2 * mu + 5 * T * mu - 4 * T ** 2 * mu + T ** 3 * mu
                + 3 * T * tO ** 2 * mu - 8 * T ** 2 * tO ** 2 * mu + 4 * T ** 3 * tO ** 2 * mu + T ** 3 * tO ** 4 * mu
                - nu + 2 * T * nu - 2 * T ** 2 * nu + T ** 3 * nu
                - tO ** 2 * nu + 5 * T * tO ** 2 * nu - 8 * T ** 2 * tO ** 2 * nu + 4 * T ** 3 * tO ** 2 * nu
                + T * tO ** 4 * nu - 2 * T ** 2 * tO ** 4 * nu + T ** 3 * tO ** 4 * nu
                - 2 * mu * nu + 6 * T * mu * nu - 6 * T ** 2 * mu * nu + 2 * T ** 3 * mu * nu
                - 2 * tO ** 2 * mu * nu + 10 * T * tO ** 2 * mu * nu - 16 * T ** 2 * tO ** 2 * mu * nu + 8 * T ** 3 * tO ** 2 * mu * nu
                - 2 * T ** 2 * tO ** 4 * mu * nu + 2 * T ** 3 * tO ** 4 * mu * nu)
        cos1 = -2.0 * (-1.0 + T) * tO * (
                mu + nu - 2.0 * T * (1.0 + tO ** 2) * nu + 2.0 * T ** 2 * (1.0 + tO ** 2) * nu
                + 2.0 * mu * nu + 2.0 * T ** 2 * (1.0 + tO ** 2) * mu * (1.0 + 2.0 * nu)
                - 2.0 * T * mu * (2.0 + (3.0 + tO ** 2) * nu)
        ) * np.cos(phi)
        cos2 = 2.0 * (-1.0 + T) ** 2 * T * tO ** 2 * (mu + nu + 2.0 * mu * nu) * np.cos(2.0 * phi)
        num = self._safe_over(main + cos1 + cos2, sin2)
        den = 4.0 * (-1.0 + T) ** 2 * T * tO ** 2 * ((mu - nu) ** 2)
        val = self._safe_over(-num, den)
        return self._finite_pos(val)

    def sN2(self, T, mu, nu, tO, phi):
        sin2 = self._sin2(phi)
        main = (T * mu - T ** 2 * mu + T ** 3 * mu
                + T * tO ** 2 * mu - 4 * T ** 2 * tO ** 2 * mu + 4 * T ** 3 * tO ** 2 * mu
                - T ** 2 * tO ** 4 * mu + T ** 3 * tO ** 4 * mu
                + T ** 2 * nu + T ** 3 * nu
                + tO ** 2 * nu - T * tO ** 2 * nu - 4 * T ** 2 * tO ** 2 * nu + 4 * T ** 3 * tO ** 2 * nu
                - tO ** 4 * nu + 3 * T * tO ** 4 * nu - 3 * T ** 2 * tO ** 4 * nu + T ** 3 * tO ** 4 * nu
                + 2 * T ** 3 * mu * nu
                + 2 * T * tO ** 2 * mu * nu - 8 * T ** 2 * tO ** 2 * mu * nu + 8 * T ** 3 * tO ** 2 * mu * nu
                + 2 * T * tO ** 4 * mu * nu - 4 * T ** 2 * tO ** 4 * mu * nu + 2 * T ** 3 * tO ** 4 * mu * nu)
        cos1 = -2.0 * T * tO * (
                (-1.0 + 2.0 * tO ** 2 - 4.0 * T * tO ** 2 + 2.0 * T ** 2 * (1.0 + tO ** 2)) * nu
                + mu * (1.0 + 2.0 * tO ** 2 * nu + 2.0 * T ** 2 * (1.0 + tO ** 2) * (1.0 + 2.0 * nu)
                        - 2.0 * T * (1.0 + tO ** 2 + nu + 3.0 * tO ** 2 * nu))
        ) * np.cos(phi)
        cos2 = 2.0 * (-1.0 + T) * T ** 2 * tO ** 2 * (mu + nu + 2.0 * mu * nu) * np.cos(2.0 * phi)
        num = self._safe_over(main + cos1 + cos2, sin2)
        den = 4.0 * (-1.0 + T) * T ** 2 * tO ** 2 * ((mu - nu) ** 2)
        val = self._safe_over(-num, den)
        return self._finite_pos(val)

    def dose11(self, T, tO, phi):
        T2, T3, T4 = T ** 2, T ** 3, T ** 4
        t2, t4 = tO ** 2, tO ** 4
        num = (
                (-1 + T * (2 + T * (-6 - 4 * (-2 + T) * T)))
                + (1 + T * (4 - 4 * T * (3 - 4 * T + 2 * T2))) * t2
                + T * (2 + T * (-6 - 4 * (-2 + T) * T)) * t4
                + 4 * (-1 + T) * T * tO * np.cos(phi)
                * ((1 - 2 * T) ** 2 * (1 + t2) - 4 * (-1 + T) * T * tO * np.cos(phi))
        )
        den = (
                (1 - t2)
                + T * (2.22045e-16 + 8 * t2 + 8 * T2 * (1 + t2) ** 2 - 4 * T3 * (1 + t2) ** 2 - 4 * T * (
                    1 + 4 * t2 + t4))
                + 4 * (-1 + T) * T * tO * np.cos(phi)
                * ((1 - 2 * T) ** 2 * (1 + t2) - 4 * (-1 + T) * T * tO * np.cos(phi))
        )
        return self._safe_over(num, den)

    def s11(self, T, tO, phi):
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
        cos_term = (4 * (1 - 2 * T) ** 2 * (-1 + T) * T * tO * (1 + tO ** 2) *
                    (-1 + 2 * tO ** 2 - 16 * T * tO ** 2 - 16 * T ** 3 * (1 + 5 * tO ** 2 + tO ** 4) +
                     8 * T ** 4 * (1 + 5 * tO ** 2 + tO ** 4) + 8 * T ** 2 * (1 + 7 * tO ** 2 + tO ** 4)) * cos_phi)
        cos2_term = (8 * (-1 + T) ** 2 * T ** 2 * tO ** 2 *
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
        denominator = (16 * (-1 + T) ** 2 * T ** 2 * tO ** 2 * (
                (1 - 2 * T) ** 2 * (1 + tO ** 2) - 8 * (-1 + T) * T * tO * cos_phi) ** 2)
        return np.abs(numerator / (np.abs(denominator) + 1e-12))

    # -------------------- Compute helpers --------------------
    def _compute_Topt_map_11(self):
        nT = 101
        Ts = np.linspace(1e-6, 0.5 - 1e-6, nT).reshape((-1, 1, 1))
        dose_vals = self.dose11(Ts, self.tO_grid[None, ...], self.phi_grid[None, ...])  # (nT, Ny, Nx)
        k = np.argmax(dose_vals, axis=0)
        self.T_map_opt11 = Ts.reshape(-1)[k]

    def _s_min_of_three(self, T, mu, nu):
        f1 = self.SensCoherentN12(T, mu, nu, self.tO_grid, self.phi_grid)
        f2 = self.SensCoNb1(T, mu, nu, self.tO_grid, self.phi_grid)
        f3 = self.SensCoNb2(T, mu, nu, self.tO_grid, self.phi_grid)
        return np.nanmin(np.stack([f1, f2, f3], axis=-1), axis=-1)

    def _compute_global_smin(self):
        if self.T_map_opt11 is None:
            self._compute_Topt_map_11()

        L = []
        L.append(self._s_min_of_three(1.0, 0, 2))  # |02>, T=1.0
        L.append(self._s_min_of_three(0.5, 2, 0))  # |20>, T=0.5
        L.append(self.s11(0.5, self.tO_grid, self.phi_grid))  # |11>, T=0.5
        L.append(self.s11(self.T_map_opt11, self.tO_grid, self.phi_grid))  # |11>, T*

        stack = np.stack(L, axis=-1)
        self.global_smin_stack = stack

        stack_safe = np.where(np.isfinite(stack), stack, np.inf)
        self.global_smin_data = np.min(stack_safe, axis=-1)
        self.global_smin_source_idx = np.argmin(stack_safe, axis=-1)

        all_inf = ~np.isfinite(self.global_smin_data) | (self.global_smin_data == np.inf)
        self.global_smin_data[all_inf] = np.nan

    # -------------------- Compute / data prep --------------------
    def compute_all_data(self):
        try:
            self.status_label.config(text="Computing data…")
            self.progress['value'] = 0
            self.root.update()

            n = max(20, int(self.resolution.get()))
            tO = np.linspace(0.01, 0.99, n)
            phi = np.linspace(0.01, np.pi - 0.01, n)
            self.tO_grid, self.phi_grid = np.meshgrid(tO, phi)
            self.T_map_opt11 = None

            self.progress['value'] = 10
            self.root.update()

            T = float(self.current_T.get())
            mu, nu = self.mu_nu_options[self.current_mu_nu.get()]

            is_11 = (self.current_mu_nu.get() == '11')
            use_optT = is_11 and (self.T11_mode.get() == 'dose-opt')

            # reset stacks (for co-winner maps)
            self.stack_s = None
            self.stack_c = None

            if use_optT:
                self._compute_Topt_map_11()
                self.T_map = self.T_map_opt11.copy()
                self.sens_data = self.s11(self.T_map, self.tO_grid, self.phi_grid)

                g1 = self.sN12(self.T_map, mu, nu, self.tO_grid, self.phi_grid)
                g2 = self.sN1(self.T_map, mu, nu, self.tO_grid, self.phi_grid)
                g3 = self.sN2(self.T_map, mu, nu, self.tO_grid, self.phi_grid)
                stack_c = np.stack([g1, g2, g3], axis=-1)
                self.stack_c = stack_c

                cm = self.coh_mode.get()
                if cm == 'min':
                    self.coherent_data = np.nanmin(stack_c, axis=-1)
                elif cm == 'CoherentN12':
                    self.coherent_data = g1
                elif cm == 'Nb1':
                    self.coherent_data = g2
                elif cm == 'Nb2':
                    self.coherent_data = g3
                else:
                    self.coherent_data = np.nanmin(stack_c, axis=-1)

                self.s_source_idx = np.zeros_like(self.tO_grid, dtype=int)
                self.s_source_names = ["s11(opt)"]
                self.coh_source_idx = np.nanargmin(stack_c, axis=-1)
                self.coh_source_names = ["sN12", "sN1", "sN2"]

            else:
                self.T_map = None
                f1 = self.SensCoherentN12(T, mu, nu, self.tO_grid, self.phi_grid)
                f2 = self.SensCoNb1(T, mu, nu, self.tO_grid, self.phi_grid)
                f3 = self.SensCoNb2(T, mu, nu, self.tO_grid, self.phi_grid)
                stack_s = np.stack([f1, f2, f3], axis=-1)
                sens_min = np.nanmin(stack_s, axis=-1)

                g1 = self.sN12(T, mu, nu, self.tO_grid, self.phi_grid)
                g2 = self.sN1(T, mu, nu, self.tO_grid, self.phi_grid)
                g3 = self.sN2(T, mu, nu, self.tO_grid, self.phi_grid)
                stack_c = np.stack([g1, g2, g3], axis=-1)
                coh_min = np.nanmin(stack_c, axis=-1)

                self.stack_s = stack_s
                self.stack_c = stack_c

                if is_11:
                    self.sens_data = self.s11(T, self.tO_grid, self.phi_grid)
                    self.s_source_idx = np.zeros_like(self.tO_grid, dtype=int)
                    self.s_source_names = ["s11"]
                else:
                    sm = self.s_mode.get()
                    if sm == 'min':
                        self.sens_data = sens_min
                    elif sm == 'N12':
                        self.sens_data = g1
                    elif sm == 'N1':
                        self.sens_data = g2
                    elif sm == 'N2':
                        self.sens_data = g3
                    else:
                        self.sens_data = sens_min

                    self.s_source_idx = np.nanargmin(stack_s, axis=-1)
                    self.s_source_names = ["SensCoherentN12", "SensCoNb1", "SensCoNb2"]

                cm = self.coh_mode.get()
                if cm == 'min':
                    self.coherent_data = coh_min
                elif cm == 'CoherentN12':
                    self.coherent_data = g1
                elif cm == 'Nb1':
                    self.coherent_data = g2
                elif cm == 'Nb2':
                    self.coherent_data = g3
                else:
                    self.coherent_data = coh_min

                self.coh_source_idx = np.nanargmin(stack_c, axis=-1)
                self.coh_source_names = ["sN12", "sN1", "sN2"]

            self.progress['value'] = 70
            self.root.update()

            self.ratio_data = self._safe_ratio(self.coherent_data, self.sens_data, clip=(1e-12, 1e300))
            self._compute_global_smin()

            self.progress['value'] = 100
            tag = 'T* (dose11 opt)' if (is_11 and self.T11_mode.get() == 'dose-opt') else f'T={T}'
            self.status_label.config(text=f"Ready ({tag}, μν={self.current_mu_nu.get()})")
            self.root.update()

        except Exception as e:
            self.status_label.config(text=f"Error: {e}")

    # -------------------- Plot styling helpers --------------------
    def _robust_range(self, arr, positive_only=False):
        if positive_only:
            valid = np.isfinite(arr) & (arr > 0)
        else:
            valid = np.isfinite(arr)
        if not np.any(valid):
            return (1e-6, 1.0)
        vmin = float(np.nanmin(arr[valid]))
        vmax = float(np.nanmax(arr[valid]))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return (1e-6, 1.0)
        if vmin == vmax:
            eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6
            vmin -= eps;
            vmax += eps
        return vmin, vmax

    def _format_cbar_ticks(self, cbar):
        """
        Apply robust, clean tick locators and formatters to a colorbar.
        This ensures the colorbar represents the relevant data range clearly.
        The layout itself is handled by make_axes_locatable; this function
        only styles the ticks for clarity.
        """
        ax = cbar.ax
        is_log = isinstance(cbar.mappable.norm, LogNorm)

        if is_log:
            # Use LogLocator to find powers of 10. It's robust and auto-scales.
            locator = LogLocator(base=10)
            ax.yaxis.set_major_locator(locator)

            # Use LogFormatter for clear "10^N" style labels.
            formatter = LogFormatter(base=10, labelOnlyBase=False, minor_thresholds=(2, 0.4))
            ax.yaxis.set_major_formatter(formatter)
        else:  # Linear scale
            # MaxNLocator chooses nice, round intervals automatically.
            # prune='both' helps remove ticks at the very edge which can look cluttered.
            locator = MaxNLocator(nbins=7, prune='both')
            ax.yaxis.set_major_locator(locator)

        ax.tick_params(which='both', direction='out')

    def _add_decade_contours(self, ax, Z, pct_low=5, pct_high=95, label=True):
        """Add contours at values 10^n. Uses percentile trimming to ignore outliers."""
        Z = np.asarray(Z)
        valid = np.isfinite(Z) & (Z > 0)
        if not np.any(valid):
            return
        lo = float(np.nanpercentile(Z[valid], pct_low))
        hi = float(np.nanpercentile(Z[valid], pct_high))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= lo:
            return
        e_min = int(np.floor(np.log10(lo)))
        e_max = int(np.ceil(np.log10(hi)))
        levels = 10.0 ** np.arange(e_min, e_max + 1)
        try:
            cs = ax.contour(self.tO_grid, self.phi_grid, Z, levels=levels,
                            colors='w', linewidths=0.9, alpha=0.7)
            if label:
                fmt = {lvl: rf"$10^{{{int(np.log10(lvl))}}}$" for lvl in levels}
                ax.clabel(cs, fmt=fmt, inline=True, fontsize=9)
        except Exception:
            pass

    # -------------------- Core drawing --------------------
    def _draw_pmesh(self, ax, cax, Z, title, log=False, add_ratio1=False, add_decades=False):
        # IMPORTANT: do NOT call tight_layout; we control layout via fixed axes
        if log:
            vmin, vmax = self._robust_range(Z, positive_only=True)
            norm = LogNorm(max(vmin, 1e-12), vmax)
            Zp = np.ma.array(Z, mask=~(np.isfinite(Z) & (Z > 0)))
        else:
            vmin, vmax = self._robust_range(Z, positive_only=False)
            norm = None
            Zp = np.ma.array(Z, mask=~np.isfinite(Z))

        pm = ax.pcolormesh(self.tO_grid, self.phi_grid, Zp, shading='auto',
                           cmap='plasma', vmin=None if log else vmin,
                           vmax=None if log else vmax, norm=norm)

        # Colorbar placed in the pre-made axis so main plot size doesn't change
        cbar = self.fig.colorbar(pm, cax=cax)
        self.colorbars.append(cbar)
        self._format_cbar_ticks(cbar)  # <-- MODIFIED CALL

        if add_ratio1:
            try:
                ax.contour(self.tO_grid, self.phi_grid, Z, levels=[1.0],
                           colors='white', linewidths=1.8)
            except Exception:
                pass

        if add_decades:
            self._add_decade_contours(ax, Z, pct_low=5, pct_high=95, label=True)

        ax.set_xlabel('tO', fontsize=12)
        ax.set_ylabel('φ [rad]', fontsize=12)
        ax.set_title(self.get_plot_title(title), fontsize=14)
        ax.grid(False)

        # min text (for positive values only)
        try:
            finite = Z[np.isfinite(Z) & (Z > 0)]
            if finite.size:
                ax.text(0.02, 0.98, f"Min: {finite.min():.2e}",
                        transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=9, va='top')
        except Exception:
            pass

    def _render_plot(self, plot_type: str):
        """Single entry point for on-screen and for saving — ensures identical layout."""
        self.fig.clear()
        self.clear_colorbars()

        if plot_type == "s_mu_nu":
            ax, cax = self._new_single_axes()
            self._draw_pmesh(ax, cax, self.sens_data, "s_mu_nu",
                             log=True, add_ratio1=False, add_decades=True)

        elif plot_type == "coherent":
            ax, cax = self._new_single_axes()
            self._draw_pmesh(ax, cax, self.coherent_data, "coherent",
                             log=True, add_ratio1=False, add_decades=True)

        elif plot_type == "ratio":
            ax, cax = self._new_single_axes()
            self._draw_pmesh(ax, cax, self.ratio_data, "ratio",
                             log=False, add_ratio1=True, add_decades=False)

        elif plot_type == "ratio_log":
            ax, cax = self._new_single_axes()
            self._draw_pmesh(ax, cax, self.ratio_data, "ratio_log",
                             log=True, add_ratio1=True, add_decades=True)

        elif plot_type == "comparison":
            (ax1, cax1), (ax2, cax2) = self._new_double_axes()
            self._draw_pmesh(ax1, cax1, self.sens_data, "s_mu_nu",
                             log=True, add_ratio1=False, add_decades=True)
            self._draw_pmesh(ax2, cax2, self.coherent_data, "coherent",
                             log=True, add_ratio1=False, add_decades=True)
            # Optional: could add a concise centered title, but avoid suptitle to keep sizes identical

        elif plot_type == "source_s":
            ax, cax = self._new_single_axes()
            # categorical map doesn't need colorbar normalization, but keep cbar for legend-parity
            ax.clear()
            base = plt.get_cmap('tab10')
            colors = [base(i % 10) for i in range(len(self.s_source_names))]
            cmap = ListedColormap(colors)
            ax.imshow(self.s_source_idx, origin='lower',
                      extent=[self.tO_grid.min(), self.tO_grid.max(),
                              self.phi_grid.min(), self.phi_grid.max()],
                      aspect='auto', cmap=cmap, vmin=-0.5, vmax=len(self.s_source_names) - 0.5)
            # borders
            try:
                ax.contour(self.tO_grid, self.phi_grid, self.s_source_idx,
                           levels=np.arange(-0.5, len(self.s_source_names), 1.0),
                           colors='white', linewidths=0.8, alpha=0.6)
            except Exception:
                pass
            ax.set_xlabel('tO', fontsize=12)
            ax.set_ylabel('φ [rad]', fontsize=12)
            ax.set_title(self.get_plot_title("source_s"), fontsize=14)
            ax.set_box_aspect(1)

            # build legend-like colorbar ticks with indices
            fake = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-0.5, vmax=len(self.s_source_names) - 0.5), cmap=cmap)
            cbar = self.fig.colorbar(fake, cax=cax)
            self.colorbars.append(cbar)
            cbar.set_ticks(np.arange(len(self.s_source_names)))
            cbar.set_ticklabels(self.s_source_names)
            cbar.ax.tick_params(which='both', direction='out')

        elif plot_type == "source_coh":
            ax, cax = self._new_single_axes()
            ax.clear()
            base = plt.get_cmap('tab10')
            colors = [base(i % 10) for i in range(len(self.coh_source_names))]
            cmap = ListedColormap(colors)
            ax.imshow(self.coh_source_idx, origin='lower',
                      extent=[self.tO_grid.min(), self.tO_grid.max(),
                              self.phi_grid.min(), self.phi_grid.max()],
                      aspect='auto', cmap=cmap, vmin=-0.5, vmax=len(self.coh_source_names) - 0.5)
            try:
                ax.contour(self.tO_grid, self.phi_grid, self.coh_source_idx,
                           levels=np.arange(-0.5, len(self.coh_source_names), 1.0),
                           colors='white', linewidths=0.8, alpha=0.6)
            except Exception:
                pass
            ax.set_xlabel('tO', fontsize=12)
            ax.set_ylabel('φ [rad]', fontsize=12)
            ax.set_title(self.get_plot_title("source_coh"), fontsize=14)
            ax.set_box_aspect(1)

            fake = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-0.5, vmax=len(self.coh_source_names) - 0.5),
                                         cmap=cmap)
            cbar = self.fig.colorbar(fake, cax=cax)
            self.colorbars.append(cbar)
            cbar.set_ticks(np.arange(len(self.coh_source_names)))
            cbar.set_ticklabels(self.coh_source_names)
            cbar.ax.tick_params(which='both', direction='out')

        elif plot_type == "global_smin":
            ax, cax = self._new_single_axes()
            self._draw_pmesh(ax, cax, self.global_smin_data, "global_smin",
                             log=True, add_ratio1=False, add_decades=True)

        elif plot_type == "source_global_s":
            ax, cax = self._new_single_axes()
            ax.clear()
            names = self.global_smin_source_names
            base = plt.get_cmap('tab10')
            colors = [base(i % 10) for i in range(len(names))]
            cmap = ListedColormap(colors)
            ax.imshow(self.global_smin_source_idx, origin='lower',
                      extent=[self.tO_grid.min(), self.tO_grid.max(),
                              self.phi_grid.min(), self.phi_grid.max()],
                      aspect='auto', cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5)
            try:
                ax.contour(self.tO_grid, self.phi_grid, self.global_smin_source_idx,
                           levels=np.arange(-0.5, len(names), 1.0),
                           colors='white', linewidths=0.8, alpha=0.6)
            except Exception:
                pass
            ax.set_xlabel('tO', fontsize=12)
            ax.set_ylabel('φ [rad]', fontsize=12)
            ax.set_title(self.get_plot_title("source_global_s"), fontsize=14)
            ax.set_box_aspect(1)

            fake = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-0.5, vmax=len(names) - 0.5), cmap=cmap)
            cbar = self.fig.colorbar(fake, cax=cax)
            self.colorbars.append(cbar)
            cbar.set_ticks(np.arange(len(names)))
            cbar.set_ticklabels(names)
            cbar.ax.tick_params(which='both', direction='out')

        # nothing like tight_layout() here — keeps the box identical across plots

    # -------------------- Public update --------------------
    def update_plot(self):
        # Ensure data present
        if self.sens_data is None or self.coherent_data is None or self.ratio_data is None:
            self.compute_all_data()

        self._render_plot(self.plot_var.get())
        self.canvas.draw_idle()

    # -------------------- Probe click handler --------------------
    def on_click(self, event):
        if not self.probe_enabled.get() or event.inaxes is None:
            return
        try:
            t = float(np.clip(event.xdata, self.tO_grid.min(), self.tO_grid.max()))
            p = float(np.clip(event.ydata, self.phi_grid.min(), self.phi_grid.max()))
            j = int(np.argmin(np.abs(self.tO_grid[0, :] - t)))
            i = int(np.argmin(np.abs(self.phi_grid[:, 0] - p)))

            s_min_idx = int(self.s_source_idx[i, j]) if hasattr(self, 's_source_idx') else 0
            c_min_idx = int(self.coh_source_idx[i, j]) if hasattr(self, 'coh_source_idx') else 0

            if self.current_mu_nu.get() == '11' and self.T11_mode.get() == 'dose-opt' and self.T_map is not None:
                s_active_name = 's11(opt)'
                coh_active_name = self.coh_source_names[c_min_idx] if hasattr(self, 'coh_source_names') else 'coh'
            else:
                s_active_name = {
                    'min': self.s_source_names[s_min_idx] if hasattr(self, 's_source_names') else 'min',
                    'N12': 'sN12', 'N1': 'sN1', 'N2': 'sN2',
                }.get(self.s_mode.get(), 'min')
                coh_active_name = {
                    'min': self.coh_source_names[c_min_idx] if hasattr(self, 'coh_source_names') else 'min',
                    'CoherentN12': 'SensCoherentN12', 'Nb1': 'SensCoNb1', 'Nb2': 'SensCoNb2',
                }.get(self.coh_mode.get(), 'min')

            s_val = float(self.sens_data[i, j]) if np.isfinite(self.sens_data[i, j]) else np.nan
            c_val = float(self.coherent_data[i, j]) if np.isfinite(self.coherent_data[i, j]) else np.nan
            t_disp = float(self.tO_grid[0, j])
            p_disp = float(self.phi_grid[i, 0])

            if self.current_mu_nu.get() == '11' and self.T11_mode.get() == 'dose-opt' and self.T_map is not None:
                Tstar = float(self.T_map[i, j])
                txt = (f"tO={t_disp:.4f}, φ={p_disp:.4f}, T*={Tstar:.5f}\n"
                       f"s_active: {s_active_name} = {s_val:.3e}\n"
                       f"coh(min): {coh_active_name} = {c_val:.3e}")
            else:
                s_min_name = self.s_source_names[s_min_idx] if hasattr(self, 's_source_names') else s_active_name
                c_min_name = self.coh_source_names[c_min_idx] if hasattr(self, 'coh_source_names') else coh_active_name
                txt = (f"tO={t_disp:.4f}, φ={p_disp:.4f}\n"
                       f"s_active: {s_active_name} = {s_val:.3e} (min-of-3: {s_min_name})\n"
                       f"coh_active: {coh_active_name} = {c_val:.3e} (min-of-3: {c_min_name})")
            self.probe_info.config(text=txt)

            ax = event.inaxes
            try:
                if self._probe_pt is not None: self._probe_pt.remove()
            except Exception:
                pass
            try:
                if self._probe_ann is not None: self._probe_ann.remove()
            except Exception:
                pass

            self._probe_pt = ax.scatter(t_disp, p_disp, s=60, edgecolors='black',
                                        facecolors='none', linewidths=1.6, zorder=10)
            self._probe_ann = ax.annotate(f"{s_active_name}\n{coh_active_name}",
                                          (t_disp, p_disp), xytext=(42, 28), textcoords='offset points',
                                          bbox=dict(boxstyle='round', fc='white', alpha=0.9),
                                          arrowprops=dict(arrowstyle='->', lw=1.2))
            self.canvas.draw_idle()
        except Exception as e:
            self.status_label.config(text=f"Probe error: {e}")

    # -------------------- Saving --------------------
    def save_plot(self):
        plot_type = self.plot_var.get()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        default_name = f"{plot_type}_{stamp}.png"

        try:
            self.root.update_idletasks()
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after(250, lambda: self.root.attributes('-topmost', False))
        except Exception:
            pass

        filename = ""
        try:
            filename = filedialog.asksaveasfilename(
                parent=self.root,
                defaultextension=".png",
                initialfilename=default_name,
                initialdir=os.path.join(os.path.expanduser("~"), "Desktop"),
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                           ("SVG files", "*.svg"), ("EPS files", "*.eps"), ("All files", "*.*")],
                title="Save Plot As"
            )
        except Exception as e:
            self.status_label.config(text=f"Save dialog unavailable, auto-saving... ({e})")

        if not filename:
            script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            outdir = os.path.join(script_dir, "plots")
            os.makedirs(outdir, exist_ok=True)
            filename = os.path.join(outdir, default_name)
            auto_msg = " (auto-saved to default folder)"
        else:
            auto_msg = ""

        try:
            # Redraw with same stable layout, then save
            self._render_plot(plot_type)
            self.fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            self.status_label.config(text=f"Plot saved: {filename}{auto_msg}")
        except Exception as e:
            self.status_label.config(text=f"Save failed: {e}")

    def _sanitize_filename(self, s: str) -> str:
        s = (s.replace('μ', 'mu').replace('ν', 'nu').replace('φ', 'phi')
             .replace('—', '-').replace('–', '-'))
        s = re.sub(r'\|(\d+)⟩', r'ket\1', s)
        s = s.replace('"', '')
        s = re.sub(r'[<>:/\\|?*]+', '-', s)
        s = re.sub(r'\s+', ' ', s).strip().strip('. ')
        return s

    def save_all_plots_batch(self):
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        out_root = os.path.join(script_dir, "plots", f"batch_{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(out_root, exist_ok=True)

        plot_keys = ("s_mu_nu", "coherent", "ratio_log", "source_s", "source_coh",
                     "global_smin", "source_global_s")
        mu_nu_keys = list(self.mu_nu_options.keys())
        total = len(self.T_values) * len(mu_nu_keys) * len(plot_keys)
        done = 0

        self.progress['value'] = 0
        self.status_label.config(text="Batch export started…")
        self.root.update_idletasks()

        for T in self.T_values:
            self.current_T.set(T)
            for code in mu_nu_keys:
                self.current_mu_nu.set(code)
                self.compute_all_data()

                for plot_key in plot_keys:
                    # Render with the exact same stable layout
                    self._render_plot(plot_key)

                    title = self.get_plot_title(plot_key)
                    fname = self._sanitize_filename(title) + ".pdf"
                    fpath = os.path.join(out_root, fname)

                    self.fig.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

                    done += 1
                    self.progress['value'] = 100 * done / total
                    self.status_label.config(text=f"Saved {done}/{total}: {fname}")
                    self.root.update_idletasks()

        self.status_label.config(text=f"Batch export complete: {out_root}")

    # --- recompute + helpers ---
    def recompute_and_plot(self):
        self.compute_all_data()
        self.update_plot()

    def update_resolution_label(self):
        n = int(self.resolution.get())
        try:
            self.res_label.config(text=f"{n}x{n}")
        except Exception:
            pass

    # -------------------- Titles --------------------
    def get_plot_title(self, plot_key_or_title):
        if self.use_custom_title.get() and self.custom_title.get().strip():
            return self.custom_title.get().strip()
        base_titles = {
            "s_mu_nu": "Sensitivity",
            "coherent": "Coherent Sensitivity",
            "ratio": "Ratio: Coherent / s(μ,ν)",
            "ratio_log": "Sensitivity Ratio Coherent / Quantum",
            "comparison": "Comparison — s(μ,ν) vs SensCoherent",
            "source_s": "Source map — min of {SensCoherentN12, SensCoNb1, SensCoNb2}",
            "source_coh": "Source map — min of {sN12, sN1, sN2}",
            "global_smin": "Global Sensitivity minimum over all configurations and inputs",
            "source_global_s": "Source map — Global Sensitivity minimum",
        }
        base = base_titles.get(plot_key_or_title, str(plot_key_or_title))

        if plot_key_or_title in {"s_mu_nu", "coherent", "ratio", "ratio_log", "comparison", "source_s", "source_coh"}:
            if self.current_mu_nu.get() == '11':
                s_tag = 's11*' if self.T11_mode.get() == 'dose-opt' else 's11'
                T_label = 'T* (dose11 opt)' if self.T11_mode.get() == 'dose-opt' else self._format_T()
            else:
                s_tag = self.s_mode.get()
                T_label = self._format_T()
            return f"{base}  for input: {self._braket_label()}  T={T_label} "
        else:
            return base


def main():
    root = tk.Tk()
    app = QuantumSensingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()