#!/usr/bin/env python
"""
Plot 1-D summaries of a SurfATT inversion.

1. Horizontally averaged 1-D Vs profile (depth vs Vs) of the initial and final
   models, with the lateral standard deviation at each depth.
2. For each src_rec file: number of data at each period, and a period-depth
   map of the 1-D sensitivity kernels computed on the averaged initial model.

Kernels are computed with bin/SURFATT_kernel1d, which uses the same surfker
engine as SURFATT_tomo (same layering, Earth flattening, fundamental mode and
empirical Vp/rho relations). Build it with:
    cd build && cmake .. && make -j SURFATT_kernel1d

Place this script in the project folder (the parent of OUTPUT_FILES), edit the
parameters below and run:
    python plot_1d_model_and_data.py

Requirements: numpy, pandas, h5py, matplotlib
"""

import os
import shutil
import subprocess
import tempfile

import h5py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# User parameters (paths are relative to the folder containing this script)
# =============================================================================
OUTPUT_DIR = "OUTPUT_FILES"
INITIAL_MODEL = os.path.join(OUTPUT_DIR, "initial_model.h5")
FINAL_MODEL = os.path.join(OUTPUT_DIR, "final_model.h5")

# Dataset to average (e.g. "vs", "vsv", "vsh"); kernels use the initial profile
MODEL_KEY = "vs"

# Data files used in the inversion. Keys are <RL|LV>_<PH|GR>; set a value to
# None (or remove the entry) to skip that data type.
SRC_REC_FILES = {
    "RL_PH": os.path.join(OUTPUT_DIR, "src_rec_file_forward_RL_PH.csv"),
    "RL_GR": None,
    "LV_PH": None,
    "LV_GR": None,
}

# Horizontal region for averaging: [lon_min, lon_max, lat_min, lat_max].
# None uses all grid points (including the margin grid).
REGION = None

# Path to the kernel tool; if not found, SURFATT_kernel1d is searched in PATH
KERNEL_BIN = "../../bin/SURFATT_kernel1d"
# Launcher prefix, e.g. ["mpirun", "-np", "1"] if the bare MPI binary fails
MPI_LAUNCHER = []

# "total": Vs kernel as used in the vs-only inversion (Rayleigh: Vp and rho
#          scaled from Vs; Love: same as "vs", as in SURFATT_tomo)
# "vs":    partial d(vel)/d(Vs) with Vp and rho fixed
KERNEL_TYPE = "total"
# Normalize each period's kernel by its maximum absolute value
NORMALIZE_KERNEL = True

# Figure output
FIG_DIR = "figures"
FIG_FORMAT = "png"
DPI = 300

# =============================================================================
# Plot style
# =============================================================================
INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text, axes
GRID = "#e2e1dc"         # recessive grid
MODEL_STYLE = {          # line color and style for each model
    "Initial": {"color": "#eb6834", "ls": "--", "zorder": 3},
    "Final": {"color": "#2a78d6", "ls": "-", "zorder": 2},
}
BAR_COLOR = "#2a78d6"
# Sequential one-hue ramp for non-negative kernels
KERNEL_CMAP = LinearSegmentedColormap.from_list(
    "kernel_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
# Diverging ramp (red - gray - blue) used when kernels have negative lobes
KERNEL_CMAP_DIV = LinearSegmentedColormap.from_list(
    "kernel_div",
    ["#9e2322", "#e34948", "#f4a8a0", "#f0efec", "#9ec5f4", "#3987e5", "#104281"],
)

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve(path):
    """Return an absolute path, interpreting relative paths from BASE_DIR."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


# =============================================================================
# Data loading
# =============================================================================
def load_model(path, key):
    """Read lon (x), lat (y), depth (z) and a model field of shape (nx, ny, nz)."""
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"Dataset '{key}' not found in {path}; available: {list(f.keys())}")
        x, y, z = f["x"][:], f["y"][:], f["z"][:]
        field = f[key][:]
    if field.shape != (len(x), len(y), len(z)):
        raise ValueError(f"{path}: {key} shape {field.shape} != (nx, ny, nz) = "
                         f"{(len(x), len(y), len(z))}")
    return x, y, z, field


def horizontal_stats(x, y, field, region=None):
    """Mean and standard deviation over horizontal grid points at each depth."""
    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
        ix = (x >= lon_min) & (x <= lon_max)
        iy = (y >= lat_min) & (y <= lat_max)
        if not ix.any() or not iy.any():
            raise ValueError(f"REGION {region} contains no grid points")
        field = field[np.ix_(ix, iy)]
    return field.mean(axis=(0, 1)), field.std(axis=(0, 1))


def load_src_rec(path):
    """Number of data and mean observed velocity at each period."""
    df = pd.read_csv(path)
    grp = df.groupby("period")
    return pd.DataFrame({"count": grp.size(), "mean_vel": grp["vel"].mean()})


# =============================================================================
# 1-D kernels via SURFATT_kernel1d
# =============================================================================
def find_kernel_bin():
    """Locate the SURFATT_kernel1d executable."""
    path = resolve(KERNEL_BIN)
    if os.path.isfile(path):
        return path
    found = shutil.which(os.path.basename(KERNEL_BIN))
    if found:
        return found
    raise FileNotFoundError(
        f"SURFATT_kernel1d not found at {path} or in PATH. Build it with:\n"
        "    cd build && cmake .. && make -j SURFATT_kernel1d")


def compute_kernel_1d(z, vs, periods, data_type):
    """Dispersion and depth kernels of a 1-D model for one data type (e.g. RL_PH)."""
    wave, vtype = data_type.split("_")
    with tempfile.TemporaryDirectory() as tmp:
        fin = os.path.join(tmp, "model1d.h5")
        fout = os.path.join(tmp, "kernel1d.h5")
        with h5py.File(fin, "w") as f:
            f["z"] = np.asarray(z, dtype=float)
            f["vs"] = np.asarray(vs, dtype=float)
            f["periods"] = np.asarray(periods, dtype=float)
        cmd = MPI_LAUNCHER + [find_kernel_bin(), "-i", fin, "-o", fout, "-w", wave, "-t", vtype]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not os.path.isfile(fout):
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stdout}\n{res.stderr}")
        with h5py.File(fout, "r") as f:
            ker = {k: f[k][:] for k in f.keys()}
    bad = [k for k, v in ker.items() if not np.isfinite(v).all()]
    if bad:
        raise ValueError(f"{data_type}: NaN/Inf in {bad} from SURFATT_kernel1d")
    return ker


# =============================================================================
# Plotting
# =============================================================================
def plot_average_model(z, stats, fname):
    """Mean +/- std profiles and std profiles of the averaged models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True,
                                   gridspec_kw={"width_ratios": [2, 1]})
    for label, (mean, std) in stats.items():
        st = MODEL_STYLE[label]
        ax1.fill_betweenx(z, mean - std, mean + std, color=st["color"], alpha=0.18, lw=0)
        ax1.plot(mean, z, color=st["color"], ls=st["ls"], lw=1.8, zorder=st["zorder"],
                 label=f"{label} mean ± 1σ")
        ax2.plot(std, z, color=st["color"], ls=st["ls"], lw=1.8, zorder=st["zorder"], label=label)

    ax1.set_xlabel(f"{MODEL_KEY} (km/s)")
    ax1.set_ylabel("Depth (km)")
    ax1.set_title("(a) Average 1-D model", loc="left")
    ax1.legend(loc="lower left")
    ax2.set_xlabel(f"Std of {MODEL_KEY} (km/s)")
    ax2.set_title("(b) Lateral std", loc="left")
    ax1.set_ylim(z.max(), z.min())

    region = "all grid points" if REGION is None else f"region {REGION}"
    fig.suptitle(f"Horizontally averaged model ({region})", color=INK)
    fig.tight_layout()
    fig.savefig(fname, dpi=DPI)
    plt.close(fig)
    print(f"Saved {fname}")


def period_edges(periods):
    """Cell edges halfway between neighbouring periods."""
    if len(periods) == 1:
        return np.array([periods[0] - 0.5, periods[0] + 0.5])
    mid = 0.5 * (periods[1:] + periods[:-1])
    return np.concatenate(([2 * periods[0] - mid[0]], mid, [2 * periods[-1] - mid[-1]]))


def plot_data_and_kernels(data_type, table, z, ker, fname):
    """Data count per period (top) and period-depth kernel map (bottom) sharing the period axis."""
    periods = table.index.values
    edges = period_edges(periods)
    key = "sen_vs_total" if KERNEL_TYPE == "total" else "sen_vs"
    vel_sym = "c" if data_type.endswith("PH") else "U"

    # kernel matrix of shape (nz, nper)
    kmat = ker[key].T
    if NORMALIZE_KERNEL:
        kmat = kmat / np.maximum(np.abs(kmat).max(axis=0), 1e-30)

    fig = plt.figure(figsize=(7.5, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.6], width_ratios=[1, 0.03],
                          hspace=0.2, wspace=0.03)
    ax_cnt = fig.add_subplot(gs[0, 0])
    ax_ker = fig.add_subplot(gs[1, 0], sharex=ax_cnt)
    cax = fig.add_subplot(gs[1, 1])

    # (a) number of data per period
    ax_cnt.bar(periods, table["count"].values, width=0.8 * np.diff(edges), color=BAR_COLOR)
    ax_cnt.set_ylabel("Number of data")
    ax_cnt.set_title(f"(a) Data count (total {int(table['count'].sum())})", loc="left")
    ax_cnt.grid(axis="x", visible=False)
    ax_cnt.tick_params(labelbottom=False)

    # (b) kernel map; diverging colors centred at zero if kernels have negative lobes
    kmin, kmax = float(kmat.min()), float(kmat.max())
    if kmin < -0.01 * max(abs(kmax), 1e-30):
        vabs = max(abs(kmin), abs(kmax))
        cmap, vmin, vmax = KERNEL_CMAP_DIV, -vabs, vabs
    else:
        cmap, vmin, vmax = KERNEL_CMAP, 0.0, kmax
    pcm = ax_ker.pcolormesh(periods, z, kmat, cmap=cmap, vmin=vmin, vmax=vmax,
                            shading="nearest", rasterized=True)
    ax_ker.set_xlim(edges[0], edges[-1])
    ax_ker.set_ylim(z.max(), z.min())
    ax_ker.grid(False)
    ax_ker.set_xlabel("Period (s)")
    ax_ker.set_ylabel("Depth (km)")
    ax_ker.set_title("(b) 1-D sensitivity, initial model", loc="left")

    prefix = "Normalized " if NORMALIZE_KERNEL else ""
    cbar = fig.colorbar(pcm, cax=cax)
    scaled = KERNEL_TYPE == "total" and data_type.startswith("RL")
    cbar.set_label(f"{prefix}∂{vel_sym}/∂{MODEL_KEY}" + (" (Vp, ρ scaled)" if scaled else ""))
    cbar.outline.set_visible(False)
    cax.grid(False)

    fig.suptitle(f"{data_type}: data count and 1-D sensitivity", color=INK, y=0.95)
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# =============================================================================
# Main
# =============================================================================
def main():
    fig_dir = resolve(FIG_DIR)
    os.makedirs(fig_dir, exist_ok=True)

    # ---- averaged 1-D models ----
    profiles, stats, z = {}, {}, None
    for label, path in (("Initial", INITIAL_MODEL), ("Final", FINAL_MODEL)):
        path = resolve(path)
        if not os.path.isfile(path):
            print(f"Warning: {path} not found; {label.lower()} model skipped")
            continue
        x, y, zz, field = load_model(path, MODEL_KEY)
        if z is not None and not np.allclose(z, zz):
            raise ValueError("Initial and final models have different depth grids")
        z = zz
        stats[label] = horizontal_stats(x, y, field, REGION)
        profiles[label] = stats[label][0]
    if not stats:
        raise FileNotFoundError("Neither the initial nor the final model was found")

    plot_average_model(z, stats, os.path.join(fig_dir, f"model_1d_average.{FIG_FORMAT}"))

    # ---- data count and kernels (on the averaged initial model) ----
    if "Initial" not in profiles:
        print("Warning: initial model not found; data count and sensitivity figures skipped")
        return
    for data_type, path in SRC_REC_FILES.items():
        if path is None:
            continue
        path = resolve(path)
        if not os.path.isfile(path):
            print(f"Warning: {path} not found; {data_type} skipped")
            continue
        table = load_src_rec(path)
        ker = compute_kernel_1d(z, profiles["Initial"], table.index.values, data_type)
        table["vel_initial"] = ker["vel"]

        print(f"\n{data_type} ({os.path.relpath(path, BASE_DIR)}):")
        print("  mean_vel: mean observed velocity; vel_initial: predicted on the averaged initial model")
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))

        plot_data_and_kernels(data_type, table, z, ker,
                              os.path.join(fig_dir, f"data_sensitivity_{data_type}.{FIG_FORMAT}"))

if __name__ == "__main__":
    main()
