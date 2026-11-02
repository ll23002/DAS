"""
Visualization Module for TSW-RTM and DAS data.
Generates:
- DAS strain-rate waterfalls and preprocessed comparisons
- Virtual shot gathers with apparent velocity lines
- Velocity models and dual-flood configurations
- Migrated RTM fault images overlayed on true velocity models
- F-K spectra per file, cross-correlation and Green's function panels
- CSV export of every image matrix
"""

import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Sequence


def save_matrix_csv(
    path: str,
    matrix: np.ndarray,
    row_axis: np.ndarray,
    col_axis: np.ndarray,
    row_name: str = "row",
    col_name: str = "col"
):
    """
    Saves a 2D image matrix as CSV.
    Row 0: column-axis values (cell [0,0] = "row_name/col_name").
    Column 0: row-axis values. Remaining cells: image amplitude.
    """
    matrix = np.asarray(matrix)
    nr, nc = matrix.shape
    assert len(row_axis) == nr and len(col_axis) == nc, "Axis length mismatch"
    with open(path, "w") as fh:
        fh.write(f"{row_name}/{col_name}," + ",".join(f"{v:.6g}" for v in col_axis) + "\n")
        for i in range(nr):
            fh.write(f"{row_axis[i]:.6g}," + ",".join(f"{v:.6e}" for v in matrix[i]) + "\n")
    print(f"Saved CSV to {path}")


def plot_das_raw_and_preprocessed(
    raw_data: np.ndarray,
    proc_data: np.ndarray,
    time_raw: np.ndarray,
    time_proc: np.ndarray,
    distance_axis: np.ndarray,
    save_path: str = "das_preprocessing_qc.png"
):
    """Plots side-by-side comparison of raw DAS vs preprocessed strain rate (with colorbars)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    vmax_raw = np.percentile(np.abs(raw_data), 98)
    im_raw = axes[0].imshow(
        raw_data,
        aspect='auto',
        cmap='seismic',
        extent=[distance_axis[0]/1000, distance_axis[-1]/1000, time_raw[-1], time_raw[0]],
        vmin=-vmax_raw,
        vmax=vmax_raw
    )
    axes[0].set_title("Raw DAS Strain Rate")
    axes[0].set_xlabel("Distance along fiber (km)")
    axes[0].set_ylabel("Time (s)")
    fig.colorbar(im_raw, ax=axes[0], label="Strain rate (micro-strain/s)")

    vmax_proc = np.percentile(np.abs(proc_data), 98)
    im_proc = axes[1].imshow(
        proc_data,
        aspect='auto',
        cmap='seismic',
        extent=[distance_axis[0]/1000, distance_axis[-1]/1000, time_proc[-1], time_proc[0]],
        vmin=-vmax_proc,
        vmax=vmax_proc
    )
    axes[1].set_title("Preprocessed DAS (Decimated, BP, 1-bit, Whitened)")
    axes[1].set_xlabel("Distance along fiber (km)")
    axes[1].set_ylabel("Time (s)")
    fig.colorbar(im_proc, ax=axes[1], label="Normalized amplitude (a.u.)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure to {save_path}")


def plot_virtual_shot_gather(
    gather: np.ndarray,
    time_axis: np.ndarray,
    offsets_km: np.ndarray,
    title: str = "Virtual Common Shot Gather",
    save_path: str = "virtual_shot_gather.png"
):
    """Plots virtual shot gather reconstructed from ambient noise interferometry (with colorbar)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    vmax = np.percentile(np.abs(gather), 98)
    if vmax == 0:
        vmax = 1.0

    im = ax.imshow(
        gather,
        aspect='auto',
        cmap='seismic',
        extent=[offsets_km[0], offsets_km[-1], time_axis[-1], time_axis[0]],
        vmin=-vmax,
        vmax=vmax
    )
    ax.set_title(title)
    ax.set_xlabel("Offset (km)")
    ax.set_ylabel("Travel Time (s)")
    ax.grid(alpha=0.3)
    fig.colorbar(im, ax=ax, label="Green's function amplitude (a.u.)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure to {save_path}")


def plot_fk_spectra(
    spectra: Sequence[np.ndarray],
    freqs: np.ndarray,
    wavenumbers: np.ndarray,
    titles: Sequence[str],
    save_path: str,
    fmax_plot: float = 5.0,
    vlines: Sequence[float] = (800.0, 3500.0)
):
    """Plots F-K power spectra (dB, common scale) one panel per file with apparent-velocity guide lines."""
    n = len(spectra)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5), squeeze=False)
    axes = axes[0]
    db = [10 * np.log10(s + 1e-30) for s in spectra]
    vmax = max(np.max(d) for d in db)
    vmin = vmax - 40.0
    fsel = (freqs >= 0) & (freqs <= fmax_plot)
    kmax = np.max(np.abs(wavenumbers))
    for ax, d, t in zip(axes, db, titles):
        im = ax.imshow(
            d[fsel], aspect='auto', cmap='magma', origin='lower',
            extent=[wavenumbers[0], wavenumbers[-1], freqs[fsel][0], freqs[fsel][-1]],
            vmin=vmin, vmax=vmax
        )
        kk = np.linspace(-kmax, kmax, 2)
        for v in vlines:
            ax.plot(kk, np.abs(kk) * v, 'c--', lw=0.8)
        ax.set_ylim(0, fmax_plot)
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("Wavenumber (1/m)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="Power (dB, common scale)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure to {save_path}")


def plot_ccf_and_green(
    ccf: np.ndarray,
    green: np.ndarray,
    lags: np.ndarray,
    offsets_km: np.ndarray,
    title: str,
    save_path: str,
    waveform_offset_km: Optional[float] = None
):
    """Cross-correlation, Green's function (-T..+T) and the waveform at one offset."""
    if waveform_offset_km is None:
        waveform_offset_km = float(np.median(offsets_km))
    trace_idx = int(np.argmin(np.abs(offsets_km - waveform_offset_km)))
    trace_offset = float(offsets_km[trace_idx])
    extent = [offsets_km[0], offsets_km[-1], lags[-1], lags[0]]

    fig, axes = plt.subplots(
        1, 3, figsize=(17, 6), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.55]}
    )

    v1 = np.percentile(np.abs(ccf), 98) or 1.0
    im0 = axes[0].imshow(ccf, aspect='auto', cmap='seismic', extent=extent, vmin=-v1, vmax=v1)
    axes[0].set_title("Stacked cross-correlation C(t)")
    axes[0].set_ylabel("Lag (s)")
    fig.colorbar(im0, ax=axes[0], label="Correlation (a.u.)")

    v2 = np.percentile(np.abs(green), 98) or 1.0
    im1 = axes[1].imshow(green, aspect='auto', cmap='seismic', extent=extent, vmin=-v2, vmax=v2)
    axes[1].set_title("Empirical Green's function dC/dt (acausal | causal)")
    axes[1].axvline(trace_offset, color='k', ls='--', lw=0.8)
    fig.colorbar(im1, ax=axes[1], label="Amplitude (a.u.)")

    for ax in axes[:2]:
        ax.axhline(0.0, color='gray', lw=0.6)
        ax.set_xlabel("Offset (km)")

    trace = green[:, trace_idx]
    scale = np.max(np.abs(trace)) or 1.0
    axes[2].plot(trace / scale, lags, color='k', lw=0.9)
    axes[2].fill_betweenx(lags, 0, trace / scale, where=trace > 0, color='tab:red', alpha=0.4)
    axes[2].fill_betweenx(lags, 0, trace / scale, where=trace < 0, color='tab:blue', alpha=0.4)
    axes[2].axhline(0.0, color='gray', lw=0.6)
    axes[2].axvline(0.0, color='gray', lw=0.6)
    axes[2].set_xlim(-1.1, 1.1)
    axes[2].set_title(f"Waveform at offset {trace_offset:.2f} km")
    axes[2].set_xlabel("Normalized amplitude")
    axes[2].grid(alpha=0.3)
    axes[2].text(0.02, 0.02, "causal", transform=axes[2].transAxes, va='bottom', fontsize=9)
    axes[2].text(0.02, 0.98, "acausal", transform=axes[2].transAxes, va='top', fontsize=9)

    axes[0].set_ylim(lags[-1], lags[0])
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure to {save_path}")


def plot_flood_models(
    true_vel: np.ndarray,
    sed_flood: np.ndarray,
    rock_flood: np.ndarray,
    dx: float,
    dz: float,
    fault_x_idx: int,
    save_path: str = "velocity_flood_models.png"
):
    """Plots true velocity model alongside sediment and metamorphic rock flood models."""
    nz, nx = true_vel.shape
    extent = [0, nx * dx / 1000, nz * dz / 1000, 0]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)

    vmin = min(np.min(true_vel), np.min(sed_flood), np.min(rock_flood))
    vmax = max(np.max(true_vel), np.max(sed_flood), np.max(rock_flood))

    im0 = axes[0].imshow(true_vel, aspect='auto', cmap='jet', extent=extent, vmin=vmin, vmax=vmax)
    axes[0].axvline(fault_x_idx * dx / 1000, color='black', linestyle='--', linewidth=2, label='True Fault')
    axes[0].set_title("(a) True Velocity Model")
    axes[0].set_ylabel("Depth (km)")
    axes[0].legend(loc='upper right')

    axes[1].imshow(sed_flood, aspect='auto', cmap='jet', extent=extent, vmin=vmin, vmax=vmax)
    axes[1].set_title("(b) Sediment Flood Model (vs) - Forward Propagation")
    axes[1].set_ylabel("Depth (km)")

    axes[2].imshow(rock_flood, aspect='auto', cmap='jet', extent=extent, vmin=vmin, vmax=vmax)
    axes[2].set_title("(c) Metamorphic Rock Flood Model (vg) - Backward Propagation")
    axes[2].set_xlabel("Distance (km)")
    axes[2].set_ylabel("Depth (km)")

    fig.colorbar(im0, ax=axes.ravel().tolist(), orientation='vertical', label='Velocity (m/s)', fraction=0.03, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure to {save_path}")


def plot_tsw_rtm_image(
    image: np.ndarray,
    true_vel: np.ndarray,
    fault_mask: np.ndarray,
    dx: float,
    dz: float,
    title: str = "TSW-RTM Fault Image",
    save_path: str = "tsw_rtm_image.png"
):
    """
    Plots migrated fault image overlayed on the reference velocity model,
    replicating the visualization style of Li et al. (2020) Figures 3, 4, 5.
    """
    nz, nx = image.shape
    extent = [0, nx * dx / 1000, nz * dz / 1000, 0]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Background: true velocity model in color
    im_bg = ax.imshow(true_vel, aspect='auto', cmap='viridis', extent=extent, alpha=0.45)

    # Foreground: migrated image in grayscale with transparency
    vmax = np.percentile(np.abs(image), 99.5)
    if vmax == 0:
        vmax = 1.0

    im_fg = ax.imshow(
        image,
        aspect='auto',
        cmap='gray',
        extent=extent,
        vmin=-vmax,
        vmax=vmax,
        alpha=0.75
    )

    # Overlay true fault position
    fault_z, fault_x = np.where(fault_mask)
    if len(fault_x) > 0:
        ax.scatter(fault_x * dx / 1000, fault_z * dz / 1000, color='red', s=3, label='True Fault Plane', alpha=0.6)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Distance (km)", fontsize=11)
    ax.set_ylabel("Depth (km)", fontsize=11)
    ax.legend(loc='upper right')

    cbar = fig.colorbar(im_bg, ax=ax, orientation='horizontal', fraction=0.05, pad=0.18)
    cbar.set_label('Background Velocity (m/s)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure to {save_path}")
