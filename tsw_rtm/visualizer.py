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
    """Plots F-K power spectra (dB, common scale) in a clean 3-column grid."""
    n = len(spectra)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False)
    db = [10 * np.log10(s + 1e-30) for s in spectra]
    vmax = max(np.max(d) for d in db)
    vmin = vmax - 40.0
    fsel = (freqs >= 0) & (freqs <= fmax_plot)
    kmax = np.max(np.abs(wavenumbers))

    for idx, (d, t) in enumerate(zip(db, titles)):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        im = ax.imshow(
            d[fsel], aspect='auto', cmap='magma', origin='lower',
            extent=[wavenumbers[0], wavenumbers[-1], freqs[fsel][0], freqs[fsel][-1]],
            vmin=vmin, vmax=vmax
        )
        kk = np.linspace(-kmax, kmax, 2)
        for v in vlines:
            ax.plot(kk, np.abs(kk) * v, 'c--', lw=0.8)
        ax.set_ylim(0, fmax_plot)
        ax.set_title(t, fontsize=8)
        ax.set_xlabel("Wavenumber (1/m)", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)
        fig.colorbar(im, ax=ax, label="Power (dB)")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure to {save_path}")


def plot_fk_spectra_batches(
    spectra: Sequence[np.ndarray],
    freqs: np.ndarray,
    wavenumbers: np.ndarray,
    titles: Sequence[str],
    save_prefix: str,
    fmax_plot: float = 5.0,
    vlines: Sequence[float] = (800.0, 3500.0),
    batch_size: int = 3
) -> List[str]:
    """
    Divides F-K spectra into separate image files of batch_size (default 3) panels each,
    so they can be embedded into documents with high legibility.
    """
    n = len(spectra)
    db = [10 * np.log10(s + 1e-30) for s in spectra]
    vmax = max(np.max(d) for d in db)
    vmin = vmax - 40.0
    fsel = (freqs >= 0) & (freqs <= fmax_plot)
    kmax = np.max(np.abs(wavenumbers))

    saved_paths = []
    num_batches = int(np.ceil(n / batch_size))

    for b in range(num_batches):
        idx_start = b * batch_size
        idx_end = min(idx_start + batch_size, n)
        b_spectra = db[idx_start:idx_end]
        b_titles = titles[idx_start:idx_end]
        n_sub = len(b_spectra)

        fig, axes = plt.subplots(1, n_sub, figsize=(5.0 * n_sub, 4.3), squeeze=False)
        axes = axes[0]
        for ax, d, t in zip(axes, b_spectra, b_titles):
            im = ax.imshow(
                d[fsel], aspect='auto', cmap='magma', origin='lower',
                extent=[wavenumbers[0], wavenumbers[-1], freqs[fsel][0], freqs[fsel][-1]],
                vmin=vmin, vmax=vmax
            )
            kk = np.linspace(-kmax, kmax, 2)
            for v in vlines:
                ax.plot(kk, np.abs(kk) * v, 'c--', lw=0.8)
            ax.set_ylim(0, fmax_plot)
            ax.set_title(t, fontsize=8)
            ax.set_xlabel("Wavenumber (1/m)", fontsize=9)
            ax.set_ylabel("Frequency (Hz)", fontsize=9)
            fig.colorbar(im, ax=ax, label="Power (dB)")

        plt.tight_layout()
        batch_path = f"{save_prefix}_batch{b + 1}.png"
        plt.savefig(batch_path, dpi=200)
        plt.close()
        saved_paths.append(batch_path)
        print(f"Saved figure to {batch_path}")

    return saved_paths


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
    save_path: str = "velocity_flood_models.png",
    fault_label: str = "True Fault"
):
    """Plots true velocity model alongside sediment and metamorphic rock flood models."""
    nz, nx = true_vel.shape
    extent = [0, nx * dx / 1000, nz * dz / 1000, 0]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)

    vmin = min(np.min(true_vel), np.min(sed_flood), np.min(rock_flood))
    vmax = max(np.max(true_vel), np.max(sed_flood), np.max(rock_flood))

    im0 = axes[0].imshow(true_vel, aspect='auto', cmap='jet', extent=extent, vmin=vmin, vmax=vmax)
    axes[0].axvline(fault_x_idx * dx / 1000, color='black', linestyle='--', linewidth=2, label=fault_label)
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

    # Position colorbar explicitly on the right without overlapping any plot
    fig.subplots_adjust(left=0.10, right=0.85, top=0.94, bottom=0.08, hspace=0.32)
    cbar_ax = fig.add_axes([0.88, 0.12, 0.025, 0.78])
    fig.colorbar(im0, cax=cbar_ax, label='Velocity (m/s)')

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
    save_path: str = "tsw_rtm_image.png",
    fault_label: str = "True Fault Plane"
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
        ax.scatter(fault_x * dx / 1000, fault_z * dz / 1000, color='red', s=3, label=fault_label, alpha=0.6)

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


def plot_fk_dispersion_image_c(
    freqs: np.ndarray,
    wavenumbers_km: np.ndarray,
    power_kf: np.ndarray,
    save_path: str = "2_1_fk_dispersion_c.png"
):
    """
    Plots Wavenumber (1/km) on Y-axis vs Frequency (Hz) on X-axis,
    with reference velocity lines (1 km/s and 4 km/s), replicating Figure c) from Numeral 2.1.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Log power for visualization
    db = 10.0 * np.log10(power_kf + 1e-30)
    vmax = np.percentile(db, 99.5)
    vmin = vmax - 35.0

    extent = [freqs[0], freqs[-1], wavenumbers_km[0], wavenumbers_km[-1]]
    im = ax.imshow(
        db,
        aspect='auto',
        origin='lower',
        cmap='hot_r',
        extent=extent,
        vmin=vmin,
        vmax=vmax
    )

    # Reference velocity slope lines: k = f / v (with v in km/s, f in Hz, k in 1/km)
    f_ref = np.linspace(freqs[0], freqs[-1], 100)
    k_1kms = f_ref / 1.0
    k_4kms = f_ref / 4.0

    ax.plot(f_ref, k_1kms, 'k--', lw=1.2, label="1 km/s")
    ax.plot(f_ref, k_4kms, 'k--', lw=1.2, label="4 km/s")

    # Place text labels along slopes like figure c)
    if freqs[-1] >= 20:
        idx_label1 = int(len(f_ref) * 0.35)
        if k_1kms[idx_label1] <= wavenumbers_km[-1]:
            ax.text(f_ref[idx_label1] - 4, k_1kms[idx_label1], "1 km/s", rotation=60, fontsize=10, fontweight='bold')
        idx_label4 = int(len(f_ref) * 0.75)
        if k_4kms[idx_label4] <= wavenumbers_km[-1]:
            ax.text(f_ref[idx_label4] + 1, k_4kms[idx_label4], "4 km/s", rotation=25, fontsize=10, fontweight='bold')

    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_ylim(wavenumbers_km[0], wavenumbers_km[-1])
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Wavenumber (1/km)", fontsize=11)
    ax.set_title("c) F-K Dispersion Spectrum (Energy ridge)", fontsize=12, fontweight='bold', loc='left')
    fig.colorbar(im, ax=ax, label="Relative Power (dB)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure to {save_path}")


def plot_fv_and_1d_profile(
    freqs: np.ndarray,
    velocities_kms: np.ndarray,
    power_fv: np.ndarray,
    depth_m: np.ndarray,
    vs_kms: np.ndarray,
    save_path: str = "2_2_fv_and_1d_profile.png"
):
    """
    Plots 2-panel figure:
    (a) Phase Velocity (km/s) vs Frequency (Hz) (Normal modes and Leaky modes)
    (b) 1D Shear-wave velocity (Vs) vs depth (m).
    Replicates Numeral 2.2 Figure b).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.4, 0.9]})

    # Panel a: f-v spectrum
    norm_fv = power_fv / (np.max(power_fv, axis=0, keepdims=True) + 1e-12)
    extent = [freqs[0], freqs[-1], velocities_kms[0], velocities_kms[-1]]
    im = axes[0].imshow(
        norm_fv,
        aspect='auto',
        origin='lower',
        cmap='hot_r',
        extent=extent,
        vmin=0.1,
        vmax=1.0
    )
    axes[0].set_xlabel("Frequency (Hz)", fontsize=11)
    axes[0].set_ylabel("Phase Velocity (km/s)", fontsize=11)
    axes[0].set_title("Espectro F-V (Velocidad de fase)", fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=axes[0], label="Normalized energy")

    # Annotations like paper
    if len(freqs) > 20:
        f_mid = float(freqs[len(freqs) // 2])
        axes[0].text(f_mid, velocities_kms[-1] * 0.85, "Leaky modes", color='navy', fontweight='bold',
                     ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        axes[0].text(f_mid, velocities_kms[0] + 0.3, "Normal modes", color='darkred', fontweight='bold',
                     ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # Panel b: 1D Vs profile in depth
    if len(depth_m) > 0 and len(vs_kms) > 0:
        axes[1].step(vs_kms * 1000.0, depth_m, where='post', color='crimson', lw=2, label="Vs estimado (1D)")
        axes[1].fill_betweenx(depth_m, 0, vs_kms * 1000.0, step='post', color='crimson', alpha=0.15)
        axes[1].set_xlim(min(200.0, np.min(vs_kms * 1000.0) * 0.9), np.max(vs_kms * 1000.0) * 1.15)
        axes[1].set_ylim(max(depth_m), 0.0)  # Inverted depth axis (0 at top)
        axes[1].set_xlabel("Vs (m/s)", fontsize=11)
        axes[1].set_ylabel("Profundidad z (m)", fontsize=11)
        axes[1].set_title("b) Modelo 1D en profundidad", fontsize=11, fontweight='bold')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend(loc='lower left')
    else:
        axes[1].text(0.5, 0.5, "Datos de profundidad insuficientes", ha='center', va='center')

    fig.suptitle("2.2 Espectros frecuencia-velocidad y modelo 1D en profundidad", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure to {save_path}")


def plot_fs_spectrum(
    freqs: np.ndarray,
    slowness_skm: np.ndarray,
    power_fs: np.ndarray,
    save_path: str = "2_3_fs_spectrum.png"
):
    """
    Plots Phase Slowness (s/km) on Y-axis vs Frequency (Hz) on X-axis,
    with reference slowness dashed lines. Replicates Numeral 2.3.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    norm_fs = power_fs / (np.max(power_fs, axis=0, keepdims=True) + 1e-12)
    extent = [freqs[0], freqs[-1], slowness_skm[0], slowness_skm[-1]]
    im = ax.imshow(
        norm_fs,
        aspect='auto',
        origin='lower',
        cmap='hot_r',
        extent=extent,
        vmin=0.1,
        vmax=1.0
    )

    # Reference slowness guide lines (dashed lines as in Figure 1 of page 3)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.axhline(1.5, color='black', linestyle='--', linewidth=1.0, alpha=0.8)

    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_ylim(slowness_skm[0], slowness_skm[-1])
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Phase Slowness (s/km)", fontsize=11)
    ax.set_title("2.3 Espectros f-s (Lentitud de fase vs frecuencia)", fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, label="Normalized energy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure to {save_path}")
