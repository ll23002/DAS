import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, gaussian_filter1d


def compute_kf_grid(
    power_fk: np.ndarray,
    freqs: np.ndarray,
    wavenumbers: np.ndarray,
    fmax: float = 100.0,
    kmax_km: float = 25.0
):
    """
    Extracts positive wavenumbers k (1/km) and frequencies f (Hz) for dispersion analysis.
    Returns (P_kf, f_sel, k_sel_km) where P_kf has shape (len(k_sel_km), len(f_sel)).
    """
    f_mask = (freqs >= 0.0) & (freqs <= fmax)
    f_sel = freqs[f_mask]

    # Fold or select positive wavenumbers: k_km = k_m * 1000
    pos_k = wavenumbers >= 0.0
    k_pos_m = wavenumbers[pos_k]
    k_pos_km = k_pos_m * 1000.0

    k_mask = k_pos_km <= kmax_km
    k_sel_km = k_pos_km[k_mask]

    # Symmetric sum or positive branch
    p_subset = power_fk[f_mask, :][:, pos_k][:, k_mask]
    # Transpose so rows are wavenumber (Y-axis) and columns are frequency (X-axis)
    p_kf = p_subset.T.astype(np.float32)

    return p_kf, f_sel, k_sel_km


def compute_fv_spectrum(
    power_fk: np.ndarray,
    freqs: np.ndarray,
    wavenumbers: np.ndarray,
    fmin: float = 1.0,
    fmax: float = 95.0,
    vmin_kms: float = 0.15,
    vmax_kms: float = 4.0,
    nv: int = 220
):
    """
    Transforms (f, k) power spectrum into frequency-phase velocity (f, v) spectrum.
    v = f / k  =>  k = f / v
    Returns:
        p_fv: 2D array of shape (nv, len(f_sel)) with rows=velocity, cols=frequency
        f_sel: 1D array of frequencies (Hz)
        v_kms: 1D array of phase velocities (km/s)
    """
    f_mask = (freqs >= fmin) & (freqs <= fmax)
    f_sel = freqs[f_mask]
    v_kms = np.linspace(vmin_kms, vmax_kms, nv, dtype=np.float32)

    pos_k = wavenumbers > 0.0
    k_pos_m = wavenumbers[pos_k]
    k_pos_km = k_pos_m * 1000.0

    p_pos = power_fk[f_mask, :][:, pos_k]
    p_fv = np.zeros((nv, len(f_sel)), dtype=np.float32)

    for i, f_val in enumerate(f_sel):
        if f_val <= 0:
            continue
        # Interpolate power across k
        f_interp = interp1d(k_pos_km, p_pos[i], bounds_error=False, fill_value=0.0)
        # target k for each v: k_km = f / v_kms
        target_k = f_val / v_kms
        p_fv[:, i] = f_interp(target_k)

    return p_fv, f_sel, v_kms


def extract_dispersion_and_1d_profile(
    p_fv: np.ndarray,
    freqs: np.ndarray,
    v_kms: np.ndarray,
    depth_max_m: float = 1000.0
):
    """
    Extracts fundamental dispersion curve v_phase(f) from peak energy
    (preferring the slowest significant mode when multiple modes coexist)
    and computes an approximate, regularized 1D shear-wave velocity profile Vs(z).
    z approx lambda / 2 = v / (2 * f)
    Vs approx v_phase / 0.92
    """
    col_max = np.max(p_fv, axis=0)
    global_max = float(np.max(p_fv))
    valid_energy = col_max > 0.03 * global_max

    raw_v_curve_kms = np.full(len(freqs), np.nan, dtype=np.float32)
    for i in range(len(freqs)):
        if not valid_energy[i] or col_max[i] <= 0:
            continue
        col = p_fv[:, i]
        # Prefer the lowest-velocity significant local peak (fundamental mode)
        # rather than jumping to high-velocity leaky/body modes near k -> 0.
        thresh = 0.35 * col_max[i]
        is_peak = np.zeros_like(col, dtype=bool)
        is_peak[1:-1] = (col[1:-1] >= col[:-2]) & (col[1:-1] >= col[2:]) & (col[1:-1] >= thresh)
        peak_idxs = np.where(is_peak)[0]
        idx = int(peak_idxs[0]) if len(peak_idxs) > 0 else int(np.argmax(col))
        raw_v_curve_kms[i] = v_kms[idx]

    # Interpolate across low-energy frequency gaps before smoothing
    good = np.isfinite(raw_v_curve_kms)
    if np.sum(good) >= 2:
        raw_v_curve_kms = np.interp(np.arange(len(freqs)), np.where(good)[0], raw_v_curve_kms[good])
    else:
        raw_v_curve_kms = v_kms[np.argmax(p_fv, axis=0)]

    # Smooth phase velocity curve
    v_curve_kms = median_filter(raw_v_curve_kms, size=7)
    v_curve_kms = gaussian_filter1d(v_curve_kms, sigma=2.0)

    # Approximate depth sensitivity: z = lambda / 2 = (v * 1000) / (2 * f)
    wavelength_m = (v_curve_kms * 1000.0) / np.maximum(freqs, 0.1)
    depth_m_raw = wavelength_m / 2.0
    vs_kms_raw = v_curve_kms / 0.92

    # Filter valid depths
    valid = (depth_m_raw > 0) & (depth_m_raw <= depth_max_m) & (vs_kms_raw >= 0.1) & (vs_kms_raw <= 4.5)
    d_valid = depth_m_raw[valid]
    vs_valid = vs_kms_raw[valid]

    # Create a regular, monotonic depth grid from 0 to depth_max_m
    z_grid = np.linspace(0.0, depth_max_m, 200)
    
    if len(d_valid) > 5:
        # Sort by depth
        sort_idx = np.argsort(d_valid)
        d_sorted = d_valid[sort_idx]
        vs_sorted = vs_valid[sort_idx]
        
        # Linear interpolation with boundary clamping
        f_vs = interp1d(d_sorted, vs_sorted, bounds_error=False,
                        fill_value=(vs_sorted[0], vs_sorted[-1]))
        vs_interp = f_vs(z_grid)
        # Apply smoothing to represent realistic geological compaction with depth
        vs_smooth = gaussian_filter1d(vs_interp, sigma=4.0)
        # Enforce realistic monotonicity (velocity increases or stays constant with depth)
        vs_kms_profile = np.maximum.accumulate(vs_smooth)
    else:
        # Fallback default sedimentary-to-rock profile if picking fails
        vs_kms_profile = 0.35 + 0.0018 * z_grid + 0.8 * (1.0 - np.exp(-z_grid / 150.0))

    return v_curve_kms, z_grid, vs_kms_profile


def compute_vs30(depth_m: np.ndarray, vs_ms: np.ndarray) -> tuple:
    """
    Computes Vs30 (average shear wave velocity in top 30 m) using time-average formula:
    Vs30 = 30 / sum( dz_i / Vs_i )
    
    Returns:
        vs30: float in m/s
        site_class: string classification according to NEHRP / Eurocode 8
    """
    # Sample top 30 meters with fine resolution (e.g., 0.5 m steps)
    z_fine = np.linspace(0.0, 30.0, 61)
    dz = z_fine[1] - z_fine[0]
    
    # Interpolate Vs to fine grid
    f_interp = interp1d(depth_m, vs_ms, bounds_error=False, fill_value=(vs_ms[0], vs_ms[-1]))
    vs_fine = f_interp(z_fine)
    vs_fine = np.maximum(vs_fine, 50.0)  # Avoid division by zero
    
    # Harmonic mean
    tt = np.sum(dz / vs_fine)
    vs30 = 30.0 / tt
    
    # NEHRP Soil classification
    if vs30 > 1500.0:
        site_class = "Clase A (Roca dura)"
    elif vs30 > 760.0:
        site_class = "Clase B (Roca)"
    elif vs30 > 360.0:
        site_class = "Clase C (Suelo muy denso / roca blanda)"
    elif vs30 >= 180.0:
        site_class = "Clase D (Suelo rígido / cohesivo)"
    else:
        site_class = "Clase E (Suelo blando / arcillas)"
        
    return float(vs30), site_class


def compute_fs_spectrum(
    power_fk: np.ndarray,
    freqs: np.ndarray,
    wavenumbers: np.ndarray,
    fmin: float = 1.0,
    fmax: float = 95.0,
    smin_skm: float = 0.25,
    smax_skm: float = 2.0,
    ns: int = 180
):
    """
    Transforms (f, k) power spectrum into frequency-slowness (f, s) spectrum.
    s = 1 / v = k / f  =>  k = f * s
    Returns:
        p_fs: 2D array of shape (ns, len(f_sel)) with rows=slowness, cols=frequency
        f_sel: 1D array of frequencies (Hz)
        s_skm: 1D array of phase slowness (s/km)
    """
    f_mask = (freqs >= fmin) & (freqs <= fmax)
    f_sel = freqs[f_mask]
    s_skm = np.linspace(smin_skm, smax_skm, ns, dtype=np.float32)

    pos_k = wavenumbers > 0.0
    k_pos_m = wavenumbers[pos_k]
    k_pos_km = k_pos_m * 1000.0

    p_pos = power_fk[f_mask, :][:, pos_k]
    p_fs = np.zeros((ns, len(f_sel)), dtype=np.float32)

    for i, f_val in enumerate(f_sel):
        if f_val <= 0:
            continue
        f_interp = interp1d(k_pos_km, p_pos[i], bounds_error=False, fill_value=0.0)
        target_k = f_val * s_skm
        p_fs[:, i] = f_interp(target_k)

    return p_fs, f_sel, s_skm
