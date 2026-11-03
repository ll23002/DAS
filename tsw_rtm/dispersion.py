import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter


def compute_kf_grid(
    power_fk: np.ndarray,
    freqs: np.ndarray,
    wavenumbers: np.ndarray,
    fmax: float = 50.0,
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
    fmin: float = 0.8,
    fmax: float = 40.0,
    vmin_kms: float = 0.5,
    vmax_kms: float = 4.0,
    nv: int = 150
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
    depth_max_m: float = 1200.0
):
    """
    Extracts fundamental dispersion curve v_phase(f) from peak energy
    and computes an approximate 1D shear-wave velocity profile Vs(z).
    z approx lambda / 2 = v / (2 * f)
    Vs approx v_phase / 0.92
    """
    peak_indices = np.argmax(p_fv, axis=0)
    v_curve_kms = v_kms[peak_indices]
    v_curve_kms = median_filter(v_curve_kms, size=5)

    # Approximate depth sensitivity: z = lambda / 2 = (v * 1000) / (2 * f)
    wavelength_m = (v_curve_kms * 1000.0) / np.maximum(freqs, 0.1)
    depth_m = wavelength_m / 2.0
    vs_kms = v_curve_kms / 0.92

    # Filter valid depths
    valid = (depth_m > 0) & (depth_m <= depth_max_m)
    depth_m = depth_m[valid]
    vs_kms = vs_kms[valid]

    # Sort monotonically by depth
    sort_idx = np.argsort(depth_m)
    depth_m = depth_m[sort_idx]
    vs_kms = vs_kms[sort_idx]

    return v_curve_kms, depth_m, vs_kms


def compute_fs_spectrum(
    power_fk: np.ndarray,
    freqs: np.ndarray,
    wavenumbers: np.ndarray,
    fmin: float = 0.8,
    fmax: float = 40.0,
    smin_skm: float = 0.25,
    smax_skm: float = 2.0,
    ns: int = 150
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
