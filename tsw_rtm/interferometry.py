"""
Ambient Noise Interferometry Module for DAS arrays.
Computes cross-correlations, empirical Green's functions via time derivatives,
and generates virtual common shot gathers.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy import signal


@dataclass
class VirtualShotGather:
    source_idx: int
    source_channel: int
    source_coord_x: float
    receiver_channels: np.ndarray
    receiver_coords_x: np.ndarray
    offsets: np.ndarray
    time_axis: np.ndarray
    shot_gather: np.ndarray  # shape: (nt, n_receivers)
    dt: float


def compute_cross_correlation(
    data: np.ndarray,
    source_channel_idx: int,
    max_lag_seconds: float,
    dt: float,
    receiver_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes ambient noise cross-correlations between a reference source channel
    and multiple receiver channels using FFT frequency domain multiplication.

    Parameters:
        data: Preprocessed continuous DAS data, shape (nt, n_channels)
        source_channel_idx: Column index of reference channel
        max_lag_seconds: Maximum lag time to keep (in seconds)
        dt: Time sampling interval (seconds)
        receiver_indices: Array of receiver column indices (default: all channels)

    Returns:
        ccfs: Cross-correlation matrix, shape (2 * n_lags + 1, len(receiver_indices))
        lags: 1D array of lag times in seconds [-max_lag, ..., 0, ..., +max_lag]
    """
    nt, n_channels = data.shape
    if receiver_indices is None:
        receiver_indices = np.arange(n_channels)

    n_fft = int(2 ** np.ceil(np.log2(2 * nt - 1)))
    n_lags = int(round(max_lag_seconds / dt))

    # FFT of source trace: conj(X_src(f))
    src_trace = data[:, source_channel_idx]
    src_fft = np.fft.rfft(src_trace, n=n_fft)
    src_fft_conj = np.conjugate(src_fft)

    # FFT of receiver traces
    rec_traces = data[:, receiver_indices]
    rec_fft = np.fft.rfft(rec_traces, n=n_fft, axis=0)

    # Cross-spectral density: src^* * rec
    cross_spec = src_fft_conj[:, np.newaxis] * rec_fft

    # Inverse FFT to time domain
    cc_full = np.fft.irfft(cross_spec, n=n_fft, axis=0)
    # Roll so lag 0 is centered at index n_fft // 2 or use fftshift
    # Full correlation has length n_fft; positive lags are [0, nt], negative lags [-nt, -1]
    cc_full = np.fft.fftshift(cc_full, axes=0)
    center = n_fft // 2

    start = center - n_lags
    end = center + n_lags + 1
    ccfs = cc_full[start:end, :].astype(np.float32)

    lags = np.arange(-n_lags, n_lags + 1) * dt
    return ccfs, lags


def time_derivative_greens(
    ccfs: np.ndarray,
    dt: float,
    scheme: str = "central"
) -> np.ndarray:
    """
    Computes time-derivative d(C_AB)/dt of cross-correlations to reconstruct
    the empirical Green's function (Equation 6 in Li et al., 2020).
    """
    if scheme == "central":
        # 2nd-order central difference
        d_ccf = np.gradient(ccfs, dt, axis=0)
    else:
        # Frequency domain derivative: iw * C(w)
        nt, nc = ccfs.shape
        n_fft = int(2 ** np.ceil(np.log2(nt)))
        freqs = np.fft.rfftfreq(n_fft, d=dt)
        omega = 2.0 * np.pi * freqs
        fft_c = np.fft.rfft(ccfs, n=n_fft, axis=0)
        d_fft = 1j * omega[:, np.newaxis] * fft_c
        d_ccf = np.fft.irfft(d_fft, n=n_fft, axis=0)[:nt, :]

    return d_ccf.astype(np.float32)


def full_green_function(ccfs: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Green's function over the full lag axis (-T..+T), keeping the acausal branch."""
    return time_derivative_greens(ccfs, float(lags[1] - lags[0]))


def build_virtual_shot_gather(
    ccfs: np.ndarray,
    lags: np.ndarray,
    source_idx: int,
    source_channel: int,
    channel_coords_x: np.ndarray,
    receiver_indices: np.ndarray,
    branch: str = "causal",
    take_derivative: bool = True
) -> VirtualShotGather:
    """
    Extracts a virtual shot gather from CCFs.

    Parameters:
        branch: 'causal' (lags >= 0), 'anticausal' (lags <= 0 reversed), or 'symmetric' (average)
        take_derivative: Whether to compute time-derivative for Green's function
    """
    dt = float(lags[1] - lags[0])
    zero_idx = np.argmin(np.abs(lags))

    processed_ccfs = time_derivative_greens(ccfs, dt) if take_derivative else ccfs

    if branch == "causal":
        gather = processed_ccfs[zero_idx:, :]
        time_axis = lags[zero_idx:]
    elif branch == "anticausal":
        gather = np.flip(processed_ccfs[:zero_idx + 1, :], axis=0)
        time_axis = np.abs(lags[:zero_idx + 1][::-1])
    elif branch == "symmetric":
        pos = processed_ccfs[zero_idx:, :]
        neg = np.flip(processed_ccfs[:zero_idx + 1, :], axis=0)
        min_len = min(pos.shape[0], neg.shape[0])
        gather = 0.5 * (pos[:min_len, :] + neg[:min_len, :])
        time_axis = lags[zero_idx:zero_idx + min_len]
    else:
        raise ValueError(f"Unknown branch option: {branch}")

    src_x = float(channel_coords_x[source_channel])
    rec_x = channel_coords_x[receiver_indices]
    offsets = np.abs(rec_x - src_x)

    return VirtualShotGather(
        source_idx=source_idx,
        source_channel=source_channel,
        source_coord_x=src_x,
        receiver_channels=receiver_indices,
        receiver_coords_x=rec_x,
        offsets=offsets,
        time_axis=time_axis,
        shot_gather=gather,
        dt=dt
    )
