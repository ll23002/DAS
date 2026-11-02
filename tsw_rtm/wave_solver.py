"""
Python wrapper for high-performance C 2D wave propagation engine.
Supports both 2D Elastic (staggered grid with free-surface Rayleigh waves)
and 2D Acoustic wave propagation and reverse-time migration.
"""

import os
import ctypes
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


# Load shared library
_lib_path = os.path.join(os.path.dirname(__file__), "c_engine", "libwave2d.so")
if not os.path.exists(_lib_path):
    raise FileNotFoundError(f"Shared library not found at {_lib_path}. Please compile libwave2d.so first.")

_lib = ctypes.CDLL(_lib_path)

# Declare C function signatures
_lib.acoustic_2d_step.argtypes = [
    ctypes.c_void_p,  # p_curr
    ctypes.c_void_p,  # p_prev
    ctypes.c_void_p,  # p_next
    ctypes.c_void_p,  # vel
    ctypes.c_void_p,  # sponge
    ctypes.c_int,     # nx
    ctypes.c_int,     # nz
    ctypes.c_float,   # dx
    ctypes.c_float,   # dz
    ctypes.c_float,   # dt
    ctypes.c_int      # free_surface_top
]

_lib.elastic_2d_step.argtypes = [
    ctypes.c_void_p,  # vx
    ctypes.c_void_p,  # vz
    ctypes.c_void_p,  # sxx
    ctypes.c_void_p,  # szz
    ctypes.c_void_p,  # sxz
    ctypes.c_void_p,  # lambda
    ctypes.c_void_p,  # mu
    ctypes.c_void_p,  # buo_x
    ctypes.c_void_p,  # buo_z
    ctypes.c_void_p,  # sponge
    ctypes.c_int,     # nx
    ctypes.c_int,     # nz
    ctypes.c_float,   # dx
    ctypes.c_float,   # dz
    ctypes.c_float,   # dt
    ctypes.c_int      # free_surface_top
]

_lib.correlate_imaging_condition.argtypes = [
    ctypes.c_void_p,  # src_wavefield
    ctypes.c_void_p,  # rec_wavefield
    ctypes.c_void_p,  # image
    ctypes.c_int,     # nx
    ctypes.c_int,     # nz
    ctypes.c_float    # dt
]

_lib.build_cerjan_sponge.argtypes = [
    ctypes.c_void_p,  # sponge
    ctypes.c_int,     # nx
    ctypes.c_int,     # nz
    ctypes.c_int,     # n_sponge
    ctypes.c_float,   # alpha
    ctypes.c_int      # free_surface_top
]


def ricker_wavelet(f0: float, dt: float, t_length: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a zero-phase Ricker wavelet."""
    nt = int(round(t_length / dt))
    t = np.arange(nt) * dt
    t0 = 1.0 / f0
    tau = t - t0
    w = (1.0 - 2.0 * (np.pi * f0 * tau) ** 2) * np.exp(-(np.pi * f0 * tau) ** 2)
    return w.astype(np.float32), t


class WaveSolver2D:
    """
    2D Wavefield Propagator using compiled C OpenMP engine.
    """

    def __init__(
        self,
        nx: int,
        nz: int,
        dx: float,
        dz: float,
        dt: float,
        n_sponge: int = 30,
        sponge_alpha: float = 0.0015,
        free_surface_top: bool = True,
        solver_type: str = "acoustic"
    ):
        self.nx = nx
        self.nz = nz
        self.dx = float(dx)
        self.dz = float(dz)
        self.dt = float(dt)
        self.n_sponge = n_sponge
        self.sponge_alpha = float(sponge_alpha)
        self.free_surface_top = free_surface_top
        self.solver_type = solver_type

        # Build sponge absorbing boundary
        self.sponge = np.ones((nz, nx), dtype=np.float32)
        _lib.build_cerjan_sponge(
            self.sponge.ctypes.data,
            nx, nz, n_sponge, sponge_alpha,
            1 if free_surface_top else 0
        )

    def check_stability(self, max_vel: float, min_vel: float, max_freq: float) -> Tuple[bool, str]:
        """Validates CFL stability and numerical dispersion."""
        cfl = max_vel * self.dt * np.sqrt(1.0 / (self.dx ** 2) + 1.0 / (self.dz ** 2))
        grid_pts_per_wl = (min_vel / max_freq) / max(self.dx, self.dz)

        msg = f"CFL number: {cfl:.3f} (target < 0.6). Grid pts per min wavelength: {grid_pts_per_wl:.1f} (target >= 5)."
        stable = (cfl < 0.65) and (grid_pts_per_wl >= 4.0)
        return stable, msg

    def forward_propagate_acoustic(
        self,
        velocity_model: np.ndarray,
        source_ts: np.ndarray,
        source_ix: int,
        source_iz: int,
        receiver_ix: np.ndarray,
        receiver_iz: int = 2,
        save_wavefield: bool = False,
        snapshot_interval: int = 1
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Runs forward acoustic simulation with source injection and receiver recording.

        Returns:
            recorded_traces: shape (nt, n_receivers)
            wavefields: optional shape (n_snaps, nz, nx)
        """
        nt = len(source_ts)
        vel = np.ascontiguousarray(velocity_model, dtype=np.float32)
        p_prev = np.zeros((self.nz, self.nx), dtype=np.float32)
        p_curr = np.zeros((self.nz, self.nx), dtype=np.float32)
        p_next = np.zeros((self.nz, self.nx), dtype=np.float32)

        n_rec = len(receiver_ix)
        rec_data = np.zeros((nt, n_rec), dtype=np.float32)

        snaps = [] if save_wavefield else None

        for it in range(nt):
            # Inject source
            p_curr[source_iz, source_ix] += source_ts[it] * (self.dt ** 2)

            # Step wavefield
            _lib.acoustic_2d_step(
                p_curr.ctypes.data,
                p_prev.ctypes.data,
                p_next.ctypes.data,
                vel.ctypes.data,
                self.sponge.ctypes.data,
                self.nx,
                self.nz,
                self.dx,
                self.dz,
                self.dt,
                1 if self.free_surface_top else 0
            )

            # Record receivers
            rec_data[it, :] = p_next[receiver_iz, receiver_ix]

            if save_wavefield and (it % snapshot_interval == 0):
                snaps.append(p_curr.copy())

            # Rotate buffers
            p_prev, p_curr, p_next = p_curr, p_next, p_prev

        wavefields_arr = np.array(snaps) if save_wavefield else None
        return rec_data, wavefields_arr

    def backward_propagate_and_correlate(
        self,
        velocity_model: np.ndarray,
        receiver_data: np.ndarray,
        receiver_ix: np.ndarray,
        receiver_iz: int,
        forward_wavefields: np.ndarray,
        image_out: np.ndarray,
        snapshot_interval: int = 1
    ):
        """
        Backpropagates time-reversed receiver traces and correlates with forward wavefields.
        """
        nt, n_rec = receiver_data.shape
        vel = np.ascontiguousarray(velocity_model, dtype=np.float32)
        p_prev = np.zeros((self.nz, self.nx), dtype=np.float32)
        p_curr = np.zeros((self.nz, self.nx), dtype=np.float32)
        p_next = np.zeros((self.nz, self.nx), dtype=np.float32)

        snap_idx = (nt - 1) // snapshot_interval

        # Loop backward in time: it from nt-1 down to 0
        for it in range(nt - 1, -1, -1):
            # Inject time-reversed receiver data
            for ir in range(n_rec):
                p_curr[receiver_iz, receiver_ix[ir]] += receiver_data[it, ir] * (self.dt ** 2)

            _lib.acoustic_2d_step(
                p_curr.ctypes.data,
                p_prev.ctypes.data,
                p_next.ctypes.data,
                vel.ctypes.data,
                self.sponge.ctypes.data,
                self.nx,
                self.nz,
                self.dx,
                self.dz,
                self.dt,
                1 if self.free_surface_top else 0
            )

            # Correlate with forward wavefield at matching snapshot
            if it % snapshot_interval == 0 and snap_idx >= 0 and snap_idx < len(forward_wavefields):
                fwd_snap = forward_wavefields[snap_idx]
                _lib.correlate_imaging_condition(
                    fwd_snap.ctypes.data,
                    p_curr.ctypes.data,
                    image_out.ctypes.data,
                    self.nx,
                    self.nz,
                    self.dt * snapshot_interval
                )
                snap_idx -= 1

            p_prev, p_curr, p_next = p_curr, p_next, p_prev
