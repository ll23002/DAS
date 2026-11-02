"""
TSW-RTM Engine Module.
Orchestrates Reverse Time Migration with Velocity Flood for Transmitted Surface Waves.
Computes Zero-Lag Cross-Correlation Imaging Condition, Illumination Normalization,
and Laplacian Filtering.
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .wave_solver import WaveSolver2D, ricker_wavelet
from .velocity_flood import FloodModels


@dataclass
class RTMResult:
    raw_image: np.ndarray
    laplacian_image: np.ndarray
    normalized_image: np.ndarray
    grid_x: np.ndarray
    grid_z: np.ndarray
    dx: float
    dz: float


def apply_laplacian_filter(image: np.ndarray, dx: float, dz: float) -> np.ndarray:
    """
    Applies 2D Laplacian filter to remove low-frequency acquisition artifacts.
    """
    lap_kernel = np.array([
        [0.0,  1.0 / (dz * dz), 0.0],
        [1.0 / (dx * dx), -2.0 / (dx * dx) - 2.0 / (dz * dz), 1.0 / (dx * dx)],
        [0.0,  1.0 / (dz * dz), 0.0]
    ], dtype=np.float32)

    filtered = signal.convolve2d(image, lap_kernel, mode='same', boundary='symm')
    return filtered


class TSWRTMEngine:
    """
    Transmitted Surface Wave Reverse Time Migration Engine.
    """

    def __init__(self, solver: WaveSolver2D):
        self.solver = solver

    def migrate_shot(
        self,
        flood_models: FloodModels,
        source_ts: np.ndarray,
        source_ix: int,
        source_iz: int,
        receiver_data: np.ndarray,
        receiver_ix: np.ndarray,
        receiver_iz: int = 2,
        snapshot_interval: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes TSW-RTM for a single virtual shot gather.

        Parameters:
            flood_models: FloodModels containing sediment_flood and rock_flood models
            source_ts: Source wavelet or virtual source waveform (1D array)
            source_ix: Grid index in x of virtual source
            source_iz: Grid index in z of virtual source
            receiver_data: Recorded / retrieved surface wave gather, shape (nt, n_receivers)
            receiver_ix: Array of receiver grid indices in x
            receiver_iz: Grid index in z of receivers

        Returns:
            shot_image: Migrated image for this shot (nz, nx)
            illumination: Source energy illumination map (nz, nx)
        """
        # 1. Forward propagation in sediment flood model vs
        rec_fwd, fwd_snaps = self.solver.forward_propagate_acoustic(
            velocity_model=flood_models.sediment_flood,
            source_ts=source_ts,
            source_ix=source_ix,
            source_iz=source_iz,
            receiver_ix=receiver_ix,
            receiver_iz=receiver_iz,
            save_wavefield=True,
            snapshot_interval=snapshot_interval
        )

        # Compute source illumination map
        illumination = np.sum(fwd_snaps ** 2, axis=0) * (self.solver.dt * snapshot_interval)

        # 2. Backward propagation in metamorphic rock flood model vg & correlate
        shot_image = np.zeros((self.solver.nz, self.solver.nx), dtype=np.float32)

        self.solver.backward_propagate_and_correlate(
            velocity_model=flood_models.rock_flood,
            receiver_data=receiver_data,
            receiver_ix=receiver_ix,
            receiver_iz=receiver_iz,
            forward_wavefields=fwd_snaps,
            image_out=shot_image,
            snapshot_interval=snapshot_interval
        )

        return shot_image, illumination

    def migrate_multiple_shots(
        self,
        flood_models: FloodModels,
        source_waveforms: List[np.ndarray],
        source_coords_ix: List[int],
        source_coords_iz: List[int],
        receiver_gathers: List[np.ndarray],
        receiver_ix: np.ndarray,
        receiver_iz: int = 2,
        snapshot_interval: int = 2,
        mute_sponge: bool = True
    ) -> RTMResult:
        """
        Runs multi-shot TSW-RTM and stacks the results.
        """
        n_shots = len(source_coords_ix)
        stacked_image = np.zeros((self.solver.nz, self.solver.nx), dtype=np.float32)
        total_illumination = np.zeros((self.solver.nz, self.solver.nx), dtype=np.float32)

        for s_idx in range(n_shots):
            src_ts = source_waveforms[s_idx]
            six = source_coords_ix[s_idx]
            siz = source_coords_iz[s_idx]
            rec_gather = receiver_gathers[s_idx]

            shot_img, shot_illum = self.migrate_shot(
                flood_models=flood_models,
                source_ts=src_ts,
                source_ix=six,
                source_iz=siz,
                receiver_data=rec_gather,
                receiver_ix=receiver_ix,
                receiver_iz=receiver_iz,
                snapshot_interval=snapshot_interval
            )

            stacked_image += shot_img
            total_illumination += shot_illum

        # Mute sponge absorbing boundaries if requested
        if mute_sponge:
            sp_mask = self.solver.sponge < 0.99
            stacked_image[sp_mask] = 0.0
            total_illumination[sp_mask] = 0.0

        # Illumination normalization
        eps = 1e-4 * np.max(total_illumination) if np.max(total_illumination) > 0 else 1e-12
        normalized_image = stacked_image / (np.sqrt(total_illumination) + eps)

        # Laplacian filtering to sharpen fault boundary
        lap_image = apply_laplacian_filter(stacked_image, self.solver.dx, self.solver.dz)
        if mute_sponge:
            lap_image[sp_mask] = 0.0

        grid_x = np.arange(self.solver.nx) * self.solver.dx
        grid_z = np.arange(self.solver.nz) * self.solver.dz

        return RTMResult(
            raw_image=stacked_image,
            laplacian_image=lap_image,
            normalized_image=normalized_image,
            grid_x=grid_x,
            grid_z=grid_z,
            dx=self.solver.dx,
            dz=self.solver.dz
        )
