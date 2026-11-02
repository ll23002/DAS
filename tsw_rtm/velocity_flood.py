"""
Velocity Flood Technique Module.
Implements lateral velocity flooding for TSW-RTM according to Li et al. (2020).
Builds:
- Sediment flood velocity model (vs)
- Metamorphic rock flood velocity model (vg)
- Perturbed flood models for sensitivity testing (+/- 10%)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FloodModels:
    true_model: np.ndarray
    sediment_flood: np.ndarray
    rock_flood: np.ndarray
    flood_line_sed: int  # x grid index where sediment flood begins
    flood_line_rock: int  # x grid index where rock flood begins


def generate_velocity_flood_models(
    base_model: np.ndarray,
    fault_x_idx: int,
    sediment_is_left: bool = True,
    flood_offset_cells: int = 15,
    perturb_sediment: float = 0.0,
    perturb_rock: float = 0.0
) -> FloodModels:
    """
    Constructs the dual velocity flood models for TSW-RTM.

    Parameters:
        base_model: 2D numpy array (nz, nx) representing the reference velocity model
        fault_x_idx: Approximate grid index in x where the fault / boundary is located
        sediment_is_left: True if low-velocity sediments are on the left (x < fault_x_idx)
        flood_offset_cells: Safety distance (in cells) from true fault to start replacement
        perturb_sediment: Fractional perturbation for sediment flood model (e.g. +0.10, -0.10)
        perturb_rock: Fractional perturbation for metamorphic rock flood model (e.g. +0.10, -0.10)

    Returns:
        FloodModels instance containing true_model, sediment_flood, rock_flood
    """
    nz, nx = base_model.shape

    sed_flood = np.zeros_like(base_model)
    rock_flood = np.zeros_like(base_model)

    if sediment_is_left:
        # Sediments on left: [0 : fault_x_idx]
        # Metamorphic rock on right: [fault_x_idx : nx]
        flood_x_sed = min(nx - 1, fault_x_idx - flood_offset_cells)
        flood_x_rock = max(0, fault_x_idx + flood_offset_cells)

        # 1. Sediment flood: take profile at flood_x_sed and extend to the right
        sed_flood[:, :flood_x_sed] = base_model[:, :flood_x_sed]
        for ix in range(flood_x_sed, nx):
            sed_flood[:, ix] = base_model[:, flood_x_sed]

        # 2. Rock flood: take profile at flood_x_rock and extend to the left
        rock_flood[:, flood_x_rock:] = base_model[:, flood_x_rock:]
        for ix in range(0, flood_x_rock):
            rock_flood[:, ix] = base_model[:, flood_x_rock]

    else:
        # Sediments on right, rock on left
        flood_x_sed = max(0, fault_x_idx + flood_offset_cells)
        flood_x_rock = min(nx - 1, fault_x_idx - flood_offset_cells)

        # 1. Sediment flood: extend from right to left
        sed_flood[:, flood_x_sed:] = base_model[:, flood_x_sed:]
        for ix in range(0, flood_x_sed):
            sed_flood[:, ix] = base_model[:, flood_x_sed]

        # 2. Rock flood: extend from left to right
        rock_flood[:, :flood_x_rock] = base_model[:, :flood_x_rock]
        for ix in range(flood_x_rock, nx):
            rock_flood[:, ix] = base_model[:, flood_x_rock]

    # Apply velocity perturbations if requested (e.g. +/- 10% sensitivity tests)
    if perturb_sediment != 0.0:
        sed_flood = sed_flood * (1.0 + perturb_sediment)
    if perturb_rock != 0.0:
        rock_flood = rock_flood * (1.0 + perturb_rock)

    return FloodModels(
        true_model=base_model.copy(),
        sediment_flood=sed_flood.astype(np.float32),
        rock_flood=rock_flood.astype(np.float32),
        flood_line_sed=flood_x_sed,
        flood_line_rock=flood_x_rock
    )
