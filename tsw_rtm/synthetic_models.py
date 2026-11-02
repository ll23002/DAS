"""
Synthetic Geological Benchmark Models from Li et al. (2020).
Generates:
1. Vertical fault model with sharp velocity contrast
2. Moderate-dip fault model (60 - 65 degrees)
3. Shallow-dip fault model (30 - 35 degrees)
4. Dual-fault zone model (resembling the Tanlu Fault Zone)
5. Perturbation configurations (+/- 10% velocity error)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SyntheticBenchmark:
    name: str
    velocity_model: np.ndarray  # (nz, nx)
    fault_mask: np.ndarray      # binary mask of true fault plane
    nx: int
    nz: int
    dx: float
    dz: float
    fault_x_surface: float
    v_sediment_surface: float
    v_rock: float


def create_vertical_fault_model(
    nx: int = 300,
    nz: int = 150,
    dx: float = 20.0,
    dz: float = 20.0,
    fault_x_idx: int = 150,
    v_sed_top: float = 1200.0,
    v_sed_grad: float = 0.2,
    v_rock: float = 3000.0
) -> SyntheticBenchmark:
    """
    Creates vertical fault model separating low-velocity sediment from high-velocity rock.
    """
    vel = np.zeros((nz, nx), dtype=np.float32)
    fault_mask = np.zeros((nz, nx), dtype=bool)

    for iz in range(nz):
        z = iz * dz
        v_sed = v_sed_top + v_sed_grad * z
        vel[iz, :fault_x_idx] = v_sed
        vel[iz, fault_x_idx:] = v_rock
        fault_mask[iz, fault_x_idx] = True

    return SyntheticBenchmark(
        name="Vertical Fault",
        velocity_model=vel,
        fault_mask=fault_mask,
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        fault_x_surface=fault_x_idx * dx,
        v_sediment_surface=v_sed_top,
        v_rock=v_rock
    )


def create_dipping_fault_model(
    dip_angle_deg: float = 65.0,
    nx: int = 300,
    nz: int = 150,
    dx: float = 20.0,
    dz: float = 20.0,
    surface_fault_x_idx: int = 150,
    v_sed_top: float = 1200.0,
    v_sed_grad: float = 0.2,
    v_rock: float = 3000.0
) -> SyntheticBenchmark:
    """
    Creates dipping fault model with specified dip angle in degrees.
    """
    vel = np.zeros((nz, nx), dtype=np.float32)
    fault_mask = np.zeros((nz, nx), dtype=bool)

    dip_rad = np.radians(dip_angle_deg)
    # cot(dip) = dx / dz -> x(z) = x_0 + z / tan(dip)
    cot_dip = 1.0 / np.tan(dip_rad)

    for iz in range(nz):
        z = iz * dz
        shift_x = (z * cot_dip) / dx
        fx = int(round(surface_fault_x_idx + shift_x))
        fx = max(0, min(nx - 1, fx))

        v_sed = v_sed_top + v_sed_grad * z
        vel[iz, :fx] = v_sed
        vel[iz, fx:] = v_rock
        fault_mask[iz, fx] = True

    name = f"Dipping Fault ({dip_angle_deg:.0f} deg)"
    return SyntheticBenchmark(
        name=name,
        velocity_model=vel,
        fault_mask=fault_mask,
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        fault_x_surface=surface_fault_x_idx * dx,
        v_sediment_surface=v_sed_top,
        v_rock=v_rock
    )


def create_fault_zone_model(
    nx: int = 350,
    nz: int = 150,
    dx: float = 20.0,
    dz: float = 20.0,
    fault_left_x_idx: int = 110,
    fault_right_x_idx: int = 240,
    v_sed_top: float = 1200.0,
    v_sed_grad: float = 0.2,
    v_rock: float = 3000.0
) -> SyntheticBenchmark:
    """
    Creates dual-fault zone model (like Tanlu fault):
    Left sediments, central metamorphic rock block, right sediments.
    """
    vel = np.zeros((nz, nx), dtype=np.float32)
    fault_mask = np.zeros((nz, nx), dtype=bool)

    dip_left_deg = 70.0
    dip_right_deg = 70.0
    cot_left = -1.0 / np.tan(np.radians(dip_left_deg))  # dips outward
    cot_right = 1.0 / np.tan(np.radians(dip_right_deg))

    for iz in range(nz):
        z = iz * dz
        v_sed = v_sed_top + v_sed_grad * z

        fx_left = int(round(fault_left_x_idx + (z * cot_left) / dx))
        fx_right = int(round(fault_right_x_idx + (z * cot_right) / dx))
        fx_left = max(0, min(nx - 1, fx_left))
        fx_right = max(0, min(nx - 1, fx_right))

        vel[iz, :fx_left] = v_sed
        vel[iz, fx_left:fx_right] = v_rock
        vel[iz, fx_right:] = v_sed

        fault_mask[iz, fx_left] = True
        fault_mask[iz, fx_right] = True

    return SyntheticBenchmark(
        name="Dual Fault Zone (TFZ Style)",
        velocity_model=vel,
        fault_mask=fault_mask,
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        fault_x_surface=fault_left_x_idx * dx,
        v_sediment_surface=v_sed_top,
        v_rock=v_rock
    )
