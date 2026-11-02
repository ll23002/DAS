import os
import time

import numpy as np

from tsw_rtm.synthetic_models import create_vertical_fault_model, create_dipping_fault_model
from tsw_rtm.velocity_flood import generate_velocity_flood_models
from tsw_rtm.wave_solver import WaveSolver2D, ricker_wavelet
from tsw_rtm.rtm_engine import TSWRTMEngine
from tsw_rtm.visualizer import plot_flood_models, plot_tsw_rtm_image


def build_benchmark(caso, malla):
    args = dict(nx=malla.celdas_x, nz=malla.celdas_z, dx=malla.paso_x_m, dz=malla.paso_z_m)
    if caso.tipo == "vertical":
        return create_vertical_fault_model(fault_x_idx=caso.falla_x, **args)
    return create_dipping_fault_model(dip_angle_deg=caso.angulo, surface_fault_x_idx=caso.falla_x, **args)


def run_benchmark_case(benchmark, sint, output_prefix: str, perturb_sed: float = 0.0, perturb_rock: float = 0.0):
    print(f"\nCaso sintético: {benchmark.name} (sedimento {perturb_sed:+.0%}, roca {perturb_rock:+.0%})")

    nx, nz = benchmark.nx, benchmark.nz
    dx, dz = benchmark.dx, benchmark.dz
    dt = sint.paso_tiempo_s

    solver = WaveSolver2D(nx=nx, nz=nz, dx=dx, dz=dz, dt=dt, n_sponge=sint.celdas_esponja)
    engine = TSWRTMEngine(solver)

    fault_idx = int(round(benchmark.fault_x_surface / dx))
    flood_models = generate_velocity_flood_models(
        base_model=benchmark.velocity_model,
        fault_x_idx=fault_idx,
        sediment_is_left=True,
        flood_offset_cells=sint.celdas_inundacion,
        perturb_sediment=perturb_sed,
        perturb_rock=perturb_rock
    )

    plot_flood_models(
        true_vel=benchmark.velocity_model,
        sed_flood=flood_models.sediment_flood,
        rock_flood=flood_models.rock_flood,
        dx=dx,
        dz=dz,
        fault_x_idx=fault_idx,
        save_path=f"{output_prefix}_flood_models.png"
    )

    src_x_indices = np.linspace(35, fault_idx - 15, sint.num_fuentes, dtype=int)
    rec_x_indices = np.linspace(fault_idx + 15, nx - 35, sint.num_receptores, dtype=int)
    src_z_idx = 3
    rec_z_idx = 3

    src_ts, _ = ricker_wavelet(f0=sint.frecuencia_ricker_hz, dt=dt, t_length=sint.duracion_s)

    print("Generando registros en el modelo verdadero...")
    gathers = []
    t_start = time.time()
    for six in src_x_indices:
        rec_data, _ = solver.forward_propagate_acoustic(
            velocity_model=benchmark.velocity_model,
            source_ts=src_ts,
            source_ix=six,
            source_iz=src_z_idx,
            receiver_ix=rec_x_indices,
            receiver_iz=rec_z_idx,
            save_wavefield=False
        )
        gathers.append(rec_data)
    print(f"Simulación completada en {time.time() - t_start:.1f} s")

    print("Ejecutando migración TSW-RTM...")
    t_rtm = time.time()
    rtm_res = engine.migrate_multiple_shots(
        flood_models=flood_models,
        source_waveforms=[src_ts] * len(src_x_indices),
        source_coords_ix=list(src_x_indices),
        source_coords_iz=[src_z_idx] * len(src_x_indices),
        receiver_gathers=gathers,
        receiver_ix=rec_x_indices,
        receiver_iz=rec_z_idx,
        snapshot_interval=sint.intervalo_instantaneas,
        mute_sponge=True
    )
    print(f"Migración completada en {time.time() - t_rtm:.1f} s")

    title = f"TSW-RTM: {benchmark.name}"
    if perturb_sed != 0 or perturb_rock != 0:
        title += f" (Sed: {perturb_sed:+.0%}, Rock: {perturb_rock:+.0%})"

    plot_tsw_rtm_image(
        image=rtm_res.laplacian_image,
        true_vel=benchmark.velocity_model,
        fault_mask=benchmark.fault_mask,
        dx=dx,
        dz=dz,
        title=title,
        save_path=f"{output_prefix}_migrated_image.png"
    )


def run_synthetic(cfg):
    sint = cfg.sintetico
    os.makedirs(sint.resultados, exist_ok=True)

    for i, caso in enumerate(sint.casos, 1):
        benchmark = build_benchmark(caso, sint.malla)
        run_benchmark_case(
            benchmark,
            sint,
            f"{sint.resultados}/{i}_{caso.nombre}",
            perturb_sed=getattr(caso, "error_sedimento", 0.0),
            perturb_rock=getattr(caso, "error_roca", 0.0)
        )

    print(f"\nCasos sintéticos terminados. Resultados en '{sint.resultados}'.")
