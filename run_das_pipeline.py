import csv
import os
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.interpolate import interp1d

from tsw_rtm.das_io import DASReader, DASFolderReader
from tsw_rtm.preprocessing import (
    bandpass_filter,
    decimate_time,
    temporal_normalization,
    spectral_whitening,
    fk_filter
)
from tsw_rtm.interferometry import (
    compute_cross_correlation,
    build_virtual_shot_gather,
    full_green_function
)
from tsw_rtm.dispersion import (
    compute_kf_grid,
    compute_fv_spectrum,
    extract_dispersion_and_1d_profile,
    compute_fs_spectrum
)
from tsw_rtm.doc_builder import build_das_guide_docx
from tsw_rtm.velocity_flood import generate_velocity_flood_models
from tsw_rtm.wave_solver import WaveSolver2D, ricker_wavelet
from tsw_rtm.rtm_engine import TSWRTMEngine
from tsw_rtm.visualizer import (
    plot_das_raw_and_preprocessed,
    plot_virtual_shot_gather,
    plot_flood_models,
    plot_tsw_rtm_image,
    plot_fk_spectra,
    plot_fk_spectra_batches,
    plot_ccf_and_green,
    plot_fk_dispersion_image_c,
    plot_fv_and_1d_profile,
    plot_fs_spectrum,
    save_matrix_csv
)


def resample_gather_to_sim_dt(gather: np.ndarray, dt_orig: float, dt_sim: float, target_nt: int) -> np.ndarray:
    """Interpolates time axis of shot gather to numerical simulation dt."""
    nt_orig, n_rec = gather.shape
    t_orig = np.arange(nt_orig) * dt_orig
    t_sim = np.arange(target_nt) * dt_sim

    interp_func = interp1d(t_orig, gather, axis=0, kind='linear', bounds_error=False, fill_value=0.0)
    return interp_func(t_sim).astype(np.float32)


def compute_fk_power(data: np.ndarray, dt: float, dx: float):
    """
    Power spectrum |FFT2|^2 of (time, channel) data with a Hann taper in both axes.
    Returns (power[f>=0, k], freqs[f>=0], wavenumbers) with fixed FFT sizes (2^n) so files are comparable.
    """
    nt, nx = data.shape
    win = np.hanning(nt)[:, None] * np.hanning(nx)[None, :]
    n_t = int(2 ** np.ceil(np.log2(nt)))
    n_x = int(2 ** np.ceil(np.log2(nx)))
    spec = np.fft.fftshift(np.fft.fft2((data - data.mean(0)) * win, s=(n_t, n_x)), axes=1)
    freqs = np.fft.fftfreq(n_t, d=dt)
    ks = np.fft.fftshift(np.fft.fftfreq(n_x, d=dx))
    pos = freqs >= 0
    power = (np.abs(spec[pos]) ** 2) / (nt * nx)
    return power, freqs[pos], ks


def fk_metrics(power, freqs, ks, vmin, vmax, fmin, fmax):
    """Scalar F-K descriptors: fraction of band energy inside [vmin,vmax] and the +k / -k balance."""
    F, K = np.meshgrid(freqs, ks, indexing='ij')
    band = (F >= fmin) & (F <= fmax)
    with np.errstate(divide='ignore', invalid='ignore'):
        vapp = np.where(np.abs(K) > 0, F / np.abs(K), np.inf)
    inside = band & (vapp >= vmin) & (vapp <= vmax)
    tot = power[band].sum() + 1e-30
    pos_k = power[band & (K > 0)].sum()
    neg_k = power[band & (K < 0)].sum()
    return {
        "band_power": float(tot),
        "frac_in_velocity_window": float(power[inside].sum() / tot),
        "frac_slow_below_vmin": float(power[band & (vapp < vmin)].sum() / tot),
        "frac_fast_above_vmax": float(power[band & (vapp > vmax)].sum() / tot),
        "pos_k_over_neg_k": float(pos_k / (neg_k + 1e-30)),
    }


def build_velocity_model(nz: int, nx: int, dz: float, vm) -> tuple:
    """Two-block velocity model (sediment | rock) with a vertical fault at the center."""
    z = np.arange(nz)[:, None] * dz
    fault_x_idx = nx // 2
    base_vel = np.zeros((nz, nx), dtype=np.float32)
    v_sed = vm.sedimento_velocidad_ms + vm.sedimento_gradiente * z
    v_rock = vm.roca_velocidad_ms + vm.roca_gradiente * z
    left, right = (v_sed, v_rock) if vm.sedimento_a_la_izquierda else (v_rock, v_sed)
    base_vel[:, :fault_x_idx] = left
    base_vel[:, fault_x_idx:] = right
    fault_mask = np.zeros((nz, nx), dtype=bool)
    fault_mask[:, fault_x_idx] = True
    return base_vel, fault_mask, fault_x_idx


def run_pipeline(cfg):
    real = cfg.real
    output_dir = real.resultados
    os.makedirs(f"{output_dir}/csv", exist_ok=True)

    channel_start = real.cable.canal_inicial
    num_channels = real.cable.num_canales
    target_fs = real.filtrado.frecuencia_muestreo_hz
    fmin = real.filtrado.frecuencia_min_hz
    fmax = real.filtrado.frecuencia_max_hz
    max_lag_s = real.correlacion.retardo_max_s
    fk_vmin = real.filtro_fk.velocidad_min_ms
    fk_vmax = real.filtro_fk.velocidad_max_ms
    wf_offset = real.correlacion.offset_forma_onda_km
    wf_offset = None if wf_offset == "auto" else float(wf_offset)

    input_path = os.path.expanduser(real.datos)
    is_folder = os.path.isdir(input_path)
    print(f"\nProcesando datos DAS: {input_path}")
    print(f"Canales: [{channel_start} : {channel_start + num_channels}]")

    if is_folder:
        folder_reader = DASFolderReader(input_path)
        metadata = folder_reader.metadata
        file_list = folder_reader.file_list
    else:
        metadata = DASReader(input_path).metadata
        file_list = [input_path]
    if real.max_archivos is not None:
        file_list = file_list[:real.max_archivos]

    dx = float(metadata.dx)
    fs_orig = float(metadata.sampling_rate)
    print(f"Archivos a procesar: {len(file_list)}")
    print(f"Fs={fs_orig} Hz, dx={dx:.2f} m, canales totales={metadata.num_channels}")

    n_srcs = real.correlacion.num_fuentes_virtuales
    v_src_local_indices = list(np.linspace(20, num_channels // 4, n_srcs, dtype=int))
    rec_local_indices = np.arange(num_channels // 3, num_channels - 20, real.correlacion.receptores_paso)

    stacked_ccfs = {src_idx: None for src_idx in v_src_local_indices}
    lags_axis = None
    dt_proc = 1.0 / target_fs
    qc_saved = False
    fk_store, fk_rows = [], []

    t_stack_start = time.time()
    for f_idx, fpath in enumerate(file_list):
        fname = os.path.basename(fpath)
        print(f"  [{f_idx + 1}/{len(file_list)}] {fname}")
        reader = DASReader(fpath)
        raw_data, time_raw, dist_raw = reader.read_window(
            channel_min=channel_start,
            channel_max=channel_start + num_channels
        )

        dec_data, dt_proc, fs_proc = decimate_time(raw_data, fs_orig, target_fs=target_fs)
        bp_data = bandpass_filter(dec_data, fs=fs_proc, fmin=fmin, fmax=fmax)
        norm_data = temporal_normalization(bp_data, method='1bit')
        white_data = spectral_whitening(norm_data, fs=fs_proc, fmin=fmin, fmax=fmax)

        fk_pow, fk_f, fk_k = compute_fk_power(bp_data, dt=dt_proc, dx=dx)
        fk_store.append((fname, fk_pow, fk_f, fk_k))
        m = fk_metrics(fk_pow, fk_f, fk_k, fk_vmin, fk_vmax, fmin, fmax)
        m.update({
            "file": fname,
            "timestamp": reader.metadata.timestamp,
            "raw_peak_abs": float(np.max(np.abs(raw_data))),
            "raw_rms": float(np.sqrt(np.mean(raw_data ** 2))),
            "peak_over_rms": float(np.max(np.abs(raw_data)) / (np.sqrt(np.mean(raw_data ** 2)) + 1e-30)),
        })
        fk_rows.append(m)
        save_matrix_csv(f"{output_dir}/csv/6_fk_file{f_idx}.csv", 10 * np.log10(fk_pow + 1e-30), fk_f, fk_k,
                        "freq_Hz", "wavenumber_1_per_m")

        if not qc_saved:
            time_proc = np.arange(len(white_data)) * dt_proc
            n_raw_qc = min(2000, len(time_raw))
            n_proc_qc = min(int(2 * fs_proc), len(time_proc))
            plot_das_raw_and_preprocessed(
                raw_data=raw_data[:n_raw_qc, :],
                proc_data=white_data[:n_proc_qc, :],
                time_raw=time_raw[:n_raw_qc],
                time_proc=time_proc[:n_proc_qc],
                distance_axis=dist_raw - dist_raw[0],
                save_path=f"{output_dir}/1_das_qc_preprocessing.png"
            )
            save_matrix_csv(f"{output_dir}/csv/1a_raw_das.csv", raw_data[:n_raw_qc, :], time_raw[:n_raw_qc],
                            dist_raw - dist_raw[0], "time_s", "distance_m")
            save_matrix_csv(f"{output_dir}/csv/1b_preprocessed_das.csv", white_data[:n_proc_qc, :],
                            time_proc[:n_proc_qc], dist_raw - dist_raw[0], "time_s", "distance_m")
            qc_saved = True

        for src_local in v_src_local_indices:
            ccf, lags = compute_cross_correlation(
                data=white_data,
                source_channel_idx=src_local,
                max_lag_seconds=max_lag_s,
                dt=dt_proc,
                receiver_indices=rec_local_indices
            )
            if stacked_ccfs[src_local] is None:
                stacked_ccfs[src_local] = ccf
                lags_axis = lags
            else:
                stacked_ccfs[src_local] += ccf

    plot_fk_spectra(
        spectra=[s[1] for s in fk_store], freqs=fk_store[0][2], wavenumbers=fk_store[0][3],
        titles=[s[0] for s in fk_store], save_path=f"{output_dir}/6_fk_per_file.png", fmax_plot=fmax + 1.0,
        vlines=(fk_vmin, fk_vmax)
    )
    plot_fk_spectra_batches(
        spectra=[s[1] for s in fk_store], freqs=fk_store[0][2], wavenumbers=fk_store[0][3],
        titles=[s[0] for s in fk_store], save_prefix=f"{output_dir}/6_fk", fmax_plot=fmax + 1.0,
        vlines=(fk_vmin, fk_vmax), batch_size=3
    )
    with open(f"{output_dir}/csv/6_fk_metrics_per_file.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fk_rows[0].keys()))
        w.writeheader()
        w.writerows(fk_rows)

    # 2.1, 2.2, 2.3 Dispersion Analysis: F-K Image C, F-V with 1D profile, F-S
    print("Calculando espectros de dispersión (F-K imagen c, F-V con 1D, F-S)...")
    mean_fk_pow = np.mean([s[1] for s in fk_store], axis=0)
    fk_f_axis = fk_store[0][2]
    fk_k_axis = fk_store[0][3]

    # Numeral 2.1: Image c (k vs f)
    p_kf, f_sel_c, k_sel_km = compute_kf_grid(
        mean_fk_pow, fk_f_axis, fk_k_axis, fmax=min(80.0, target_fs / 2.0), kmax_km=25.0
    )
    plot_fk_dispersion_image_c(f_sel_c, k_sel_km, p_kf, save_path=f"{output_dir}/2_1_fk_dispersion_c.png")
    save_matrix_csv(f"{output_dir}/csv/2_1_fk_dispersion_c.csv", p_kf, k_sel_km, f_sel_c, "wavenumber_1_per_km", "freq_Hz")

    # Numeral 2.2: F-V and 1D Vs profile
    p_fv, f_sel_v, v_kms = compute_fv_spectrum(
        mean_fk_pow, fk_f_axis, fk_k_axis, fmin=fmin, fmax=min(45.0, target_fs / 2.0)
    )
    v_curve, depth_m, vs_kms = extract_dispersion_and_1d_profile(p_fv, f_sel_v, v_kms)
    plot_fv_and_1d_profile(f_sel_v, v_kms, p_fv, depth_m, vs_kms, save_path=f"{output_dir}/2_2_fv_and_1d_profile.png")
    save_matrix_csv(f"{output_dir}/csv/2_2_fv_spectrum.csv", p_fv, v_kms, f_sel_v, "velocity_km_per_s", "freq_Hz")

    # Numeral 2.3: F-S
    p_fs, f_sel_s, s_skm = compute_fs_spectrum(
        mean_fk_pow, fk_f_axis, fk_k_axis, fmin=fmin, fmax=min(45.0, target_fs / 2.0)
    )
    plot_fs_spectrum(f_sel_s, s_skm, p_fs, save_path=f"{output_dir}/2_3_fs_spectrum.png")
    save_matrix_csv(f"{output_dir}/csv/2_3_fs_spectrum.csv", p_fs, s_skm, f_sel_s, "slowness_s_per_km", "freq_Hz")

    n_files_done = len(file_list)
    for src_local in v_src_local_indices:
        stacked_ccfs[src_local] /= n_files_done
    print(f"Apilado completado en {time.time() - t_stack_start:.1f} s ({n_files_done} archivos)")

    print("Extrayendo funciones de Green empíricas...")
    virtual_gathers = []
    dist_rel = np.arange(num_channels) * dx

    for v_idx, src_local in enumerate(v_src_local_indices):
        ccf_i = stacked_ccfs[src_local]
        vshot = build_virtual_shot_gather(
            ccfs=ccf_i,
            lags=lags_axis,
            source_idx=v_idx,
            source_channel=src_local,
            channel_coords_x=dist_rel,
            receiver_indices=rec_local_indices,
            branch="causal",
            take_derivative=True
        )
        filtered_gather = fk_filter(vshot.shot_gather, dt=vshot.dt, dx=dx * real.correlacion.receptores_paso,
                                    vmin=fk_vmin, vmax=fk_vmax)
        virtual_gathers.append(filtered_gather)

        offs_km = (dist_rel[rec_local_indices] - dist_rel[src_local]) / 1000.0
        green_full = full_green_function(ccf_i, lags_axis)
        plot_ccf_and_green(
            ccf=ccf_i, green=green_full, lags=lags_axis, offsets_km=offs_km,
            title=f"Source channel #{channel_start + src_local} ({n_files_done} files stacked)",
            save_path=f"{output_dir}/5_ccf_green_src{v_idx}.png",
            waveform_offset_km=wf_offset
        )
        save_matrix_csv(f"{output_dir}/csv/5_ccf_src{v_idx}.csv", ccf_i, lags_axis, offs_km, "lag_s", "offset_km")
        save_matrix_csv(f"{output_dir}/csv/5_green_src{v_idx}.csv", green_full, lags_axis, offs_km,
                        "lag_s", "offset_km")
        save_matrix_csv(f"{output_dir}/csv/2_virtual_shot_gather_fkfiltered_src{v_idx}.csv", filtered_gather,
                        np.arange(filtered_gather.shape[0]) * dt_proc, offs_km, "time_s", "offset_km")

    offs_first = (dist_rel[rec_local_indices] - dist_rel[v_src_local_indices[0]]) / 1000.0
    time_first = np.arange(virtual_gathers[0].shape[0]) * dt_proc
    plot_virtual_shot_gather(
        gather=virtual_gathers[0],
        time_axis=time_first,
        offsets_km=offs_first,
        title=f"Stacked Virtual Shot Gather ({n_files_done} files, Source #{channel_start + v_src_local_indices[0]})",
        save_path=f"{output_dir}/2_virtual_shot_gather.png"
    )
    save_matrix_csv(f"{output_dir}/csv/2_virtual_shot_gather.csv", virtual_gathers[0], time_first, offs_first,
                    "time_s", "offset_km")

    print("Construyendo modelo de velocidad e inundaciones...")
    vm = real.modelo_velocidad
    nx = num_channels // 2
    nz = vm.num_celdas_profundidad
    dx_sim = dx * 2.0
    dz_sim = vm.paso_profundidad_m

    base_vel, fault_mask, fault_x_idx = build_velocity_model(nz, nx, dz_sim, vm)
    flood_models = generate_velocity_flood_models(
        base_model=base_vel,
        fault_x_idx=fault_x_idx,
        sediment_is_left=vm.sedimento_a_la_izquierda,
        flood_offset_cells=vm.celdas_inundacion
    )

    plot_flood_models(
        true_vel=base_vel,
        sed_flood=flood_models.sediment_flood,
        rock_flood=flood_models.rock_flood,
        dx=dx_sim,
        dz=dz_sim,
        fault_x_idx=fault_x_idx,
        save_path=f"{output_dir}/3_das_flood_models.png",
        fault_label="Falla supuesta"
    )
    for name, model in (("true_vel", base_vel), ("sediment_flood", flood_models.sediment_flood),
                        ("rock_flood", flood_models.rock_flood)):
        save_matrix_csv(f"{output_dir}/csv/3_{name}.csv", model, np.arange(nz) * dz_sim / 1000.0,
                        np.arange(nx) * dx_sim / 1000.0, "depth_km", "distance_km")

    mig = real.migracion
    dt_sim = mig.paso_tiempo_s
    sim_nt = int(round(max_lag_s / dt_sim))

    solver = WaveSolver2D(nx=nx, nz=nz, dx=dx_sim, dz=dz_sim, dt=dt_sim, n_sponge=mig.celdas_esponja)
    _, cfl_msg = solver.check_stability(float(base_vel.max()), float(base_vel.min()), fmax)
    print(f"Estabilidad numérica: {cfl_msg}")

    engine = TSWRTMEngine(solver)

    src_sim_ix = [int(round(s / 2)) for s in v_src_local_indices]
    rec_sim_ix = np.array([int(round(r / 2)) for r in rec_local_indices], dtype=int)
    src_sim_iz = [2] * len(src_sim_ix)

    resampled_gathers = [
        resample_gather_to_sim_dt(g, dt_orig=dt_proc, dt_sim=dt_sim, target_nt=sim_nt)
        for g in virtual_gathers
    ]

    src_wave, _ = ricker_wavelet(f0=mig.frecuencia_ricker_hz, dt=dt_sim, t_length=max_lag_s)
    source_waveforms = [src_wave] * len(src_sim_ix)

    print("Ejecutando migración TSW-RTM...")
    t_rtm_start = time.time()
    rtm_res = engine.migrate_multiple_shots(
        flood_models=flood_models,
        source_waveforms=source_waveforms,
        source_coords_ix=src_sim_ix,
        source_coords_iz=src_sim_iz,
        receiver_gathers=resampled_gathers,
        receiver_ix=rec_sim_ix,
        receiver_iz=2,
        snapshot_interval=mig.intervalo_instantaneas,
        mute_sponge=True
    )
    print(f"Migración completada en {time.time() - t_rtm_start:.1f} s")

    plot_tsw_rtm_image(
        image=rtm_res.laplacian_image,
        true_vel=base_vel,
        fault_mask=fault_mask,
        dx=dx_sim,
        dz=dz_sim,
        title=f"TSW-RTM Fault Image from DAS Ambient Noise ({n_files_done} files stacked)",
        save_path=f"{output_dir}/4_das_migrated_fault_image.png",
        fault_label="Falla supuesta"
    )
    save_matrix_csv(f"{output_dir}/csv/4_migrated_fault_image.csv", rtm_res.laplacian_image,
                    np.arange(nz) * dz_sim / 1000.0, np.arange(nx) * dx_sim / 1000.0, "depth_km", "distance_km")

    lado_sed = "izquierda" if vm.sedimento_a_la_izquierda else "derecha"
    parameters = [
        ("Modelo de velocidad", "Velocidad del sedimento en superficie", f"{vm.sedimento_velocidad_ms:g} m/s"),
        ("Modelo de velocidad", "Aumento de velocidad con la profundidad (sedimento)", f"{vm.sedimento_gradiente:g} m/s por metro"),
        ("Modelo de velocidad", "Velocidad de la roca en superficie", f"{vm.roca_velocidad_ms:g} m/s"),
        ("Modelo de velocidad", "Aumento de velocidad con la profundidad (roca)", f"{vm.roca_gradiente:g} m/s por metro"),
        ("Modelo de velocidad", "Posición de la falla supuesta", f"x = {fault_x_idx * dx_sim / 1000.0:.2f} km (centro de la ventana)"),
        ("Modelo de velocidad", "Lado del sedimento respecto a la falla", lado_sed),
        ("Modelo de velocidad", "Tamaño de la malla", f"{nx} x {nz} celdas de {dx_sim:.2f} m x {dz_sim:g} m"),
        ("Modelo de velocidad", "Profundidad del modelo", f"{nz * dz_sim / 1000.0:g} km"),
        ("Modelo de velocidad", "Margen de inundación lateral", f"{vm.celdas_inundacion} celdas"),
        ("Migración", "Paso de tiempo de la simulación", f"{mig.paso_tiempo_s:g} s"),
        ("Migración", "Frecuencia central de la fuente (Ricker)", f"{mig.frecuencia_ricker_hz:g} Hz"),
        ("Migración", "Duración simulada", f"{max_lag_s:g} s"),
        ("Migración", "Celdas de borde absorbente", f"{mig.celdas_esponja}"),
        ("Migración", "Intervalo de instantáneas", f"cada {mig.intervalo_instantaneas} pasos"),
        ("Migración", "Fuentes virtuales", f"{n_srcs} (canales {', '.join(str(channel_start + int(s)) for s in v_src_local_indices)})"),
        ("Migración", "Receptores", f"{len(rec_local_indices)}, uno cada {real.correlacion.receptores_paso} canales"),
        ("Migración", "Filtro F-K aplicado a los registros", f"{fk_vmin:g} a {fk_vmax:g} m/s"),
    ]

    print("\nGenerando documento editable Word/Google Docs...")
    try:
        doc_file = build_das_guide_docx(output_dir, parameters=parameters)
        print(f"Documento de guía generado con éxito: '{doc_file}'")
    except Exception as e:
        print(f"Advertencia al generar documento .docx: {e}")

    print(f"\nProcesamiento DAS terminado. Resultados en '{output_dir}'.")
