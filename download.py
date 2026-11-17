#!/usr/bin/env python3
"""
Script: download.py
Descripción: Descarga archivos DAS (FBE y/o Strain Rate) desde el almacenamiento
             S3 (RustFS) hacia las carpetas locales correspondientes:
               - FBE:         bucket 'acajutla-fbe'        -> ./FBE/
               - Strain Rate: bucket 'acajutla-strainrate' -> ./StrainRate/
             Mantiene la estructura de carpetas original (YYYY-MM-DD/HH/archivo.h5).
             Usa Python nativo con boto3 y descargas multihilo en paralelo.

Instalación de dependencias:
  pip install -r requirements.txt

Uso básico:
  python download.py                             # Menú interactivo
  python download.py fbe                         # Descargar FBE
  python download.py strain                      # Descargar Strain Rate
  python download.py both                        # Descargar ambos canales

Descarga por hora específica:
  python download.py strain 2026-09-30 08        # Fecha y hora como argumentos
  python download.py strain 2026-09-30 --hour 08 # Con bandera --hour o --hora
  python download.py strain "2026-09-30 08"      # En una sola cadena
  python download.py fbe 2026-11-07 00           # FBE a la hora 00

Otros filtros (día completo, rango o todo):
  python download.py strain 2026-09-30           # Día completo
  python download.py strain "2026-09-30 08:20" "2026-09-30 08:35" # Rango de minutos
  python download.py strain all                  # Todo el bucket
  python download.py strain 2026-09-30 08 --dry-run # Simulación
"""

import sys
import os
import re
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    print("=" * 65)
    print("[ERROR] Falta la librería 'boto3'. Instálala ejecutando:")
    print("    pip install -r requirements.txt")
    print("=" * 65)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ==============================================================================
# CONFIGURACIÓN DEL SERVIDOR S3 (RUSTFS)
# ==============================================================================
S3_ENDPOINT = "http://s3-rustfs-80a06b-168-232-48-211.sslip.io"
S3_ACCESS_KEY = "ivzKS2E95WJRDI2JEenC"
S3_SECRET_KEY = "5sF3cji90Jg6WhRTDTEIcFXcuxhUFgWCE6VOFIlh"

SCRIPT_DIR = Path(__file__).resolve().parent

CHANNELS = {
    "fbe": {
        "name": "FBE",
        "bucket": "acajutla-fbe",
        "default_folder": "FBE"
    },
    "strain": {
        "name": "Strain Rate",
        "bucket": "acajutla-strainrate",
        "default_folder": "StrainRate"
    }
}


def get_s3_client():
    """Crea y retorna un cliente S3 thread-safe configurado para RustFS con path-style."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(
            s3={"addressing_style": "path"},
            signature_version="s3v4",
            connect_timeout=15,
            read_timeout=60,
            retries={"max_attempts": 5, "mode": "standard"}
        )
    )


def norm_ts(val: str, is_end: bool = False) -> str:
    """Convierte cadenas de fecha/hora como '2026-11-07 10:30' a '20261107103000'."""
    val = val.strip()
    digits = re.sub(r"\D", "", val)
    if len(digits) == 8:       # YYYYMMDD
        return digits + ("235959" if is_end else "000000")
    elif len(digits) == 10:    # YYYYMMDDHH
        return digits + ("5959" if is_end else "0000")
    elif len(digits) == 12:    # YYYYMMDDHHMM
        return digits + ("59" if is_end else "00")
    elif len(digits) >= 14:    # YYYYMMDDHHMMSS...
        return digits[:14]
    else:
        raise ValueError(f"Formato no reconocido: '{val}'. Usa YYYY-MM-DD [HH:MM[:SS]].")


def fmt_ts(ts: str) -> str:
    """Formatea '20261107103000' a '2026-11-07 10:30:00'."""
    if len(ts) == 14:
        return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:10]}:{ts[10:12]}:{ts[12:14]}"
    return ts


def list_available_dates(s3, bucket: str):
    """Lista las carpetas de fecha de primer nivel en el bucket (ej: 2026-11-07)."""
    paginator = s3.get_paginator("list_objects_v2")
    dates = []
    for page in paginator.paginate(Bucket=bucket, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            d = cp.get("Prefix", "").strip("/")
            if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
                dates.append(d)
    return sorted(dates)


def find_objects(s3, bucket: str, start_input=None, end_input=None, hour_input=None, all_flag=False):
    """
    Busca los objetos .h5 a descargar optimizando las consultas según los filtros:
      - all_flag: todo el bucket
      - hour_input: hora específica (ej: 08) en una fecha
      - start_input solo (8 dígitos): día completo
      - start_input y end_input: rango de tiempo
    Retorna una lista de tuplas: (key, size).
    """
    paginator = s3.get_paginator("list_objects_v2")

    # 1. Modo ALL (todo el bucket)
    if all_flag or (not start_input and not end_input and hour_input is None):
        print(f"Consultando todos los archivos de {bucket} (puede tardar unos segundos)...")
        matched = []
        for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".h5"):
                    matched.append((key, obj["Size"]))
        matched.sort(key=lambda x: x[0])
        return matched, "Todos los archivos disponibles"

    # 2. Modo HORA ESPECÍFICA (ej: 2026-09-30 hora 08)
    if hour_input is not None:
        try:
            h = int(re.sub(r"\D", "", str(hour_input)))
            if not (0 <= h <= 23):
                raise ValueError()
            hour_str = f"{h:02d}"
        except Exception:
            raise ValueError(f"Hora inválida: '{hour_input}'. Debe ser un número entre 00 y 23.")

        digits_d = re.sub(r"\D", "", start_input)
        if len(digits_d) >= 8:
            date_str = f"{digits_d[:4]}-{digits_d[4:6]}-{digits_d[6:8]}"
        else:
            raise ValueError(f"Fecha inválida para la hora indicada: '{start_input}'. Usa YYYY-MM-DD.")

        prefix = f"{date_str}/{hour_str}/"
        desc = f"Fecha {date_str} a la hora {hour_str}:00 (Prefijo: {prefix})"
        print(f"Consultando archivos para {desc} en {bucket}...")
        matched = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".h5"):
                    matched.append((key, obj["Size"]))
        matched.sort(key=lambda x: x[0])
        return matched, desc

    # 3. Fecha completa sin hora ni fin: YYYY-MM-DD
    digits_s = re.sub(r"\D", "", start_input)
    if len(digits_s) == 8 and not end_input:
        date_str = f"{digits_s[:4]}-{digits_s[4:6]}-{digits_s[6:8]}"
        desc = f"Fecha completa: {date_str}"
        print(f"Consultando archivos para {desc} en {bucket}...")
        matched = []
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{date_str}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".h5"):
                    matched.append((key, obj["Size"]))
        matched.sort(key=lambda x: x[0])
        return matched, desc

    # 4. Rango de fecha/hora o inicio específico
    start_ts = norm_ts(start_input, is_end=False)
    if end_input:
        end_ts = norm_ts(end_input, is_end=True)
        if start_ts > end_ts:
            raise ValueError(f"La fecha de inicio ({fmt_ts(start_ts)}) es posterior a la de fin ({fmt_ts(end_ts)}).")
        desc = f"{fmt_ts(start_ts)}  -->  {fmt_ts(end_ts)}"
    else:
        end_ts = "99999999999999"
        desc = f"Desde {fmt_ts(start_ts)} en adelante"

    start_d = f"{start_ts[:4]}-{start_ts[4:6]}-{start_ts[6:8]}"
    end_d = f"{end_ts[:4]}-{end_ts[4:6]}-{end_ts[6:8]}" if end_ts != "99999999999999" else "9999-99-99"

    print(f"Identificando fechas relevantes en {bucket}...")
    all_dates = list_available_dates(s3, bucket)
    target_dates = [d for d in all_dates if start_d <= d <= end_d]

    if not target_dates:
        return [], desc

    print(f"Consultando archivos en {len(target_dates)} fecha(s) de {bucket}...")
    matched = []
    for d in target_dates:
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{d}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".h5"):
                    continue
                m = re.search(r"(\d{8})T(\d{6})", key)
                if m:
                    file_ts = m.group(1) + m.group(2)
                    if start_ts <= file_ts <= end_ts:
                        matched.append((key, obj["Size"]))

    matched.sort(key=lambda x: x[0])
    return matched, desc


def download_single_file(s3_client_factory, bucket: str, key: str, size: int, dest_path: Path, overwrite: bool = False) -> tuple:
    """Descarga un archivo individual asegurando la creación de subcarpetas."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Si el archivo ya existe con el mismo tamaño, no volver a descargarlo
    if not overwrite and dest_path.exists() and dest_path.stat().st_size == size:
        return (key, size, "skipped")

    tmp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")
    client = s3_client_factory()
    try:
        client.download_file(bucket, key, str(tmp_dest))
        tmp_dest.replace(dest_path)
        return (key, size, "downloaded")
    except Exception as e:
        if tmp_dest.exists():
            tmp_dest.unlink(missing_ok=True)
        return (key, size, f"error: {e}")


def process_download_channel(channel_key: str, start_input, end_input, hour_input, all_flag, base_dest: Path, workers: int, dry_run: bool, overwrite: bool):
    """Ejecuta el flujo completo de consulta y descarga para un canal específico."""
    info = CHANNELS[channel_key]
    channel_name = info["name"]
    bucket = info["bucket"]
    folder_name = info["default_folder"]

    # Determinar ruta destino exacta
    if base_dest.name.lower() in ("fbe", "strainrate"):
        target_dir = base_dest
    else:
        target_dir = base_dest / folder_name

    s3 = get_s3_client()

    print("\n" + "=" * 65)
    print(f"PREPARANDO CANAL: {channel_name.upper()} (Bucket: {bucket})")
    print("=" * 65)

    try:
        objects, desc = find_objects(s3, bucket, start_input, end_input, hour_input, all_flag)
    except Exception as e:
        print(f"[ERROR en {channel_name}] {e}", file=sys.stderr)
        return False

    if not objects:
        print(f"\n[INFO] No se encontraron archivos para {channel_name} en el rango/hora indicada.")
        return True

    total_bytes = sum(size for _, size in objects)
    total_mb = total_bytes / (1024 * 1024)

    print("\n" + "-" * 65)
    print(f"RESUMEN DE DESCARGA: {channel_name}")
    print(f"  Bucket S3:     {bucket} ({S3_ENDPOINT})")
    print(f"  Filtro:        {desc}")
    print(f"  Archivos:      {len(objects)} archivo(s)")
    print(f"  Tamaño total:  {total_mb:.2f} MB (~{total_mb / 1024:.2f} GB)")
    print(f"  Carpeta local: {target_dir}")
    print(f"  Hilos:         {workers} en paralelo")
    print(f"  Modo Dry-Run:  {dry_run}")
    print("-" * 65)

    if len(objects) <= 6:
        for k, _ in objects:
            print(f"  - {k}")
    else:
        for k, _ in objects[:3]:
            print(f"  - {k}")
        print(f"  ... ({len(objects) - 6} archivos más) ...")
        for k, _ in objects[-3:]:
            print(f"  - {k}")

    if dry_run:
        print(f"\n[AVISO] Modo --dry-run activo. No se descargaron archivos de {channel_name}.")
        return True

    print(f"\nIniciando descarga de {channel_name}...")

    downloaded_count = 0
    skipped_count = 0
    error_count = 0

    progress_bar = tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc=f"Progreso {channel_name}"
    ) if tqdm else None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_single_file,
                get_s3_client,
                bucket,
                key,
                size,
                target_dir / key,
                overwrite
            ): (key, size)
            for key, size in objects
        }

        for future in as_completed(futures):
            key, size = futures[future]
            try:
                _, fsize, status = future.result()
                if status == "downloaded":
                    downloaded_count += 1
                elif status == "skipped":
                    skipped_count += 1
                else:
                    error_count += 1
                    print(f"\n[ERROR] Falló {key}: {status}", file=sys.stderr)
            except Exception as e:
                error_count += 1
                print(f"\n[ERROR] Excepción en {key}: {e}", file=sys.stderr)

            if progress_bar:
                progress_bar.update(size)
            elif (downloaded_count + skipped_count) % 20 == 0:
                print(f"[{channel_name}] Descargados: {downloaded_count + skipped_count}/{len(objects)} archivos...")

    if progress_bar:
        progress_bar.close()

    print("\n" + "-" * 65)
    print(f"FINALIZADO: {channel_name}")
    print(f"  Descargados nuevos:      {downloaded_count}")
    if skipped_count:
        print(f"  Omitidos (ya existían):  {skipped_count}")
    if error_count:
        print(f"  Errores de descarga:     {error_count}")
    print(f"  Archivos guardados en:   {target_dir}")
    print("-" * 65)
    return error_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Descargador de archivos DAS (FBE y Strain Rate) desde S3 RustFS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Descargar una hora en específico:
  python download.py strain 2026-09-30 08
  python download.py strain 2026-09-30 --hour 08
  python download.py fbe 2026-11-07 00

  # Otros filtros:
  python download.py strain 2026-09-30
  python download.py strain "2026-09-30 08:20" "2026-09-30 08:35"
  python download.py both 2026-09-30 08
  python download.py fbe all --workers 8
  python download.py strain 2026-09-30 08 --dry-run
        """
    )
    parser.add_argument("posicionales", nargs="*", help="[fbe | strain | both] [FECHA] [HORA o FIN]")
    parser.add_argument("--hour", "--hora", type=str, default=None, help="Hora específica a descargar (00 a 23, ej: --hour 08)")
    parser.add_argument("--fbe", action="store_true", help="Descargar canal FBE")
    parser.add_argument("--strain", action="store_true", help="Descargar canal Strain Rate")
    parser.add_argument("--both", action="store_true", help="Descargar ambos canales (FBE y Strain Rate)")
    parser.add_argument("--dest", default=str(SCRIPT_DIR), help=f"Carpeta base de destino (por defecto: {SCRIPT_DIR})")
    parser.add_argument("--workers", type=int, default=6, help="Número de descargas simultáneas en paralelo (por defecto: 6)")
    parser.add_argument("--dry-run", action="store_true", help="Simula la descarga sin guardar archivos reales")
    parser.add_argument("--overwrite", action="store_true", help="Vuelve a descargar archivos aunque ya existan en disco")

    parsed = parser.parse_args()

    selected_channel = None
    if parsed.fbe:
        selected_channel = "fbe"
    elif parsed.strain:
        selected_channel = "strain"
    elif parsed.both:
        selected_channel = "both"

    start_input = None
    end_input = None
    hour_input = parsed.hour
    all_flag = False

    args = list(parsed.posicionales)

    # 1. Detectar si el primer argumento es el tipo de canal
    if args:
        first_lower = args[0].lower().strip()
        if first_lower in ("fbe",):
            selected_channel = "fbe"
            args.pop(0)
        elif first_lower in ("strain", "strainrate", "strain_rate", "sr"):
            selected_channel = "strain"
            args.pop(0)
        elif first_lower in ("both", "ambos", "todos_canales"):
            selected_channel = "both"
            args.pop(0)

    # 2. Parsear fechas/horas/rangos de los argumentos restantes
    if args:
        arg_val = args[0].strip()
        if arg_val.lower() in ("all", "todos", "todo"):
            all_flag = True
        else:
            # Comprobar si viene en formato combinado "YYYY-MM-DD HH" (ej: "2026-09-30 08")
            m_combined = re.match(r"^(\d{4}-\d{2}-\d{2})[ T](\d{1,2})$", arg_val)
            if m_combined:
                start_input = m_combined.group(1)
                if hour_input is None:
                    hour_input = m_combined.group(2)
            else:
                start_input = arg_val
                if len(args) > 1:
                    val_next = args[1].strip()
                    # Si el segundo argumento es solo 1 o 2 dígitos entre 0 y 23, es una hora
                    if re.match(r"^\d{1,2}$", val_next) and 0 <= int(val_next) <= 23:
                        if hour_input is None:
                            hour_input = val_next
                    else:
                        end_input = val_next

    # 3. Menú interactivo si no se especificó canal y corre en terminal interactiva
    if not selected_channel:
        if sys.stdin.isatty():
            print("=" * 65)
            print("          DESCARGADOR DE DATOS DAS (S3 RustFS)")
            print("============================================================")
            print("Selecciona el tipo de datos a descargar:")
            print("  1) FBE          (bucket: acajutla-fbe        -> ./FBE/)")
            print("  2) Strain Rate  (bucket: acajutla-strainrate -> ./StrainRate/)")
            print("  3) Ambos        (FBE y Strain Rate)")
            print("  4) Salir")
            print("=" * 65)
            try:
                opc = input("Opción [1-4]: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nCancelado.")
                sys.exit(0)

            if opc == "1":
                selected_channel = "fbe"
            elif opc == "2":
                selected_channel = "strain"
            elif opc == "3":
                selected_channel = "both"
            else:
                print("Operación cancelada.")
                sys.exit(0)

            if not start_input and not all_flag and hour_input is None:
                print("\nFiltro de fecha / tiempo:")
                print("  1) Descargar TODO el bucket")
                print("  2) Descargar una FECHA completa (ej: 2026-09-30)")
                print("  3) Descargar una HORA específica (ej: 2026-09-30 a las 08)")
                print("  4) Descargar un RANGO de fecha/hora (ej: 2026-09-30 08:20 a 08:35)")
                print("  5) Salir")
                try:
                    opc_f = input("Opción [1-5]: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nCancelado.")
                    sys.exit(0)

                if opc_f == "1":
                    all_flag = True
                elif opc_f == "2":
                    start_input = input("Ingresa la fecha (formato YYYY-MM-DD): ").strip()
                elif opc_f == "3":
                    start_input = input("Ingresa la fecha (formato YYYY-MM-DD): ").strip()
                    hour_input = input("Ingresa la hora (00-23, ej: 08): ").strip()
                elif opc_f == "4":
                    start_input = input("Fecha/Hora de INICIO (ej: 2026-09-30 08:00): ").strip()
                    end_input = input("Fecha/Hora de FIN    (ej: 2026-09-30 09:00, Enter para solo inicio): ").strip()
                else:
                    print("Cancelado.")
                    sys.exit(0)
        else:
            selected_channel = "fbe"
            all_flag = True

    base_dest = Path(parsed.dest).resolve()

    channels_to_run = ["fbe", "strain"] if selected_channel == "both" else [selected_channel]

    for ch in channels_to_run:
        process_download_channel(
            channel_key=ch,
            start_input=start_input,
            end_input=end_input,
            hour_input=hour_input,
            all_flag=all_flag,
            base_dest=base_dest,
            workers=parsed.workers,
            dry_run=parsed.dry_run,
            overwrite=parsed.overwrite
        )

    print("\n" + "=" * 65)
    print("PROCESO GENERAL DE DESCARGA COMPLETADO")
    print("=" * 65)


if __name__ == "__main__":
    main()
