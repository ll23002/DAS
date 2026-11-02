import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


def main() -> int:
    os.chdir(BASE_DIR)
    try:
        from tsw_rtm.config import load_config, ConfigError
    except ModuleNotFoundError as exc:
        print(f"Falta instalar un paquete ({exc.name}). Use ./ejecutar.sh para instalarlo automáticamente.")
        return 1

    try:
        cfg = load_config(CONFIG_PATH)
    except ConfigError as exc:
        print(f"\nError en la configuración: {exc}")
        print("Corrija el archivo config.yaml y vuelva a ejecutar.")
        return 1

    start = time.time()
    try:
        if "real" in cfg.modo:
            from run_das_pipeline import run_pipeline
            run_pipeline(cfg)
        if "sintetico" in cfg.modo:
            from run_synthetic_benchmarks import run_synthetic
            run_synthetic(cfg)
    except FileNotFoundError as exc:
        print(f"\nNo se encontró un archivo necesario: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nEjecución cancelada.")
        return 1

    print(f"\nTodo terminado en {(time.time() - start) / 60:.1f} minutos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
