"""Loads and validates config.yaml, reporting problems in plain Spanish."""

import os
from types import SimpleNamespace

import yaml


class ConfigError(Exception):
    pass


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _require(section: dict, name: str, path: str, kinds):
    kinds = kinds if isinstance(kinds, tuple) else (kinds,)
    if name not in section:
        raise ConfigError(f"Falta el parámetro '{path}{name}' en config.yaml")
    value = section[name]
    if isinstance(value, bool) and bool not in kinds:
        raise ConfigError(f"'{path}{name}' debe ser un número, no verdadero/falso")
    if not isinstance(value, kinds):
        raise ConfigError(f"'{path}{name}' tiene un valor no válido: {value!r}")
    return value


def _positive(section: dict, names, path: str):
    for name in names:
        if _require(section, name, path, (int, float)) <= 0:
            raise ConfigError(f"'{path}{name}' debe ser mayor que cero")


def _validate_real(real: dict):
    p = "real."
    _require(real, "datos", p, str)
    _require(real, "resultados", p, str)
    _require(real, "max_archivos", p, (int, type(None)))

    datos = os.path.expanduser(real["datos"])
    if not os.path.exists(datos):
        raise ConfigError(f"No se encontró la carpeta o archivo de datos: {real['datos']}")

    cable = _require(real, "cable", p, dict)
    _positive(cable, ["num_canales"], p + "cable.")
    if _require(cable, "canal_inicial", p + "cable.", int) < 0:
        raise ConfigError("'real.cable.canal_inicial' no puede ser negativo")

    filtrado = _require(real, "filtrado", p, dict)
    _positive(filtrado, ["frecuencia_muestreo_hz", "frecuencia_min_hz", "frecuencia_max_hz"], p + "filtrado.")
    if filtrado["frecuencia_min_hz"] >= filtrado["frecuencia_max_hz"]:
        raise ConfigError("La frecuencia mínima debe ser menor que la máxima (real.filtrado)")
    if filtrado["frecuencia_max_hz"] >= filtrado["frecuencia_muestreo_hz"] / 2:
        raise ConfigError("La frecuencia máxima debe ser menor que la mitad de la frecuencia de muestreo")

    corr = _require(real, "correlacion", p, dict)
    _positive(corr, ["retardo_max_s", "num_fuentes_virtuales", "receptores_paso"], p + "correlacion.")
    offset = _require(corr, "offset_forma_onda_km", p + "correlacion.", (int, float, str))
    if isinstance(offset, str) and offset != "auto":
        raise ConfigError("'real.correlacion.offset_forma_onda_km' debe ser 'auto' o un número en km")

    fk = _require(real, "filtro_fk", p, dict)
    _positive(fk, ["velocidad_min_ms", "velocidad_max_ms"], p + "filtro_fk.")
    if fk["velocidad_min_ms"] >= fk["velocidad_max_ms"]:
        raise ConfigError("La velocidad mínima debe ser menor que la máxima (real.filtro_fk)")

    modelo = _require(real, "modelo_velocidad", p, dict)
    _positive(modelo, ["num_celdas_profundidad", "paso_profundidad_m", "sedimento_velocidad_ms",
                       "roca_velocidad_ms", "celdas_inundacion"], p + "modelo_velocidad.")
    _require(modelo, "sedimento_gradiente", p + "modelo_velocidad.", (int, float))
    _require(modelo, "roca_gradiente", p + "modelo_velocidad.", (int, float))
    _require(modelo, "sedimento_a_la_izquierda", p + "modelo_velocidad.", bool)

    mig = _require(real, "migracion", p, dict)
    _positive(mig, ["paso_tiempo_s", "frecuencia_ricker_hz", "celdas_esponja", "intervalo_instantaneas"],
              p + "migracion.")


def _validate_sintetico(sint: dict):
    p = "sintetico."
    _require(sint, "resultados", p, str)
    _positive(sint, ["paso_tiempo_s", "duracion_s", "frecuencia_ricker_hz", "celdas_esponja",
                     "celdas_inundacion", "num_fuentes", "num_receptores", "intervalo_instantaneas"], p)
    malla = _require(sint, "malla", p, dict)
    _positive(malla, ["celdas_x", "celdas_z", "paso_x_m", "paso_z_m"], p + "malla.")

    casos = _require(sint, "casos", p, list)
    if not casos:
        raise ConfigError("'sintetico.casos' no puede estar vacío")
    for i, caso in enumerate(casos, 1):
        if not isinstance(caso, dict):
            raise ConfigError(f"El caso {i} de 'sintetico.casos' no es válido")
        for campo in ("nombre", "tipo", "falla_x"):
            if campo not in caso:
                raise ConfigError(f"Al caso {i} de 'sintetico.casos' le falta '{campo}'")
        if caso["tipo"] not in ("vertical", "inclinada"):
            raise ConfigError(f"El tipo del caso '{caso['nombre']}' debe ser 'vertical' o 'inclinada'")
        if caso["tipo"] == "inclinada" and "angulo" not in caso:
            raise ConfigError(f"Al caso '{caso['nombre']}' le falta 'angulo'")
        if not 0 < caso["falla_x"] < malla["celdas_x"]:
            raise ConfigError(f"'falla_x' del caso '{caso['nombre']}' debe estar dentro de la malla")


def load_config(path: str) -> SimpleNamespace:
    if not os.path.exists(path):
        raise ConfigError(f"No se encontró el archivo de configuración: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"El archivo config.yaml tiene un error de formato: {exc}")
    if not isinstance(raw, dict):
        raise ConfigError("El archivo config.yaml está vacío o mal formado")

    modos = _require(raw, "modo", "", list)
    if not modos or any(m not in ("real", "sintetico") for m in modos):
        raise ConfigError("'modo' solo puede contener 'real' y/o 'sintetico'")
    if "real" in modos:
        _validate_real(_require(raw, "real", "", dict))
    if "sintetico" in modos:
        _validate_sintetico(_require(raw, "sintetico", "", dict))
    return _to_namespace(raw)
