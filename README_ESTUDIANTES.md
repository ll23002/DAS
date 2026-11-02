# Guía para estudiantes

Este programa toma el ruido ambiental registrado por un cable de fibra óptica (DAS) y produce imágenes de una falla geológica. No necesita saber programar.

## 1. Requisitos (una sola vez)

- **Linux (Ubuntu)**. En Windows, use el Subsistema de Windows para Linux (WSL) con Ubuntu.
- En una terminal de Ubuntu, instale lo básico:

```
sudo apt install python3 python3-venv python3-pip build-essential
```

## 2. Preparar los datos

Copie los archivos `.h5` en una carpeta, por ejemplo `~/Downloads/DAS/M4.1_Guatemala`.

## 3. Elegir los valores

Abra `config.yaml` con cualquier editor de texto. Cada línea tiene una explicación a la derecha. Lo más habitual es cambiar solo:

| Parámetro | Para qué sirve |
|---|---|
| `modo` | `real` (datos de fibra), `sintetico` (modelos de prueba) o ambos |
| `real.datos` | Carpeta con los datos |
| `real.resultados` | Carpeta donde se guardan los resultados |
| `real.cable.canal_inicial` y `num_canales` | Tramo del cable que se analiza |
| `real.filtrado.frecuencia_min_hz` y `frecuencia_max_hz` | Banda de frecuencias de interés |

Guarde el archivo. Si algo está mal escrito, el programa se lo indicará con un mensaje claro.

## 4. Ejecutar

En la terminal, dentro de la carpeta del proyecto:

```
./ejecutar.sh
```

La primera vez instala lo necesario y tarda unos minutos. Después solo procesa.

## 5. Resultados

Se guardan en la carpeta indicada en `resultados`: figuras (`.png`) y tablas (`.csv`).

| Archivo | Qué muestra |
|---|---|
| `1_das_qc_preprocessing.png` | Datos crudos y datos preparados |
| `6_fk_per_file.png` | Espectro frecuencia–número de onda de cada archivo |
| `5_ccf_green_src*.png` | Correlación cruzada, función de Green de −T a +T (partes acausal y causal) y la forma de onda a la par |
| `2_virtual_shot_gather.png` | Registro de fuente virtual |
| `3_das_flood_models.png` | Modelos de velocidad usados |
| `4_das_migrated_fault_image.png` | Imagen final de la falla |

En la carpeta `sintetico` se guardan las imágenes de los casos de prueba, cada una con la falla verdadera marcada en rojo.

## Si algo falla

- **"No se encontró la carpeta o archivo de datos"**: revise `real.datos`.
- **"La frecuencia mínima debe ser menor que la máxima"**: corrija esos dos valores.
- **"Falta el compilador"**: ejecute `sudo apt install build-essential`.
