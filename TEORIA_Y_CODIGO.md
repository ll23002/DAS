# Manual Completo: Teoría y Código de TSW-RTM sobre DAS
*Para programadores e ingenieros de software (sin conocimientos previos de física o sismología)*

---

## 1. La Gran Analogía: ¿Qué estamos construyendo en código?

Imagina el siguiente problema de visión por computadora o medicina:
Quieres hacer una **tomografía o ultrasonido del subsuelo** para mapear una grieta geológica profunda (una **falla tectónica**), pero tienes dos restricciones críticas:
1. **No puedes meter fuentes activas:** No puedes detonar dinamita, ni usar martillos hidráulicos, ni esperar a que haya un terremoto destructivo.
2. **Tu sensor es un cable de fibra óptica:** Tienes enterrado un cable de telecomunicaciones estándar a lo largo de una carretera.

¿De dónde sacas energía para iluminar el subsuelo? **Del ruido ambiental**.
La Tierra nunca está en silencio: el oleaje del mar chocando en la costa, camiones y autos en la carretera, turbinas eólicas, fábricas y personas caminando generan una vibración constante y caótica (ruido de fondo).

El software que tienes en este repositorio toma ese ruido caótico, extrae mediante **procesamiento digital de señales (DSP)** las ondas coherentes, convierte matemáticamente los canales del cable en "emisores virtuales", y luego ejecuta un **motor de simulación de ondas en reversa (Reverse Time Migration, RTM)** acelerado en C con OpenMP para enfocar la imagen de la falla.

---

## 2. Mapa de Arquitectura del Software

El pipeline completo sigue este flujo lineal de datos:

```
[ Archivo HDF5 (DAS crudo) ]
         │
         ▼  (tsw_rtm/preprocessing.py)
[ Preprocesamiento: Decimación + Filtro Pasabanda + Normalización 1-bit + Whitening ]
         │
         ▼  (tsw_rtm/interferometry.py)
[ Interferometría de Ruido: Correlación Cruzada (FFT) + Promedio acumulativo (Stacking) ]
         │
         ▼  (tsw_rtm/interferometry.py)
[ Función de Green Empírica (Derivada dC/dt) -> Virtual Common Shot Gather ]
         │
         ▼  (tsw_rtm/preprocessing.py)
[ Filtro 2D F-K: Eliminar ruido y retener velocidades aparentes (800 a 3500 m/s) ]
         │
         ▼  (tsw_rtm/velocity_flood.py)
[ Generación de Modelos 'Flood': Universo Sedimento (vs) y Universo Roca (vg) ]
         │
         ▼  (tsw_rtm/wave_solver.py + c_engine/libwave2d.so)
[ RTM: Simulación Forward (vs) + Retropropagación Backward (vg) + Correlación cruzada ]
         │
         ▼  (tsw_rtm/rtm_engine.py + visualizer.py)
[ Postprocesamiento: Normalización de Iluminación + Filtro Laplaciano (Bordes) -> Imagen Final + CSV ]
```

---

## 3. Desglose Módulo por Módulo: Teoría vs. Implementación

### Módulo A: La Entrada de Datos (DAS e I/O)
*Archivos:* `tsw_rtm/das_io.py`, `run_das_pipeline.py`

#### 1. ¿Qué es físicamente un cable DAS para un programador?
Un equipo interrogador envía pulsos láser ultra rápidos por el cable de fibra óptica. Cada vez que el suelo vibra, estira o encoge microscópicamente el vidrio de la fibra, alterando la fase de la luz retrodispersada.
El interrogador calcula la **tasa de deformación** (*Strain Rate*).

#### 2. Estructura de Datos en RAM y Disco
En los archivos `.h5` (HDF5), el dataset `Acoustic` es simplemente una **matriz 2D de punto flotante** de dimensiones `(N_muestras_tiempo, N_canales_espacio)`:
- **Eje 0 (Filas = Tiempo):** Con una frecuencia de muestreo de $500\text{ Hz}$, se toma una lectura cada $dt = 0.002\text{ s}$. Un archivo de 60 segundos tiene $60 \times 500 = 30,000$ filas.
- **Eje 1 (Columnas = Espacio):** Cada columna es un "canal" (un punto fijo en el cable). La separación entre canales adyacentes es $dx \approx 5.133\text{ m}$. Si el cable tiene $10,891$ canales, cubre casi $56\text{ km}$ de longitud.
- **Ventana de Procesamiento:** En `run_das_pipeline.py`, procesar $56\text{ km}$ sería computacionalmente inviable. Por eso se define una ventana local:
  ```python
  # Canales 1200 al 1800 (600 canales = ~3.08 km de cable)
  data, time_raw, dist_raw = reader.read_window(
      channel_min=channel_start,
      channel_max=channel_start + num_channels
  )
  ```

---

### Módulo B: Preprocesamiento de Señales (DSP)
*Archivo:* `tsw_rtm/preprocessing.py`

El ruido crudo no se puede meter directo al algoritmo porque contiene ruidos electromagnéticos, derivas térmicas del láser y eventos locales gigantescos (un camión pasando justo encima de un canal).

```
Dato Crudo ──> [Decimate (500Hz -> 50Hz)] ──> [Bandpass (0.8 - 4.5Hz)] ──> [1-bit (-1, 0, +1)] ──> [Spectral Whitening]
```

#### 1. Decimación Temporal (`decimate_time`)
- **Analogía de software:** Bajar la resolución de un video de 500 FPS a 50 FPS para ahorrar memoria y CPU.
- **El peligro del Aliasing:** Si simplemente tomas una de cada 10 muestras (`data[::10]`), las frecuencias de alta vibración se "disfrazarán" matemáticamente de frecuencias bajas (como cuando las ruedas de un auto en video parecen girar al revés).
- **La solución en código:** Primero se aplica un filtro paso bajo (Butterworth) a la frecuencia de Nyquist nueva ($f_{nyq} = 50 / 2 = 25\text{ Hz}$) y recién después se hace el submuestreo:
  ```python
  filtered = signal.sosfiltfilt(sos, data, axis=0)
  subsampled = filtered[::down_factor, :]
  ```

#### 2. Filtro Pasabanda (`bandpass_filter`)
- **Por qué:** Las ondas sísmicas de superficie que viajan por el subsuelo y que pueden resolver una falla a cientos de metros de profundidad se concentran entre **$0.8\text{ Hz}$ y $4.5\text{ Hz}$**. Todo lo demás (el zumbido eléctrico de 60 Hz, el ruido del viento de 20 Hz, las mareas de 0.05 Hz) es basura y se descarta con un filtro Butterworth digital de orden 4.
- **Cero desplazamiento de fase (`sosfiltfilt`):** En sismología el tiempo exacto de llegada de un frente de onda lo es todo. Un filtro normal retrasa la señal. `sosfiltfilt` filtra el arreglo hacia adelante y luego en reversa, cancelando cualquier desfase temporal.

#### 3. Normalización Temporal de 1-Bit (`temporal_normalization(method='1bit')`)
- **Concepto radical:**
  ```python
  return np.sign(data).astype(np.float32)
  ```
  Convierte cada muestra a $+1.0$ (si era positiva), $-1.0$ (si era negativa) o $0.0$. Destruye el 100% de la información de volumen o amplitud.
- **¿Por qué un programador haría esto?:** En interferometría de ruido no nos interesa qué tan fuerte fue una vibración, sino **la fase temporal** (cuándo cruza por cero). Si un micro-sismo o un tráiler genera amplitudes $1,000$ veces mayores que el ruido de fondo, sesgaría todo el promedio matemático. El filtro de 1-bit democratiza todas las muestras para que solo importe la correlación temporal.

#### 4. Blanqueamiento Espectral (`spectral_whitening`)
- **El problema de las resonancias:** Si hay una bomba de agua o un motor cerca del cable vibrando constantemente a $2.7\text{ Hz}$, esa sola frecuencia monopolizaría la correlación cruzada con una onda senoidal eterna.
- **La solución:** Pasar la señal al dominio de Fourier (`rfft`), calcular la envolvente del espectro de amplitud, y dividir el espectro real por dicha envolvente:
  $$\text{Dato\_Blanqueado}(f) = \frac{\text{Dato}(f)}{|\text{Dato}(f)| + \epsilon}$$
  Así, todas las frecuencias dentro del rango $[0.8, 4.5]\text{ Hz}$ tienen exactamente la misma energía (espectro plano, como el ruido blanco).

---

### Módulo C: Interferometría Sísmica (El "Disparo Virtual")
*Archivo:* `tsw_rtm/interferometry.py`

Este es el núcleo matemático del método: **convertir dos micrófonos pasivos en un emisor y un receptor activos**.

```
Canal A (Micrófono) ──┐
                      ├──> Correlación Cruzada (Stack de 9 archivos) ──> Derivada (d/dt) ──> [Función de Green G_AB]
Canal B (Micrófono) ──┘                                                                       (Como si A disparara y B grabara)
```

#### 1. ¿Cómo funciona la Correlación Cruzada ($C_{AB}$)?
La correlación cruzada mide qué tanto se parece la señal registrada en el canal A con la registrada en el canal B si desplazamos una con respecto a la otra un retraso $\tau$ (*lag*):
$$C_{AB}(\tau) = \int_{-\infty}^{\infty} \text{Señal}_A(t) \cdot \text{Señal}_B(t + \tau) \, dt$$

- **Ejemplo intuitivo:**
  Imagina que en medio de la estática viaja una onda a $2,000\text{ m/s}$. El canal A está en $x=0\text{ m}$ y el canal B en $x=2,000\text{ m}$.
  Cualquier fluctuación aleatoria del ruido que pase por A llegará a B exactamente **$1.0\text{ segundo}$ después**.
  Si multiplicas la señal de A por la señal de B desplazada por $\tau$:
  - En $\tau = 0.2\text{ s}$ se multiplican números aleatorios $\to$ el promedio da $0$.
  - En $\tau = 1.0\text{ s}$ se multiplican los mismos picos con los mismos valles $\to$ el producto da un valor positivo gigante.

- **Optimización algorítmica (Teorema de Convolución):**
  Calcular esto en bucles temporales $O(N^2)$ sería lentísimo. Se calcula en $O(N \log N)$ en el dominio de Fourier mediante FFT:
  ```python
  # En el dominio frecuencial: la correlación es el producto por el conjugado complejo
  cross_spec = np.conjugate(src_fft)[:, np.newaxis] * rec_fft
  cc_full = np.fft.irfft(cross_spec, n=n_fft, axis=0)
  ```

#### 2. Stacking (Promedio Acumulativo)
Con un solo archivo de 1 minuto, el ruido no es perfectamente estocástico y se ven patrones caóticos. Al apilar (promediar) decenas de archivos:
$$\overline{C_{AB}}(\tau) = \frac{1}{N} \sum_{i=1}^N C_{AB}^{(i)}(\tau)$$
El ruido incoherente se cancela por interferencia destructiva y el pulso coherente que viaja de A a B emerge progresivamente (la relación señal/ruido crece con $\sqrt{N}$).

#### 3. La Función de Green Empírica y la Derivada Temporal
En física matemática, la **Función de Green** ($G(A, B, t)$) es la respuesta al impulso de un sistema: **la señal que registraría el sensor B si detonáramos una fuente impulsiva perfecta (martillazo) en el punto A en el tiempo $t=0$**.

El teorema fundamental de la interferometría sísmica (Lobkis & Weaver, 2001; Wapenaar, 2004) demuestra que la derivada temporal de la correlación cruzada es proporcional a la función de Green:
$$\frac{d}{dt} C_{AB}(t) \approx -G(A, B, t) + G(A, B, -t)$$
- Tiempos positivos ($t > 0$, rama causal): La onda propagándose de A hacia B.
- Tiempos negativos ($t < 0$, rama anticausal): La onda en reversa de B hacia A.

En `build_virtual_shot_gather`:
```python
processed_ccfs = np.gradient(ccfs, dt, axis=0) # Derivada temporal
gather = processed_ccfs[zero_idx:, :]          # Tomamos solo la rama causal
```
Ahora tenemos un **Virtual Common Shot Gather**: una matriz idéntica a la que obtendríamos en una campaña de sísmica activa tradicional de petróleo o minería.

---

### Módulo D: El Dominio F-K (Filtrado Bidimensional)
*Archivo:* `tsw_rtm/preprocessing.py` (función `fk_filter`)

Un gather sísmico es una imagen con dos dimensiones: **Tiempo $t$** y **Offset/Distancia $x$**.

```
Dominio Espacio-Tiempo (t, x)  ──[ 2D FFT ]──>  Dominio Frecuencia-Número de onda (f, k)
(Ondas como hipérbolas/líneas)                   (Ondas con velocidad v = f/k son rayos que salen del origen)
```

1. **¿Qué es el número de onda $k$?**
   Así como la frecuencia temporal $f$ mide ciclos por segundo ($\text{Hz} = 1/\text{s}$), el número de onda $k$ mide **ciclos por metro** ($1/\text{m}$). Está relacionado con la longitud de onda espacial: $k = 1/\lambda$.
2. **La Velocidad Aparente como una Pendiente:**
   Puesto que $\text{Velocidad} = \text{Distancia}/\text{Tiempo} = \lambda \cdot f = f / k$, en una gráfica 2D donde el eje Y es $f$ y el eje X es $k$, **cualquier onda que viaje a una velocidad constante $v$ es simplemente una línea recta con pendiente $v$**.
3. **El Filtro:**
   En el subsuelo viajan muchas cosas:
   - Ondas P (muy rápidas: $3,000 - 5,000\text{ m/s}$, pendientes muy empinadas).
   - Ruido ambiental no propagativo (llega a todos los canales al mismo tiempo: $k \approx 0$, velocidad aparente $\infty$, línea vertical en $k=0$).
   - **Ondas superficiales de Rayleigh (nuestro objetivo):** Viajan lentas, típicamente entre **$800\text{ m/s}$ y $3,500\text{ m/s}$**.
   El filtro F-K aplica una máscara en forma de cuña o abanico que apaga todo lo que esté fuera de ese rango de pendientes y luego regresa al dominio espacio-temporal mediante `ifft2`.

---

### Módulo E: El Truco de la "Inundación de Velocidad" (Velocity Flood)
*Archivo:* `tsw_rtm/velocity_flood.py`

Este es el concepto más ingenioso del paper de Li et al. (2020) y resuelve un problema fundamental de la imagenología sísmica:

#### El Dilema
Una falla tectónica es una fractura geológica que pone en contacto dos tipos de roca:
- **Lado Izquierdo (Sedimentos blandos):** Velocidad baja ($v_s \approx 1,100 - 1,500\text{ m/s}$).
- **Lado Derecho (Roca ígnea/metamórfica dura):** Velocidad alta ($v_g \approx 2,400 - 2,800\text{ m/s}$).

Si intentamos simular numéricamente la propagación de ondas usando un modelo que ya tiene esa discontinuidad dibujada, el simulador generará reflexiones artificiales en la falla que destruirán la correlación de la imagen.

#### La Solución: Dos "Universos Falsos" Continuos
Construimos dos modelos de velocidad artificiales donde la falla ha sido eliminada por completo mediante **"inundación lateral"**:

```
Modelo Real:            [ Sedimento (1100 m/s) | Falla | Roca Metamórfica (2400 m/s) ]

1. Sediment Flood (vs): [ Sedimento (1100 m/s) ---------> INUNDADO A LA DERECHA (1100 m/s) ]
   (Se usa para simular la fuente hacia adelante)

2. Rock Flood (vg):     [ INUNDADO A LA IZQUIERDA (2400 m/s) <--------- Roca Metamórfica (2400 m/s) ]
   (Se usa para retropropagar los receptores en reversa)
```

- En el **Universo Sedimento (`vs`)**, la onda de la fuente virtual se propaga de izquierda a derecha sin enterarse de que la roca existe.
- En el **Universo Roca (`vg`)**, los datos grabados por los receptores se inyectan en reversa temporal y viajan de derecha a izquierda sin enterarse de que el sedimento existe.
- **La magia:** En la naturaleza real, la onda sísmica transmitida cambió de velocidad **justo al atravesar la falla**. En el software, la trayectoria hacia adelante calculada con $v_s$ y la trayectoria en reversa calculada con $v_g$ **coincidirán en tiempo y espacio en un único lugar: el plano tridimensional de la falla**.

---

### Módulo F: Reverse Time Migration (RTM) y Simulación Física
*Archivos:* `tsw_rtm/wave_solver.py`, `tsw_rtm/rtm_engine.py`, `tsw_rtm/c_engine/wave2d.c`

#### 1. El Motor Numérico FDTD (Diferencias Finitas en Dominio del Tiempo)
Para propagar las ondas numéricamente, resolvemos la ecuación de onda escalar 2D:
$$\frac{\partial^2 p}{\partial t^2} = v(x, z)^2 \left( \frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial z^2} \right)$$
En código (`c_engine/wave2d.c`), las derivadas espaciales continuas se discretizan en una rejilla rectangular $nx \times nz$ con stencil de diferencias finitas de segundo orden:
```c
float d2p_dx2 = (p_curr[iz*nx + (ix+1)] - 2.0f*p_curr[iz*nx + ix] + p_curr[iz*nx + (ix-1)]) / (dx * dx);
float d2p_dz2 = (p_curr[(iz+1)*nx + ix] - 2.0f*p_curr[iz*nx + ix] + p_curr[(iz-1)*nx + ix]) / (dz * dz);

// Integración temporal (Salto de rana / Leapfrog):
p_next[iz*nx + ix] = 2.0f * p_curr[iz*nx + ix] - p_prev[iz*nx + ix] 
                     + (vel * vel) * (dt * dt) * (d2p_dx2 + d2p_dz2);
```

#### 2. La Condición de Estabilidad CFL (Courant-Friedrichs-Lewy)
Un simulador numérico explícito puede volverse inestable y explotar matemáticamente a valores `NaN` / $\infty$ si el paso de tiempo $dt$ es demasiado grande para la cuadrícula espacial.
El método `check_stability` valida la condición CFL:
$$CFL = v_{max} \cdot dt \cdot \sqrt{\frac{1}{dx^2} + \frac{1}{dz^2}} \le 0.6$$
Si $dx \approx 10.26\text{ m}$ y $v_{max} = 2,600\text{ m/s}$, el código exige que $dt \le 0.0015\text{ s}$ para garantizar estabilidad absoluta.

#### 3. Fronteras Absorbentes (Esponja de Cerjan)
Una simulación computacional ocurre en una caja finita. Si la onda choca con el borde de la matriz, rebotará hacia adentro creando ecos fantasmas. En los bordes laterales e inferior se aplica una capa esponjosa de 20 celdas que atenúa exponencialmente la energía:
```c
p_next[iz*nx + ix] *= sponge[iz*nx + ix]; // Factor < 1.0 en las orillas
```
La superficie superior ($z=0$) no tiene esponja: tiene una condición de contorno de **superficie libre** ($p=0$), tal como la superficie real de la Tierra que refleja las ondas hacia abajo.

#### 4. La Condición de Imagen (Zero-Lag Cross-Correlation)
¿Cómo se forma la foto de la falla?
Para cada disparo virtual:
1. Se corre la simulación hacia adelante desde la fuente $S(x, z, t)$ y se guardan snapshots periódicos en memoria.
2. Se toma el registro del gather virtual de los receptores, **se invierte el eje del tiempo** (se corre la película hacia atrás), y se inyecta como fuente en las coordenadas de los receptores $R(x, z, t)$.
3. En cada píxel de la cuadrícula $(x, z)$, se integran los dos campos multiplicados simultáneamente:
   $$Image(x, z) = \sum_{t=0}^T S(x, z, t) \cdot R(x, z, t)$$

#### 5. Postprocesamiento de la Imagen
- **Normalización por Iluminación:** Cerca de la superficie donde están los sensores, la energía de la onda es millones de veces mayor que a 1 km de profundidad. Se divide la imagen por la raíz cuadrada de la energía acumulada de la fuente ($\sqrt{\sum S^2}$) para balancear el contraste en profundidad.
- **Filtro Laplaciano 2D (`apply_laplacian_filter`):** Actúa como un detector de bordes de convolución (análogo a un filtro Sobel en OpenCV):
  $$\text{Laplaciano} = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial z^2}$$
  Elimina las manchas borrosas de baja frecuencia (los lóbulos anchos de la onda Ricker) y colapsa la energía en una línea delgada y definida donde se encuentra la falla.

---

## 4. Diagnóstico de los Datos Actuales (Guatemala M4.1)

Al procesar los 9 archivos de Guatemala (`results_m41_guatemala`), observamos resultados que difieren del caso ideal sintético. He aquí la explicación desde la perspectiva de datos y señales:

```
Espectro F-K Actual (Guatemala):
  Frecuencia (Hz)
     5 │         │ (k ≈ 0)
     4 │─────────┼─────────  <-- Línea horizontal de resonancia (~4.0 Hz)
     3 │─────────┼─────────  <-- Línea horizontal de resonancia (~2.7 Hz)
     2 │         │
     1 │─────────┼─────────  <-- Línea horizontal de resonancia (~1.3 Hz)
     0 └─────────┴─────────
       -k        0        +k (Número de onda)
```

1. **Bandas Horizontales en las Correlaciones Cruzadas (`5_ccf_src*.png`):**
   - **Lo esperado:** Frentes de onda inclinados (efecto Doppler / moveout lineal), donde a mayor distancia (offset), mayor tiempo de llegada ($t = x / v$).
   - **Lo obtenido:** Bandas perfectamente horizontales e independientes de la distancia.
   - **Causa:** En el análisis F-K se observan líneas espectrales monocromáticas persistentes en $\sim 1.3\text{ Hz}$, $2.7\text{ Hz}$ y $4.0\text{ Hz}$ con $k \approx 0$. Esto corresponde a una vibración armónica en fase a lo largo de toda la fibra (por ejemplo, vibración inducida en la subestación eléctrica de Acajutla o resonancia del equipo interrogador). La correlación de una onda senoidal pura es otra onda senoidal infinita.
2. **Discontinuidades en bloques por offset:**
   En los gráficos de correlación se aprecian saltos bruscos verticales a distancias de $\sim 1.2\text{ km}$, $1.65\text{ km}$, $2.1\text{ km}$. Esto suele coincidir con empalmes de la fibra óptica, pasos de ducto aéreo a subterráneo, o cambios en el material del suelo donde la fibra está enterrada.
3. **¿Se observa el sismo M4.1?**
   El archivo número 8 (`..._1012.h5`, 08:30:05) registra un salto abrupto en la métrica `band_power` ($24,859$ frente al promedio de $\sim 9,000$ de los archivos previos) y el valor de pico absoluto de deformación salta a $33.8$. Esto es la firma directa del paquete de ondas del sismo llegando a la red.

---

## 5. Referencia Rápida de Archivos CSV Generados

Para cualquier análisis posterior en Python, Excel o MATLAB, todos los archivos en `results_m41_guatemala/csv/` siguen el estándar:
- **Celda `[0, 0]`:** Nombre del eje Y / Nombre del eje X (`fila_nombre / columna_nombre`).
- **Fila 0 (Encabezados):** Coordenadas del eje horizontal X.
- **Columna 0:** Coordenadas del eje vertical Y.
- **Matriz interior:** Valores numéricos en coma flotante.

| Archivo CSV | Dimensión | Eje Y (Filas) | Eje X (Columnas) | Contenido de los Datos |
|---|---|---|---|---|
| `1a_raw_das.csv` | $2000 \times 600$ | Tiempo ($0 \to 2.0\text{ s}$) | Distancia cable ($0 \to 3.07\text{ km}$) | Tasa de deformación cruda ($\mu\epsilon/\text{s}$) |
| `1b_preprocessed_das.csv` | $100 \times 600$ | Tiempo ($0 \to 2.0\text{ s}$) | Distancia cable ($0 \to 3.07\text{ km}$) | Señal normalizada (1-bit + Whitening) |
| `2_virtual_shot_gather.csv` | $176 \times 95$ | Tiempo viaje ($0 \to 3.5\text{ s}$) | Offset ($0.92 \to 2.85\text{ km}$) | Disparo virtual de la Fuente 0 (filtrado F-K) |
| `3_true_vel.csv` | $120 \times 300$ | Profundidad ($0 \to 1.2\text{ km}$) | Distancia ($0 \to 3.08\text{ km}$) | Modelo de velocidad de referencia ($\text{m/s}$) |
| `3_sediment_flood.csv` | $120 \times 300$ | Profundidad ($0 \to 1.2\text{ km}$) | Distancia ($0 \to 3.08\text{ km}$) | Modelo inundado de baja velocidad ($v_s$) |
| `3_rock_flood.csv` | $120 \times 300$ | Profundidad ($0 \to 1.2\text{ km}$) | Distancia ($0 \to 3.08\text{ km}$) | Modelo inundado de alta velocidad ($v_g$) |
| `4_migrated_fault_image.csv` | $120 \times 300$ | Profundidad ($0 \to 1.2\text{ km}$) | Distancia ($0 \to 3.08\text{ km}$) | Imagen RTM final con filtro Laplaciano |
| `5_ccf_src[0-3].csv` | $351 \times 95$ | Retardo Lag ($-3.5 \to +3.5\text{ s}$) | Offset ($0.25 \to 2.85\text{ km}$) | Matriz de correlación cruzada apilada $C(t)$ |
| `5_green_src[0-3].csv` | $176 \times 95$ | Tiempo ($0 \to 3.5\text{ s}$) | Offset ($0.25 \to 2.85\text{ km}$) | Función de Green empírica ($dC/dt$) |
| `6_fk_file[0-8].csv` | $2048 \times 1024$ | Frecuencia ($0 \to 250\text{ Hz}$) | Wavenumber ($-0.097 \to +0.097\text{ m}^{-1}$) | Espectro de potencia 2D F-K en decibeles ($\text{dB}$) |
| `6_fk_metrics_per_file.csv`| $9 \times 9$ | Índice del archivo `.h5` | Columnas de métricas | Métricas estadísticas de energía, RMS y picos |
