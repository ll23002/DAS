# FUNDAMENTOS TEÓRICOS Y ARQUITECTURA DEL SISTEMA DAS-RTM
## Tomografía Pasiva de Ruido Sísmico e Interferometría de Ondas Superficiales con Cable de Fibra Óptica (DAS) y Migración en Reversa en el Tiempo (TSW-RTM)

---

## 0. INTRODUCCIÓN GENERAL Y VISIÓN FÍSICA GLOBAL

Este documento expone de forma exhaustiva los fundamentos teóricos, matemáticos y computacionales implementados en la plataforma de software `tsw_rtm`. El objetivo primordial del código es resolver un problema clásico y desafiante de la geofísica aplicada: **obtener una imagen de alta resolución espacial de discontinuidades tectónicas profundas (planos de falla)** utilizando exclusivamente:

1. **Sensores distribuidos pasivos:** Un cable de fibra óptica para telecomunicaciones enterrado en superficie o trinchera somera, interrogado mediante tecnología **DAS** (*Distributed Acoustic Sensing*).
2. **Fuentes naturales no controladas:** Ruido sísmico ambiental (*ambient seismic noise* generado por oleaje marino, tránsito vehicular pesado, microtremores y actividad urbana) en lugar de explosivos (dinamita) o camiones vibradores activos (*vibroseis*).
3. **Propagación de ondas superficiales transmitidas:** En lugar de reflexiones acústicas tradicionales de ondas de cuerpo ($P$ y $S$), el sistema explota las ondas de Rayleigh guiadas en la superficie libre.
4. **Migración en Reversa en el Tiempo con Inundación de Velocidad (*TSW-RTM* - *Transmitted Surface Wave Reverse Time Migration*):** Algoritmo basado en la formulación de Li et al. (2020, *Journal of Geophysical Research: Solid Earth*), resolviendo numéricamente la ecuación de onda en dos sentidos temporales (adelante y atrás) con un motor acelerado en lenguaje C con paralelismo OpenMP.

El desglose de este documento sigue estrictamente la estructura y las figuras del reporte técnico generado en Word (`RESULTADOS_DAS.docx` / `Guia_Laboratorio_DAS.docx`).

---

## 1. CONTROL DE CALIDAD Y PREPROCESAMIENTO DE SEÑALES (DSP)

### 1.1 FÍSICA DEL SENSOR DAS (*DISTRIBUTED ACOUSTIC SENSING*)
A diferencia de los sismómetros mecánicos tradicionales (geófonos o acelerómetros de masa suspendida en un punto discreto), el sistema DAS transforma una fibra óptica convencional en un arreglo denso y continuo de miles de sensores de deformación.

#### Principio de Retrodispersión de Rayleigh (*Rayleigh Backscattering*)
Un equipo interrogador optoelectrónico emite periódicamente pulsos de luz láser coherente de ancho temporal estrecho $\tau_p$ a lo largo del núcleo de sílice de la fibra. A medida que el pulso luminoso viaja, interactúa con fluctuaciones submicrométricas microscópicas e inhomogeneidades en el índice de refracción inherentes al proceso de manufactura del vidrio. Esta interacción produce **dispersión de Rayleigh**, haciendo que una fracción diminuta de la luz se propague en sentido inverso hacia el interrogador (*luz retrodispersada*).

Cuando una onda sísmica cruza el cable, induce tensiones elásticas en el terreno que se transmiten al cable, provocando compresiones y elongaciones microscópicas de la fibra en la coordenada espacial $x$. La longitud de deformación entre dos secciones reflectantes separadas por una distancia característica denominada **longitud de calibración** o *gauge length* ($L_g$, comúnmente $8 - 10\text{ m}$) experimenta una alteración $\Delta L$:
$$\Delta \phi(x, t) = \frac{4\pi n}{\lambda_0} \Delta L(x, t)$$
donde:
- $n \approx 1.468$ es el índice de refracción del núcleo de sílice.
- $\lambda_0$ es la longitud de onda de la portadora láser (típicamente $1550\text{ nm}$ en la banda C de telecomunicaciones).
- $\Delta \phi$ es el desfase óptico medido mediante demodulación interferométrica de fase óptica.

#### Deformación y Tasa de Deformación (*Strain* vs *Strain Rate*)
La deformación longitudinal promedio (*average longitudinal strain*) $\epsilon_{xx}$ se define como:
$$\epsilon_{xx}(x, t) = \frac{\Delta L(x, t)}{L_g} = \frac{\lambda_0}{4\pi n L_g} \Delta \phi(x, t)$$

Dado que la medición se actualiza a una frecuencia de muestreo $f_s$ muy rápida (frecuentemente $200 - 1000\text{ Hz}$), el sistema registra directamente la derivada temporal de la deformación, conocida como **tasa de deformación** (*strain rate*, $\dot{\epsilon}_{xx}$):
$$\dot{\epsilon}_{xx}(x, t) = \frac{\partial \epsilon_{xx}(x, t)}{\partial t} = \frac{\partial^2 u_x(x, t)}{\partial x \partial t} = \frac{\partial v_x(x, t)}{\partial x}$$
donde $u_x$ es el desplazamiento elástico horizontal de las partículas de la tierra paralelo a la trayectoria del cable, y $v_x = \frac{\partial u_x}{\partial t}$ es la velocidad de partícula registrada habitualmente por geófonos convencionales.

> **Propiedad Física Clave:** El cable DAS no mide la velocidad de partícula $v_x$ pura, sino su **gradiente espacial horizontal** ($\frac{\partial v_x}{\partial x}$). Para una onda que se propaga a una velocidad aparente $v_{app}$, por la relación de onda viajera $u_x(t - x/v_{app})$:
> $$\frac{\partial v_x}{\partial x} = -\frac{1}{v_{app}} \frac{\partial v_x}{\partial t} = -\frac{1}{v_{app}} a_x(t)$$
> Por ende, la tasa de deformación es proporcional a la aceleración de partícula dividida por la velocidad aparente de la onda. Las ondas más lentas (como las ondas superficiales de Rayleigh) generan gradientes espaciales mucho mayores y son registradas con amplitudes intrínsecamente más altas en DAS que las ondas $P$ rápidas.

---

### 1.2 ETAPAS DEL PREPROCESAMIENTO DIGITAL DE SEÑALES (DSP)

Los registros brutos de DAS contienen perturbaciones indeseadas: derivas térmicas ambientales en la fibra (de frecuencias $< 0.1\text{ Hz}$), acoplamiento de vibraciones del interrogador, ruido eléctrico a 50/60 Hz, perturbaciones no propagativas locales (microdeformaciones térmicas) y sismos o tráileres gigantescos cuyas amplitudes enmascaran el ruido microbárico de fondo. Para acondicionar los datos previo a la interferometría, el módulo `tsw_rtm/preprocessing.py` ejecuta una cadena de filtrado determinista y robusto:

```
[ Datos Brutos DAS ] (Fs = 200 Hz, Strain Rate)
         │
         ▼
[ 1. Decimación Temporal con Anti-Aliasing ] ──► Reducción a Fs = 50 Hz (Nyquist = 25 Hz)
         │
         ▼
[ 2. Filtro Pasabanda Butterworth Cero-Fase ] ──► Aislamiento de 0.8 a 4.5 Hz (sosfiltfilt)
         │
         ▼
[ 3. Normalización Temporal de 1-Bit ] ──► Destrucción de amplitud, conservación estricta de fase
         │
         ▼
[ 4. Blanqueamiento Espectral en Frecuencia ] ──► Aplanamiento de la densidad de potencia espectral
         │
         ▼
[ Señal Normalizada Lista para Correlación Cruzada ]
```

#### 1. Decimación Temporal y Teorema de Muestreo de Nyquist-Shannon
Si una señal con frecuencia de muestreo original $f_{s, orig} = 200\text{ Hz}$ (límite de Nyquist $f_{nyq} = 100\text{ Hz}$) se submuestrea directamente tomando una de cada cuatro muestras sin prefiltrar, cualquier componente energética por encima del nuevo límite de Nyquist ($f_{nyq, new} = 50 / 2 = 25\text{ Hz}$) se plegará sobre la banda audible según la relación de aliasing:
$$f_{alias} = |f - k \cdot f_{s, target}|$$
Por ejemplo, un ruido de motor a $35\text{ Hz}$ aparecería artificialmente como una oscilación ficticia a $|35 - 50| = 15\text{ Hz}$, contaminando el espectro de dispersión.

Para erradicar el aliasing, la función `decimate_time` aplica primero un filtro pasabajas Butterworth digital de 4º orden con frecuencia de corte al 80% del nuevo Nyquist ($f_c \le 0.8 \cdot 25 = 20\text{ Hz}$):
$$H(s) H(-s) = \frac{1}{1 + (-1)^N \left(\frac{s}{\omega_c}\right)^{2N}}$$
Implementado mediante **Secciones de Segundo Orden (SOS - *Second-Order Sections*)** para mitigar la inestabilidad de redondeo numérico en coma flotante inherente a polinomios de coeficientes directos:
$$H(z) = \prod_{k=1}^K \frac{b_{0k} + b_{1k} z^{-1} + b_{2k} z^{-2}}{1 + a_{1k} z^{-1} + a_{2k} z^{-2}}$$

#### 2. Filtro Pasabanda Butterworth de Cero Desplazamiento de Fase (`sosfiltfilt`)
Las ondas superficiales de Rayleigh útiles para caracterizar fallas en profundidades someras a intermedias ($0 - 1.2\text{ km}$) tienen su mayor densidad de energía coherente entre **$0.8\text{ Hz}$ y $4.5\text{ Hz}$**. Las frecuencias inferiores a $0.8\text{ Hz}$ están dominadas por derivas térmicas y mareas terrestres; las superiores a $4.5\text{ Hz}$ presentan fuerte atenuación elástica intrínseca ($Q_s$) y dispersión difusa en la capa meteorizada más superficial.

**Condición Crítica de Cero Fase:** En sismología e interferometría, el tiempo de retraso $\tau$ define distancias exactas. Un filtro digital IIR convencional introduciría un desfase dependiente de la frecuencia $\theta(f)$, provocando un retardo de grupo:
$$\tau_g(f) = -\frac{1}{2\pi} \frac{d\theta(f)}{df} \neq 0$$
Este retardo desplazaría artificialmente la velocidad aparente calculada. La función `sosfiltfilt` filtra los datos hacia adelante en el tiempo y posteriormente invierte la serie temporal y la vuelve a filtrar hacia atrás:
$$y(t) = f(-t) * (f(t) * x(t)) \quad \Longleftrightarrow \quad Y(\omega) = |H(\omega)|^2 X(\omega)$$
El espectro de fase resultante es idénticamente cero ($\angle Y(\omega) = \angle X(\omega) + \theta(\omega) - \theta(\omega) = 0$), garantizando una fidelidad temporal absoluta en los tiempos de arribo.

#### 3. Normalización Temporal de 1-Bit
Los métodos clásicos de correlación cruzada son cuadráticos: si un camión de volteo de 30 toneladas o un micro-sismo local cruza por encima de los canales 1300-1350 generando amplitudes $1000$ veces mayores que el ruido de microseísmo oceánico constante, el producto cruzado estará monopolizado en un $99.99\%$ por ese evento aislado, violando la hipótesis de estacionariedad del ruido difuso.

La función `temporal_normalization(method='1bit')` reemplaza cada muestra por su función signo:
$$y[n, x] = \text{sgn}(x[n, x]) = \begin{cases} +1.0 & \text{si } x[n, x] > 0 \\ 0.0 & \text{si } x[n, x] = 0 \\ -1.0 & \text{si } x[n, x] < 0 \end{cases}$$
**Fundamento Físico:** En interferometría de ruido no importa la amplitud del movimiento (energía de la fuente), sino la **fase de onda coherente** (los cruces por cero que definen cuándo un frente de fase viaja del canal A al canal B). La normalización de 1-bit elimina por completo los picos gigantes destructivos y homogeniza uniformemente el registro temporal.

#### 4. Blanqueamiento Espectral (*Spectral Whitening*)
Si existe una maquinaria o generador industrial cercano que vibre ininterrumpidamente a una frecuencia tonal pura (por ejemplo, $f_0 = 2.4\text{ Hz}$), la correlación cruzada $C_{AB}(\tau)$ produciría una sinusoide armónica eterna $\cos(2\pi f_0 \tau)$ que enmascararía los pulsos impulsivos de banda ancha de las ondas de Rayleigh.

La función `spectral_whitening` transforma cada canal al dominio frecuencial mediante FFT real (`rfft`):
$$X(f) = \mathcal{F}\{x(t)\}$$
Calcula la envolvente espectral alisada mediante una convolución en frecuencia con una ventana móvil rectangular de ancho $B$ (*boxcar filter*):
$$A_{smooth}(f) = \frac{1}{B} \int_{f - B/2}^{f + B/2} |X(\nu)| \, d\nu$$
Y divide el espectro original por su envolvente, normalizando la amplitud a la unidad dentro de la banda de paso $[f_{min}, f_{max}]$:
$$X_{white}(f) = \frac{X(f)}{A_{smooth}(f) + \epsilon} \cdot W(f)$$
donde $W(f)$ es una ventana de Hann que suaviza las transiciones en $f_{min}$ y $f_{max}$ para prevenir oscilaciones de Gibbs al volver al dominio del tiempo mediante la transformada inversa $\mathcal{F}^{-1}$. De esta forma, todas las componentes frecuenciales del ruido aportan con idéntico peso estadístico.

---

### 1.3 INTERPRETACIÓN DE LA FIGURA 1.1: DATOS BRUTOS VS. PREPROCESADOS
La imagen generada por el software (`1_das_qc_preprocessing.png`) muestra la comparación lado a lado en un diagrama de cascada espacio-temporal (*waterfall plot*):

![Figura 1.1: Control de calidad del preprocesamiento DAS](file:///home/alexander/Fisica/resultados/real/1_das_qc_preprocessing.png)

- **Panel Izquierdo (Raw DAS Strain Rate):** 
  - El eje horizontal representa la distancia relativa a lo largo de la fibra óptica en kilómetros ($0.0$ a $\sim 6.16\text{ km}$, cubriendo 600 canales con espaciamiento de $\approx 10.27\text{ m}$).
  - El eje vertical representa el tiempo en segundos ($0$ a $10\text{ s}$).
  - Se aprecian bandas verticales continuas e inhomogéneas que corresponden a variaciones de acoplamiento local del cable (tramos donde el cable está empotrado en asfalto rígido frente a tramos en tierra suelta o aire en arquetas de empalme), y componentes de alta frecuencia electromagnética que velan los frentes de onda sísmica coherentes.
- **Panel Derecho (Preprocessed DAS):**
  - Tras la decimación a $50\text{ Hz}$, el filtrado $0.8 - 4.5\text{ Hz}$, la cuantificación de 1-bit y el blanqueamiento, las bandas estáticas verticales han desaparecido.
  - El fondo se observa plano y balanceado en todas las trazas espaciales.
  - Emergen claramente **alineaciones diagonales coherentes** con pendientes bien definidas, que corresponden a frentes de onda sísmica reales que se desplazan longitudinalmente a través de la carretera.

---

## 2. IDENTIFICACIÓN Y ANÁLISIS DE ESPECTROS DE DISPERSIÓN

### 2.1 ANÁLISIS FRECUENCIA-NÚMERO DE ONDA ($f-k$)

#### Transformada de Fourier Espacio-Temporal Bidimensional (2D FT)
Un registro sísmico DAS multicanal es un campo bidimensional discreto $u(t, x)$ donde $t$ es el tiempo continuo y $x$ es la posición espacial a lo largo del cable. La transformada de Fourier bidimensional proyecta este campo hacia el dominio frecuencia-número de onda:
$$U(f, k) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} u(t, x) \, e^{-i 2\pi (f t - k x)} \, dt \, dx$$
donde:
- $f$ es la **frecuencia temporal**, medida en Hertz ($\text{Hz} = \text{s}^{-1}$), que cuantifica las oscilaciones por unidad de tiempo.
- $k$ es el **número de onda espacial**, medido en ciclos por metro ($1/\text{m}$) o ciclos por kilómetro ($1/\text{km}$), que cuantifica las oscilaciones espaciales por unidad de longitud ($k = 1/\lambda$, donde $\lambda$ es la longitud de onda).

#### Velocidad Aparente como Pendiente en el Plano $f-k$
Para una onda monocromática plana que se propaga a una velocidad aparente de fase $v_{app}$:
$$x(t) = v_{app} \cdot t \implies \lambda \cdot f = v_{app} \implies \frac{f}{k} = v_{app}$$
En una gráfica 2D con la frecuencia $f$ en un eje y el número de onda $k$ en el otro:
$$f = v_{app} \cdot k \quad \Longleftrightarrow \quad k = \frac{1}{v_{app}} f$$
Cualquier evento ondulatorio que viaje a velocidad constante traza **una línea recta radial que pasa por el origen $(0, 0)$**. La pendiente de dicha recta define la velocidad aparente:
- Ondas ultra rápidas ($P$ y $S$ reflejadas verticalmente): Llegan a casi todos los sensores simultáneamente ($k \approx 0$, longitud de onda infinita, pendiente $v_{app} \to \infty$, línea vertical).
- Ruido ambiental no propagativo (ruido de modo común, perturbaciones eléctricas): Concentrado en el eje $k = 0$.
- **Ondas superficiales de Rayleigh:** Viajan relativamente lentas a través de los estratos geológicos ($800\text{ m/s} \le v_{app} \le 3500\text{ m/s}$), proyectándose en una región cónica o abanico característico.

#### Direccionalidad de Propagación ($+k$ vs $-k$)
La convención de signo en el exponente de Fourier distingue el sentido vectorial del avance de la onda:
- Si el producto $f \cdot k > 0$ (cuadrante de número de onda positivo $+k$): Las ondas viajan en la dirección positiva del cable (de canal menor a mayor, por ejemplo de izquierda a derecha).
- Si el producto $f \cdot k < 0$ (cuadrante de número de onda negativo $-k$): Las ondas viajan en la dirección opuesta (de derecha a izquierda).
La asimetría de energía entre ambos lados del espectro $f-k$ revela directamente la procedencia azimutal de las fuentes naturales dominantes de ruido sísmico.

---

### INTERPRETACIÓN DE LA FIGURA 2.1: ESPECTRO DE DISPERSIÓN $f-k$ Y CRESTA DE ENERGÍA

![Figura 2.1: Espectro de dispersión f-k con número de onda k vs frecuencia f](file:///home/alexander/Fisica/resultados/real/2_1_fk_dispersion_c.png)

La Figura 2.1 (`2_1_fk_dispersion_c.png`) representa el espectro de potencia promedio acumulado calculado a partir del submuestreo de banda ancha (hasta $100\text{ Hz}$):
- **Eje X:** Frecuencia $f$ ($0$ a $100\text{ Hz}$).
- **Eje Y:** Número de onda $k$ ($0$ a $25\text{ km}^{-1}$).
- **Líneas Guía Discontinuas:** 
  - La recta punteada superior corresponde a la trayectoria teórica $k = f / 1.0$, indicando una velocidad lenta de **$1.0\text{ km/s}$ ($1000\text{ m/s}$)**.
  - La recta punteada inferior corresponde a $k = f / 4.0$, indicando una velocidad rápida de **$4.0\text{ km/s}$ ($4000\text{ m/s}$)**.
- **Cresta de Energía (*Energy Ridge*):**
  Se distingue con extraordinaria nitidez una banda hiperbólica densa y continua de color oscuro/cálido confinada con precisión matemática entre las dos líneas de referencia ($1\text{ km/s}$ y $4\text{ km/s}$). Esta cresta representa la curva de dispersión de las ondas de Rayleigh de la cuenca geológica. A bajas frecuencias ($< 10\text{ Hz}$), la energía se aproxima a la cota de alta velocidad ($2.5 - 3.0\text{ km/s}$) porque muestrea la roca cristalina profunda; a frecuencias medias y altas ($20 - 80\text{ Hz}$), la cresta se reclina hacia $1.0 - 1.2\text{ km/s}$, dominada por los sedimentos no consolidados superficiales.

---

### INTERPRETACIÓN DE LAS FIGURAS 2.1.1, 2.1.2 Y 2.1.3: ESPECTROS $f-k$ POR LOTES DE ARCHIVOS

![Figura 2.1.1: Espectros f-k individuales archivos 1 a 3](file:///home/alexander/Fisica/resultados/real/6_fk_batch1.png)
![Figura 2.1.2: Espectros f-k individuales archivos 4 a 6](file:///home/alexander/Fisica/resultados/real/6_fk_batch2.png)
![Figura 2.1.3: Espectros f-k individuales archivos 7 a 9](file:///home/alexander/Fisica/resultados/real/6_fk_batch3.png)

Estas figuras desglosan el comportamiento espectral individual para cada uno de los archivos HDF5 registrados a lo largo del tiempo:
- Permiten verificar la **estacionariedad temporal** del ruido ambiental.
- Se observa que la cresta de Rayleigh entre $800\text{ m/s}$ y $3500\text{ m/s}$ es persistente a través de todos los registros independientes, lo que valida que el campo ondulatorio no obedece a un evento esporádico aislado, sino a un régimen acústico difuso sostenido y óptimo para interferometría sísmica.

---

### 2.2 ESPECTROS FRECUENCIA-VELOCIDAD ($f-v$) Y MODELO 1D DE VELOCIDAD DE ONDA DE CORTE ($V_s$)

#### Física de la Dispersión de Ondas de Rayleigh
Las ondas de Rayleigh son ondas elásticas superficiales generadas por la interferencia elíptica acoplada entre ondas compresionales longitudinales ($P$) y ondas de cizalla vertical ($SV$) en presencia de una frontera de esfuerzo nulo (la superficie libre de la Tierra). En un medio homogéneo semi-infinito, la onda de Rayleigh no es dispersiva; su velocidad $v_R$ es constante y ligeramente menor que la velocidad de onda de corte $V_s$:
$$v_R \approx \frac{0.87 + 1.12 \nu}{1 + \nu} V_s \approx 0.92 \, V_s \quad (\text{para un sólido de Poisson con } \nu = 0.25)$$

Sin embargo, en la Tierra real, las propiedades mecánicas (módulo de cizalla $\mu$ y densidad $\rho$) aumentan con la profundidad debido a la compactación litoestática y a los cambios litológicos (sedimentos que suprayacen roca dura):
$$V_s(z) = \sqrt{\frac{\mu(z)}{\rho(z)}}$$

La amplitud del movimiento elíptico retrógrado de una onda de Rayleigh decae exponencialmente con la profundidad según una función de sensibilidad $\xi(z, \omega)$ (ecuación 3 de Li et al., 2020):
$$u_s(x, z, t, \omega) \approx \xi(z, \omega) \exp\left[ i \omega \left( t - \frac{x}{C(\omega)} \right) \right]$$
La profundidad efectiva de penetración de la onda superficial está gobernada por su longitud de onda:
$$z_{sensibilidad} \approx \frac{\lambda}{2} = \frac{v_R(f)}{2 f}$$
- **Altas frecuencias ($f > 20\text{ Hz}$):** Longitudes de onda cortas ($\lambda < 50\text{ m}$). Su energía está confinada exclusivamente a los primeros metros de la corteza, propagándose a la velocidad lenta de los sedimentos superficiales ($1000 - 1200\text{ m/s}$).
- **Bajas frecuencias ($f < 3\text{ Hz}$):** Longitudes de onda largas ($\lambda > 800\text{ m}$). La energía penetra a profundidades superiores a $400 - 1000\text{ m}$, donde la deformación involucra a la roca cristalina consolidada, propagándose a velocidades elevadas ($2000 - 2800\text{ m/s}$).
Por lo tanto, la velocidad de fase $C(f)$ es función directa de la frecuencia: **la onda es dispersiva**.

#### Transformación al Dominio $f-v$ (`compute_fv_spectrum`)
Para transformar la densidad de potencia espectral $P(f, k)$ al dominio de frecuencia-velocidad de fase $P(f, v)$, se aprovecha la transformación de coordenadas $k = f / v$:
$$P_{fv}(f, v) = P_{fk}\left(f, \, k = \frac{f}{v}\right)$$
La función discretiza un arreglo regular de velocidades $v \in [0.5, 4.0]\text{ km/s}$ con $N_v = 180$ puntos y remuestrea mediante interpolación unidimensional los valores de potencia correspondientes para cada frecuencia $f$.

#### Modos Normales y Modos de Fuga (*Normal Modes* vs *Leaky Modes*)
En el espectro $f-v$ emergen dos familias físicas de ramas energéticas:
1. **Modos Normales (*Normal Modes*):** Ondas guiadas atrapadas en la capa de baja velocidad debido a reflexión interna total en las interfaces profundas. El modo fundamental (modo 0) es el más lento y de mayor amplitud energética continua. Los sobretonos o modos superiores (modo 1, modo 2) aparecen a velocidades de fase más altas en frecuencias específicas de corte (*cut-off frequencies*).
2. **Modos de Fuga (*Leaky Modes*):** Ondas pseudoguiadas que no cumplen la condición de reflexión total y radian parcialmente energía refractada hacia el semiespacio profundo de alta velocidad. Aparecen en el espectro como ramas continuas pero difusas a velocidades superiores ($> 2.5\text{ km/s}$) a frecuencias intermedias.

---

### INTERPRETACIÓN DE LA FIGURA 2.2: ESPECTRO $f-v$ Y PERFIL 1D REGIONAL $V_s(z)$

![Figura 2.2: Espectro frecuencia-velocidad f-v y modelo 1D en profundidad](file:///home/alexander/Fisica/resultados/real/2_2_fv_and_1d_profile.png)

- **Panel a (Espectro F-V):**
  - Muestra la energía normalizada con la velocidad de fase en el eje Y ($0.5$ a $4.0\text{ km/s}$) y la frecuencia en el eje X ($0$ a $100\text{ Hz}$).
  - La trayectoria de máxima energía traza con precisión la curva de dispersión fundamental de Rayleigh, marcada con la anotación **"Normal modes"** cerca de $1.2 - 1.5\text{ km/s}$ a frecuencias de $20 - 40\text{ Hz}$.
  - Se visualizan también aportes energéticos en la parte superior correspondientes a **"Leaky modes"** alcanzando $\sim 2.7\text{ km/s}$.
- **Panel b (Modelo 1D regional de $V_s$ hasta 1000 m):**
  - La función `extract_dispersion_and_1d_profile` extrae los picos de máxima amplitud espectral en cada frecuencia, aplica un filtro de mediana y suavizado gaussiano, calcula la profundidad asintótica $z \approx \lambda/2 = \frac{v_R}{2f}$, ajusta la velocidad de corte mediante el factor elástico $V_s \approx v_R / 0.92$, e impone monotonicidad física (la velocidad aumenta o se mantiene constante con la profundidad debido al incremento de presión litoestática).
  - El perfil resultante (curva roja escalonada) muestra una capa sedimentaria superior que inicia a $\sim 500\text{ m/s}$ en superficie, aumentando gradualmente hasta alcanzar $\sim 1800 - 2000\text{ m/s}$ a $1000\text{ m}$ de profundidad.

---

### INTERPRETACIÓN DE LA FIGURA 2.2.c: ZOOM A 400 METROS Y ESTIMACIÓN DE $V_{s30}$

![Figura 2.2.c: Perfil 1D somero y Vs30](file:///home/alexander/Fisica/resultados/real/2_2_c_vs30_depth_zoom.png)

#### Significado Geotécnico de $V_{s30}$
El parámetro **$V_{s30}$** representa la velocidad media ponderada de las ondas de corte en los primeros 30 metros de profundidad. Es el estándar geotécnico y sismológico universal establecido por el **NEHRP** (*National Earthquake Hazards Reduction Program*, EE. UU.) y el **Eurocódigo 8** para clasificar el tipo de suelo y predecir los efectos de amplificación sísmica de sitio (los estratos de suelo blando atrapan y amplifican drásticamente las ondas sísmicas destruyendo edificios durante terremotos).

Matemáticamente, $V_{s30}$ se calcula mediante la **media armónica ponderada por tiempo de viaje**:
$$V_{s30} = \frac{30\text{ m}}{\sum_{i} \frac{\Delta z_i}{V_{s, i}}}$$
No es un promedio aritmético simple: una capa delgada muy blanda cerca de la superficie reduce fuertemente el valor de $V_{s30}$ porque el tiempo de tránsito a través de ella es grande.

#### Clasificación del Suelo según NEHRP:
| Clase de Sitio | $V_{s30}$ (m/s) | Descripción Geotécnica |
| :--- | :--- | :--- |
| **Clase A** | $> 1500$ | Roca dura (ígnea o metamórfica inalterada) |
| **Clase B** | $760 - 1500$ | Roca competente |
| **Clase C** | $360 - 760$ | Suelo muy denso y roca blanda |
| **Clase D** | $180 - 360$ | Suelo rígido (arcillas rígidas o arenas densas) |
| **Clase E** | $< 180$ | Suelo blando (arcillas blandas de alta compresibilidad) |

#### Resultado Obtenido en el Código:
En la Figura 2.2.c (`2_2_c_vs30_depth_zoom.png`), el software grafica el perfil somero de $0$ a $400\text{ m}$ destacando la línea horizontal azul a $z = 30\text{ m}$. El recuadro informativo arroja:
$$V_{s30} \approx 643.5\text{ m/s} \quad \implies \quad \textbf{Clase C (Suelo muy denso / roca blanda)}$$
Este valor refleja fielmente las características de depósitos aluviales compactados y formaciones sedimentarias consolidadas típicas de la región del cable DAS.

---

### 2.3 ESPECTROS FRECUENCIA-LENTITUD DE FASE ($f-s$)

#### Física de la Lentitud de Fase (*Phase Slowness*)
La **lentitud de fase** ($s$) es el inverso multiplicativo directo de la velocidad de fase:
$$s = \frac{1}{v} = \frac{k}{f}$$
Se mide en unidades de tiempo por unidad de distancia ($\text{s/m}$ o comúnmente $\text{s/km}$).
- Mientras que la velocidad $v \to \infty$ cuando la longitud de onda tiende a infinito (produciendo asíntotas difíciles de resolver numéricamente), la lentitud $s \to 0$, comportándose de forma estrictamente acotada y lineal.
- En problemas de cálculo de tiempos de viaje e inversión tomográfica de caminos de rayo, el tiempo de tránsito acumulado es una integral lineal de la lentitud: $T = \int s(\mathbf{x}) dl$.

---

### INTERPRETACIÓN DE LA FIGURA 2.3: ESPECTRO $f-s$

![Figura 2.3: Espectro frecuencia-lentitud de fase f-s](file:///home/alexander/Fisica/resultados/real/2_3_fs_spectrum.png)

La Figura 2.3 (`2_3_fs_spectrum.png`) representa la energía espectral proyectada sobre el plano de lentitud:
- **Eje X:** Frecuencia $f$ ($0$ a $100\text{ Hz}$).
- **Eje Y:** Lentitud de fase $s$ ($0.25$ a $2.0\text{ s/km}$).
- **Líneas de Referencia:**
  - $s = 0.33\text{ s/km} \implies v \approx 3.0\text{ km/s}$ ($3000\text{ m/s}$, velocidad de la roca profunda).
  - $s = 0.50\text{ s/km} \implies v = 2.0\text{ km/s}$ ($2000\text{ m/s}$).
  - $s = 1.00\text{ s/km} \implies v = 1.0\text{ km/s}$ ($1000\text{ m/s}$, velocidad de los sedimentos someros).
- La cresta de energía converge hacia $s \approx 0.35 - 0.45\text{ s/km}$ en las bajas frecuencias ($< 10\text{ Hz}$) y se desplaza gradualmente hacia $s \approx 0.85 - 1.0\text{ s/km}$ en las altas frecuencias ($> 40\text{ Hz}$), confirmando la dispersión física observada en los diagramas $f-k$ y $f-v$.

---

## 3. CORRELACIÓN CRUZADA Y FUNCIONES DE GREEN EMPÍRICAS

### 3.1 TEORÍA DE LA INTERFEROMETRÍA SÍSMICA DE RUIDO AMBIENTAL

#### Teorema de Reconstrucción de la Función de Green
En física de medios elásticos, la **Función de Green** $G(\mathbf{x}_B, \mathbf{x}_A, t)$ describe la respuesta del medio en el punto $\mathbf{x}_B$ y tiempo $t$ provocada por una fuente impulsiva unitaria (función delta de Dirac espacial y temporal) disparada en el punto $\mathbf{x}_A$ en el instante $t = 0$.

El principio fundamental de la interferometría sísmica pasiva (Lobkis & Weaver, 2001; Wapenaar, 2004; Shapiro & Campillo, 2004) establece que si un campo de ruido elástico es estacionario, difuso y generado por fuentes naturales aleatorias distribuidas homogéneamente en el espacio, la **correlación cruzada temporal promediada** de las señales registradas en dos sensores $A$ y $B$ converge a la función de Green entre ambos puntos.

Matemáticamente, para la correlación cruzada de tasa de deformación DAS:
$$C_{AB}(\tau) = \langle u_A(t) \star u_B(t) \rangle = \int_{-\infty}^{\infty} u_A(t) \cdot u_B(t + \tau) \, dt$$
De acuerdo con el teorema de correlación (ecuación 6 de Li et al., 2020):
$$\frac{d C_{AB}(t)}{dt} \approx -G(B, A, t) + G(A, B, -t)$$
donde:
- $\frac{d C_{AB}(t)}{dt}$ es la **primera derivada temporal** de la correlación cruzada.
- $G(B, A, t)$ para $t > 0$ es la **rama causal**, representando una onda física real propagándose desde la estación $A$ (actuando como *fuente virtual*) hacia la estación $B$.
- $G(A, B, -t)$ para $t < 0$ es la **rama acausal** (o anticausal), representando la onda virtual propagándose en sentido contrario desde $B$ hacia $A$.

> **¿Por qué es indispensable la Derivada Temporal $\frac{d C_{AB}}{dt}$?**
> La correlación cruzada pura de señales de campo difuso 2D introduce un desfasaje intrínseco de $\pi/2$ radianes ($90^\circ$) y una atenuación espectral proporcional a $1/\omega$. Al calcular la derivada temporal en el dominio de Fourier ($\mathcal{F}\{\frac{dC}{dt}\} = i\omega C(\omega)$), la multiplicación por $i\omega$ cancela exactamente el retardo de $\pi/2$ y compensa el factor $1/\omega$, restaurando la fase e impulso de onda equivalentes a una detonación sísmica impulsiva real.

#### Implementación Espectral por Teorema de Convolución
Calcular correlaciones en bucles de tiempo requeriría un costo computacional prohibido de $\mathcal{O}(N^2)$. El código (`tsw_rtm/interferometry.py`) computa la correlación cruzada en el dominio de Fourier en $\mathcal{O}(N \log N)$ mediante densidad espectral cruzada:
$$S_{AB}(f) = X_A^*(f) \cdot X_B(f)$$
$$C_{AB}(\tau) = \mathcal{F}^{-1}\left\{ X_A^*(f) \cdot X_B(f) \right\}$$
donde $X_A^*(f)$ denota el complejo conjugado del espectro de Fourier de la traza de la fuente virtual, y $X_B(f)$ es el espectro del receptor.

#### Apilado Coherente (*Stacking*)
Al procesar un único archivo de un minuto, el ruido no es isotrópico y el fondo presenta oscilaciones aleatorias. Al apilar (sumar y promediar) decenas de archivos a lo largo del tiempo:
$$\overline{C_{AB}}(\tau) = \frac{1}{N_{archivos}} \sum_{k=1}^{N_{archivos}} C_{AB}^{(k)}(\tau)$$
El ruido incoherente se cancela por **interferencia destructiva**, mientras que los pulsos deterministas correspondientes a las ondas de Rayleigh que viajan entre ambos canales se refuerzan por **interferencia constructiva**. La relación señal/ruido (SNR) se incrementa de acuerdo con la ley estadística de grandes números como:
$$\text{SNR} \propto \sqrt{N_{archivos}}$$

---

### INTERPRETACIÓN DE LA FIGURA 3.1: CORRELACIÓN CRUZADA, FUNCIÓN DE GREEN Y FORMA DE ONDA

![Figura 3.1: Correlación cruzada, Función de Green y forma de onda](file:///home/alexander/Fisica/resultados/real/5_ccf_green_src0.png)

La Figura 3.1 (`5_ccf_green_src0.png`) ilustra la reconstrucción interferométrica para la fuente virtual principal:
- **Panel Izquierdo (Stacked Cross-Correlation $C(t)$):**
  - Muestra la matriz de correlación acumulada en función del offset de distancia ($0$ a $\sim 4.5\text{ km}$) y el retardo temporal $\tau$ en el rango simétrico de $-3.5\text{ s}$ a $+3.5\text{ s}$.
- **Panel Central (Empirical Green's Function $dC/dt$):**
  - Muestra el resultado tras aplicar la derivada temporal. Se aprecian frentes de onda lineares bien definidos.
  - **Asimetría Causal / Acausal:** Se observa que la energía en la rama causal ($t > 0$, abajo del eje $0.0$) es más vigorosa y continua que en la rama acausal ($t < 0$, arriba). Esta asimetría física demuestra que en la región de estudio existe una fuente energética dominante de ruido (por ejemplo, oleaje oceánico en el Pacífico o tránsito vehicular principal en una dirección específica de la carretera) que viaja preponderantemente desde los canales menores hacia los mayores.
- **Panel Derecho (Waveform at offset):**
  - Muestra el sismograma unidimensional empírico recuperado a una distancia intermedia representativa. Las áreas rellenas en rojo indican la fase compresiva positiva y en azul la fase dilatacional negativa, revelando un paquete de ondas de pulso coherente idéntico al registro de un sismómetro real frente a un disparo controlado.

---

### 3.2 DISPARO VIRTUAL COMÚN APILADO (*VIRTUAL COMMON SHOT GATHER*)

#### Selección de la Rama Causal y Filtrado Direccional $F-K$
Para alimentar el simulador de migración sísmica hacia adelante, se aísla la rama causal ($t \ge 0$), lo que crea un sismograma sintético equivalente a haber detonado un vibrador o explosivo justo en la coordenada del cable del canal fuente en $t = 0$.

Para eliminar cualquier remanente de ondas que viajen en sentido retrógrado o ruidos de velocidad aparente incompatible, el gather virtual se somete a un **filtro bidimensional en abanico $F-K$** (`preprocessing.fk_filter`):
- Se construye una máscara en el plano $(f, k)$ que deja pasar únicamente la energía cuyas velocidades aparentes cumplan:
  $$800\text{ m/s} \le \frac{f}{|k|} \le 3500\text{ m/s}$$
- Las componentes con $v_{app} > 3500\text{ m/s}$ (ondas de cuerpo y ruido acoplado de modo común) y $v_{app} < 800\text{ m/s}$ (ondas acústicas directas en el aire a $340\text{ m/s}$) son silenciadas.

---

### INTERPRETACIÓN DE LA FIGURA 3.2: DISPARO VIRTUAL COMÚN APILADO

![Figura 3.2: Registro de disparo virtual apilado](file:///home/alexander/Fisica/resultados/real/2_virtual_shot_gather.png)

La Figura 3.2 (`2_virtual_shot_gather.png`) presenta el registro final del disparo virtual:
- **Eje X:** Offset o distancia relativa desde la fuente virtual ($0.0$ a $\sim 4.5\text{ km}$).
- **Eje Y:** Tiempo de viaje de propagación ($0.0$ a $3.5\text{ s}$).
- Se distingue un tren de ondas de Rayleigh dispersivo con una pendiente promedio correspondiente a $\sim 1500 - 2000\text{ m/s}$. El sismograma exhibe una notable relación señal/ruido, con fases continuas libres de ruido incoherente, listo para ser inyectado como condición de frontera en la migración RTM.

---

## 4. MIGRACIÓN DE ONDAS SUPERFICIALES EN REVERSA EN EL TIEMPO (TSW-RTM)

### 4.1 EL PARADIGMA DE MIGRACIÓN: POR QUÉ FALLA LA RTM CONVENCIONAL Y CÓMO FUNCIONA TSW-RTM

#### El Principio de Imagen de Claerbout (RTM Convencional de Reflexión)
La migración estándar en reversa en el tiempo (*Reverse Time Migration*, Claerbout, 1971) es el algoritmo más avanzado de la industria petrolera para iluminar trampas estructurales complejas (domos de sal, fallas de sobreescurrimiento). Su hipótesis se fundamenta en la **reflexión de ondas de cuerpo**:
1. Una fuente detona en superficie y emite un frente de onda que baja al subsuelo ($S(\mathbf{x}, t)$, campo incidente propagado hacia adelante en el tiempo).
2. La onda choca con un reflector litológico que presenta un contraste de impedancia acústica ($Z_2 \neq Z_1$) y produce un eco reflejado hacia arriba.
3. El eco es grabado por geófonos en superficie. En el software, este registro es invertido en el tiempo e inyectado como condición de frontera para retropropagarlo hacia abajo ($R(\mathbf{x}, t)$, campo receptor propagado en reversa temporal).
4. El plano del reflector se ilumina aplicando la **condición de imagen por correlación cruzada de tiempo cero**:
   $$I(\mathbf{x}) = \int_0^T S(\mathbf{x}, t) \cdot R(\mathbf{x}, t) \, dt$$
   En cualquier punto donde haya un reflector, la onda que baja y la onda que sube coinciden exactamente en tiempo y espacio en el instante del rebote, produciendo un producto multiplicativo positivo gigante ($S \cdot R > 0$).

#### El Desafío de las Ondas Transmitidas a Través de Fallas Tectónicas
En una falla tectónica profunda como la Falla de San Andrés o la Falla de Tanlu, dos bloques litológicos muy diferentes entran en contacto lateral abrupto:
- **Bloque A:** Sedimentos cenozoicos blandos ($v_s \approx 1100 - 1500\text{ m/s}$).
- **Bloque B:** Roca metamórfica o ígnea del basamento cristalino ($v_g \approx 2400 - 3000\text{ m/s}$).

Cuando usamos ruido ambiental pasivo y un cable en superficie:
1. Las ondas que registramos con mayor energía son **ondas superficiales transmitidas** que se propagan horizontalmente a lo largo de la superficie libre, cruzando a través del plano de la falla desde los sedimentos hacia la roca.
2. Si intentáramos simular la migración usando un modelo de velocidad que ya tuviera dibujada la discontinuidad de la falla, la simulación numérica crearía **reflexiones artificiales numéricas** espurias en la interfaz que colapsarían y falsearían la imagen.
3. Peor aún: **no conocemos de antemano la posición ni la inclinación exacta de la falla** (¡eso es precisamente lo que queremos descubrir!).

---

### 4.2 LA TÉCNICA DE INUNDACIÓN DE VELOCIDAD DUAL (*VELOCITY FLOOD TECHNIQUE*)

Para superar esta paradoja, Li et al. (2020) desarrollaron la técnica de **inundación lateral de velocidad dual**: se sustituye deliberadamente el modelo geológico verdadero desconocido por **dos modelos auxiliares continuos y sin fallas**:

```
[ Modelo Geológico Real ] 
    Sedimento (vs)       │       Roca Metamórfica (vg)
   (Velocidad baja)      │         (Velocidad alta)
                         ▼
                   PLANO DE FALLA

──────────────────────────────────────────────────────────────────

[ Modelo Inundado 1: Sedimento Flood (vs) ]  ──► Simulación Hacia Adelante (Forward)
    Sedimento (vs)       │       Sedimento Inundado (vs)
  (La roca de la derecha es reemplazada por sedimento continuo)

──────────────────────────────────────────────────────────────────

[ Modelo Inundado 2: Roca Flood (vg) ]       ──► Simulación en Reversa (Backward)
     Roca Inundada (vg)  │       Roca Metamórfica (vg)
  (El sedimento de la izquierda es reemplazado por roca continua)
```

#### Fundamento Matemático de la Coincidencia Espacio-Temporal
1. **Propagación hacia adelante en $v_s$:**
   Se dispara la fuente virtual $x_s$ situada sobre el sedimento y se simula la onda viajando hacia la derecha a través del modelo continuo $v_s$. La onda alcanza una posición arbitraria $x$ en el tiempo de viaje teórico:
   $$t_{fwd}(x) = \frac{x - x_s}{v_s}$$

2. **Propagación en reversa en $v_g$:**
   Se toma el sismograma de ondas de Rayleigh transmitidas registradas por los receptores $x_g$ situados sobre la roca, se invierte el tiempo y se retropropagan hacia la izquierda a través del modelo continuo $v_g$. El tiempo aparente en que esa onda retropropagada alcanza la posición $x$ es:
   $$t_{bwd}(x) = t_{total} - \frac{x_g - x}{v_g}$$

3. **En la Naturaleza Real:**
   La onda física real que viajó de $x_s$ a $x_g$ gastó un tiempo total que consistió en dos tramos: viajó por el sedimento a velocidad $v_s$ hasta encontrar la falla en $x_{falla}$, y a partir de allí viajó por la roca a velocidad $v_g$ hasta el receptor $x_g$:
   $$t_{total} = \frac{x_{falla} - x_s}{v_s} + \frac{x_g - x_{falla}}{v_g}$$

4. **El Milagro del Colapso en el Plano de la Falla:**
   Si igualamos los dos tiempos simulados en el software ($t_{fwd}(x) = t_{bwd}(x)$):
   $$\frac{x - x_s}{v_s} = \left( \frac{x_{falla} - x_s}{v_s} + \frac{x_g - x_{falla}}{v_g} \right) - \frac{x_g - x}{v_g}$$
   Reagrupando términos:
   $$\frac{x - x_{falla}}{v_s} = \frac{x_{falla} - x}{v_g} \iff (x - x_{falla}) \left( \frac{1}{v_s} + \frac{1}{v_g} \right) = 0$$
   Dado que las velocidades son números positivos finitos ($\frac{1}{v_s} + \frac{1}{v_g} > 0$), la única solución matemática en todo el espacio para la cual los dos campos de onda coinciden simultáneamente en tiempo y espacio es:
   $$x = x_{falla}$$

> **Conclusión Teórica Fundamental:**
> En cualquier punto a la izquierda o derecha de la falla, las velocidades simuladas son incorrectas y los frentes de onda se desfasan progresivamente (interferencia destructiva). **Única y exclusivamente sobre el plano tridimensional de la discontinuidad tectónica**, los dos frentes de onda se tocan en tiempo cero, iluminando la falla con nitidez fotográfica.

---

### TABLA DE PARÁMETROS GEOMÉTRICOS, ELÁSTICOS Y NUMÉRICOS UTILIZADOS

Los siguientes parámetros (definidos en `config.yaml` y compilados en el reporte Word) gobiernan cuantitativamente la construcción del modelo y la propagación de ondas:

| Bloque Temático | Parámetro Geofísico / Numérico | Valor Asignado | Justificación Física |
| :--- | :--- | :--- | :--- |
| **Modelo de Velocidad** | Velocidad de sedimento en superficie ($v_{s0}$) | **$1100.0\text{ m/s}$** | Velocidad típica de ondas $S$ en sedimentos aluviales someros. |
| **Modelo de Velocidad** | Gradiente vertical de sedimento ($k_s$) | **$0.4\text{ m/s por metro}$** | Aumento lineal de velocidad por compactación y confinamiento. |
| **Modelo de Velocidad** | Velocidad de roca en superficie ($v_{g0}$) | **$2400.0\text{ m/s}$** | Basamento rocoso metamórfico competente. |
| **Modelo de Velocidad** | Gradiente vertical de roca ($k_g$) | **$0.2\text{ m/s por metro}$** | Gradiente menor en roca densa consolidada. |
| **Modelo de Velocidad** | Posición de la falla supuesta ($x_{falla}$) | **$x = 1.54\text{ km}$** | Centro geométrico de la ventana de canales analizada. |
| **Modelo de Velocidad** | Configuración lateral | **Sedimento a la izquierda** | Falla vertical dividiendo el dominio en dos bloques. |
| **Modelo de Velocidad** | Tamaño de la malla numérica ($N_x \times N_z$) | **$300 \times 120\text{ celdas}$** | Discretización 2D del semiespacio geológico. |
| **Modelo de Velocidad** | Espaciamiento horizontal ($\Delta x_{sim}$) | **$10.27\text{ m}$** | Resolución espacial equivalente a 2 canales DAS físicos. |
| **Modelo de Velocidad** | Espaciamiento vertical ($\Delta z_{sim}$) | **$10.0\text{ m}$** | Resolución en profundidad. |
| **Modelo de Velocidad** | Profundidad total del modelo | **$1.2\text{ km}$ ($1200\text{ m}$)** | Rango de sensibilidad de penetración de las ondas de Rayleigh. |
| **Modelo de Velocidad** | Margen de inundación lateral | **$8\text{ celdas}$** | Distancia de seguridad para evitar tocar la falla verdadera. |
| **Simulación RTM** | Paso de tiempo temporal ($\Delta t$) | **$0.0015\text{ s}$ ($1.5\text{ ms}$)** | Satisface la condición de estabilidad numérica de Courant (CFL). |
| **Simulación RTM** | Frecuencia central de la fuente (Ricker) | **$2.0\text{ Hz}$** | Sintonizada con la frecuencia dominante de los datos DAS. |
| **Simulación RTM** | Duración simulada ($T$) | **$3.5\text{ s}$** | Ventana temporal suficiente para cruzar todo el cable. |
| **Simulación RTM** | Espesor de borde absorbente (Esponja) | **$20\text{ celdas}$** | Amortiguamiento Cerjan para extinguir reflexiones artificiales. |
| **Simulación RTM** | Intervalo de instantáneas (*snapshots*) | **Cada 2 pasos ($3\text{ ms}$)** | Balance entre precisión de correlación y consumo de RAM. |
| **Simulación RTM** | Número de fuentes virtuales ($N_s$) | **4 fuentes** | Canales 1220, 1263, 1306, 1350 del cable DAS. |
| **Simulación RTM** | Receptores sísmicos ($N_r$) | **95 receptores** | Uno cada 4 canales ($41.08\text{ m}$) en el bloque derecho. |
| **Simulación RTM** | Filtro $F-K$ direccional aplicado | **$800\text{ m/s}$ a $3500\text{ m/s}$** | Aislamiento estricto de las ondas superficiales transmitidas. |

---

### INTERPRETACIÓN DE LA FIGURA 4.1: MODELOS DE INUNDACIÓN DE VELOCIDAD DUAL

![Figura 4.1: Modelos de velocidad de inundación dual](file:///home/alexander/Fisica/resultados/real/3_das_flood_models.png)

La Figura 4.1 (`3_das_flood_models.png`) muestra la preparación de los modelos de propagación según Li et al. (2020):
- **Panel a (True / Reference Velocity Model):**
  - Muestra el modelo geológico de referencia con dos bloques litológicos contrastantes: sedimentos lentos a la izquierda ($1100 - 1580\text{ m/s}$, tonos fríos) y roca metamórfica rápida a la derecha ($2400 - 2640\text{ m/s}$, tonos cálidos/rojos), separados por la falla en $x = 1.54\text{ km}$.
- **Panel b (Sediment Flood Model $v_s$):**
  - Las velocidades de sedimento de la izquierda se extienden horizontalmente hacia la derecha a lo largo de todo el modelo, borrando la roca. Este modelo se utiliza exclusivamente para propagar el campo de la fuente hacia adelante.
- **Panel c (Metamorphic Rock Flood Model $v_g$):**
  - Las velocidades de roca de la derecha se extienden horizontalmente hacia la izquierda a lo largo de todo el modelo, borrando los sedimentos. Este modelo se utiliza exclusivamente para retropropagar las ondas registradas por los receptores en reversa temporal.

---

### 4.3 MOTOR DE PROPAGACIÓN FÍSICA FDTD ACELERADO EN C CON OPENMP

Para ejecutar las simulaciones numéricas bidimensionales con el máximo rendimiento, el software no utiliza bucles lentos de Python, sino una biblioteca dinámica compilada en C (`tsw_rtm/c_engine/wave2d.c`) vectorizada con directivas multihilo de OpenMP.

#### Ecuación de Onda Acústica 2D y Discretización en Diferencias Finitas
El campo de presión escalar $p(x, z, t)$ satisface la ecuación diferencial en derivadas parciales hiperbólica:
$$\frac{\partial^2 p}{\partial t^2} = v(x, z)^2 \left( \frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial z^2} \right)$$

En el motor C (`acoustic_2d_step`), las derivadas espaciales se discretizan sobre una malla regular con un **operador centrado de 4º orden en el espacio**:
$$\frac{\partial^2 p}{\partial x^2} \approx \frac{c_2 (p_{i+2} + p_{i-2}) + c_1 (p_{i+1} + p_{i-1}) + c_0 p_i}{\Delta x^2}$$
donde los coeficientes óptimos de Taylor son:
$$c_0 = -2.5, \quad c_1 = \frac{4}{3} \approx 1.3333, \quad c_2 = -\frac{1}{12} \approx -0.08333$$

La integración temporal se realiza mediante el algoritmo explícito de segundo orden tipo salto de rana (*Leapfrog*):
$$p^{n+1}_{i, j} = 2 p^n_{i, j} - p^{n-1}_{i, j} + v_{i, j}^2 \Delta t^2 \left( \left.\frac{\partial^2 p}{\partial x^2}\right|_{i, j}^n + \left.\frac{\partial^2 p}{\partial z^2}\right|_{i, j}^n \right)$$

#### Condición de Estabilidad de Courant-Friedrichs-Lewy (CFL)
Para evitar que la solución numérica diverja exponencialmente hacia el infinito o genere valores `NaN`, el paso temporal $\Delta t$ debe satisfacer el criterio estricto de CFL:
$$\text{CFL} = v_{max} \cdot \Delta t \cdot \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta z^2}} \le \gamma_{max} \approx 0.60$$
Para los parámetros del proyecto:
$$v_{max} = 2640\text{ m/s}, \quad \Delta x = 10.27\text{ m}, \quad \Delta z = 10.0\text{ m}, \quad \Delta t = 0.0015\text{ s}$$
$$\text{CFL} = 2640 \cdot 0.0015 \cdot \sqrt{\frac{1}{10.27^2} + \frac{1}{10.0^2}} = 3.96 \cdot 0.1397 \approx 0.553 \le 0.60$$
El sistema opera en una zona de estabilidad numérica absoluta demostrada.

#### Condición de Superficie Libre en la Cota Superior ($z = 0$)
Las ondas de Rayleigh son guiadas por la frontera superior de tracción nula con la atmósfera. En el modelo acústico, la presión acústica se anula en la superficie libre ($p(x, z=0) = 0$). Para preservar el orden de aproximación espacial, se aplica una reflexión anti-simétrica en la celda ficticia superior:
$$p_{x, 0}^{n+1} = 0.0, \quad p_{x, 1}^{n+1} = -p_{x, 2}^{n+1}$$
Esto obliga a la onda a rebotar hacia abajo con inversión de polaridad de fase idéntica a la superficie topográfica terrestre.

#### Capa Esponjosa Absorbente de Cerjan en los Bordes Laterales e Inferior
Una simulación por computadora debe resolverse dentro de un dominio numérico truncado. Si una onda choca con el borde de la matriz, rebotará hacia el interior generando reflexiones parásitas que destruirían la imagen. Para absorber completamente los frentes salientes, la función `build_cerjan_sponge` define una capa esponjosa de 20 celdas donde los valores del campo se multiplican en cada paso temporal por un factor de amortiguamiento gaussiano:
$$S(d) = \exp(-\alpha \cdot d^2)$$
donde $d$ es la distancia hacia el interior desde el inicio de la capa absorbente y $\alpha = 0.0015$.

---

### 4.4 POSTPROCESAMIENTO DE LA IMAGEN MIGRADA

#### 1. Normalización por Iluminación de la Fuente (*Illumination Compensation*)
La energía de una onda sísmica se atenúa drásticamente con la distancia y la profundidad debido a la dispersión geométrica esférica ($1/r$). Cerca de la superficie donde están las fuentes y los receptores, el campo de onda $S(x, z, t)$ es órdenes de magnitud más potente que a $1\text{ km}$ de profundidad. Si no se compensa este efecto, la imagen RTM resultante mostraría una mancha hiperbrillante en los primeros 50 metros y la falla profunda quedaría totalmente invisible.

El motor calcula el mapa de iluminación acumulada de energía de la fuente:
$$E_{src}(x, z) = \sum_{shots} \int_0^T S(x, z, t)^2 \, dt$$
Y normaliza la imagen migrada dividiendo por la raíz cuadrada de la energía:
$$I_{norm}(x, z) = \frac{I_{raw}(x, z)}{\sqrt{E_{src}(x, z)} + \epsilon}$$
donde $\epsilon$ es un término de regularización para evitar divisiones entre cero en áreas no iluminadas.

#### 2. Filtro Laplaciano Espacial 2D (`apply_laplacian_filter`)
La correlación cruzada de tiempo cero entre frentes de onda incidentes y retropropagados produce manchas de baja frecuencia (*low-frequency acquisition artifacts*) causadas por el solapamiento angular amplio de las ondas viajeras (la longitud finita del pulso de la ondícula Ricker).

Para eliminar las componentes de muy baja frecuencia y colapsar la energía sobre el plano de discontinuidad, se aplica el operador diferencial Laplaciano espacial bidimensional:
$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial z^2}$$
Discretizado mediante una máscara de convolución 2D de $3 \times 3$ elementos:
$$\nabla^2 I_{i, j} \approx \frac{I_{i+1, j} - 2 I_{i, j} + I_{i-1, j}}{\Delta x^2} + \frac{I_{i, j+1} - 2 I_{i, j} + I_{i, j-1}}{\Delta z^2}$$
**Acción Matemática:** El Laplaciano actúa como un filtro pasaltas espacial cuadrático ($|\mathbf{k}|^2 = k_x^2 + k_z^2$). Suprime completamente las manchas de fondo casi constantes ($k \to 0$) y resalta los gradientes espaciales agudos, afinando la imagen de la falla hasta convertirla en una línea delgada de alta resolución.

---

### INTERPRETACIÓN DE LA FIGURA 4.2: IMAGEN MIGRADA FINAL DE LA FALLA TECTÓNICA

![Figura 4.2: Imagen migrada TSW-RTM final de la falla tectónica](file:///home/alexander/Fisica/resultados/real/4_das_migrated_fault_image.png)

La Figura 4.2 (`4_das_migrated_fault_image.png`) es la culminación de todo el procesamiento geofísico:
- **Fondo en Color (Viridis):** Representa el modelo de velocidad de fondo regional con sedimentos lentos a la izquierda y roca a la derecha.
- **Primer Plano en Escala de Grises:** Imagen de reflectividad sísmica migrada con TSW-RTM tras la aplicación del filtro Laplaciano 2D.
- **Puntos Rojos Superpuestos:** Representan la posición real supuesta de la falla tectónica vertical en $x = 1.54\text{ km}$.
- **Colapso Focal:** 
  - La máxima concentración de amplitudes positivas y negativas de la imagen colapsa de manera nítida e inequívoca sobre el plano vertical de la discontinuidad litológica.
  - La energía se mantiene bien enfocada desde la superficie hasta aproximadamente $800 - 1000\text{ m}$ de profundidad, demostrando la capacidad de las ondas superficiales guiadas recuperadas de fibra óptica para cartografiar estructuras corticales someras sin recurrir a explosivos ni sismos activos.

---

## 5. VALIDACIÓN CON MODELOS SINTÉTICOS Y ANÁLISIS DE SENSIBILIDAD

Para complementar y consolidar la solidez del método, el sistema implementa en `run_synthetic_benchmarks.py` un banco de pruebas numéricas sintéticas bajo distintas condiciones estructurales extremas (basadas en los experimentos de Li et al., 2020), cuyos resultados se encuentran archivados en `resultados/sintetico/`:

### 1. Fallas con Distintos Ángulos de Buzamiento (*Dipping Faults*)
- **Falla Vertical ($90^\circ$):** Modelo base de referencia (`1_falla_vertical_migrated_image.png`).
- **Falla Inclinada a $65^\circ$:** Modela una falla normal o inversa de ángulo pronunciado (`2_falla_inclinada_65_migrated_image.png`).
- **Falla Inclinada a $35^\circ$:** Modela una falla de bajo ángulo (*thrust fault* o cabalgamiento profundo, `3_falla_inclinada_35_migrated_image.png`).
El algoritmo TSW-RTM reconstruye con idéntica precisión la inclinación espacial de la discontinuidad debido a que el método de diferencias finitas en dos sentidos no impone restricciones sobre el ángulo de propagación de los rayos sísmicos (a diferencia de las migraciones de Kirchhoff estándar que fallan con ángulos $> 60^\circ$).

### 2. Pruebas de Sensibilidad ante Errores de Velocidad de Migración ($\pm 10\%$)
En la práctica geofísica real, nunca conocemos la velocidad exacta del subsuelo con un 100% de certeza. Se evaluó la robustez de la imagen migrada introduciendo deliberadamente perturbaciones de velocidad en el modelo de inundación de roca:
- **Error del $+10\%$ en la velocidad de la roca:** (`4_vertical_roca_mas10_migrated_image.png`). La velocidad de retropropagación es un $10\%$ más rápida, lo que hace que los frentes de onda viajen más lejos en el mismo tiempo, desplazando levemente la imagen de la falla unos metros hacia la izquierda.
- **Error del $-10\%$ en la velocidad de la roca:** (`5_vertical_roca_menos10_migrated_image.png`). La velocidad es menor y la imagen se desplaza ligeramente hacia la derecha.
**Conclusión de Robustez:** A pesar de un error sistemático severo del $\pm 10\%$ en la velocidad (muy por encima del error típico de un modelo tomográfico 1D/2D bien calibrado), la discontinuidad de la falla sigue emergiendo como un rasgo estructural prominente y claramente interpretable, ratificando la estabilidad y viabilidad operativa del método TSW-RTM propuesto.

---

## 6. MAPEO COMPLETO ENTRE MÓDULOS DE CÓDIGO Y CONCEPTOS FÍSICOS

Para facilitar la navegación del código fuente, la siguiente tabla relaciona cada módulo del software con su fundamento físico directo:

| Archivo de Código | Función Principal | Concepto Físico / Algorítmico Implementado |
| :--- | :--- | :--- |
| `tsw_rtm/das_io.py` | `DASReader`, `read_window` | Lectura de archivos HDF5 de fibra óptica, extracción de tasa de deformación (*strain rate*) y metadatos espaciotemporales ($f_s, \Delta x$). |
| `tsw_rtm/preprocessing.py` | `decimate_time` | Decimación temporal con filtro pasabajas Butterworth anti-aliasing (muestreo de Nyquist). |
| `tsw_rtm/preprocessing.py` | `bandpass_filter` | Filtro pasabanda Butterworth cero-desplazamiento de fase (`sosfiltfilt`) para ondas de Rayleigh ($0.8 - 4.5\text{ Hz}$). |
| `tsw_rtm/preprocessing.py` | `temporal_normalization` | Normalización de 1-bit ($\text{sign}(x)$) para supresión de sismos locales y tráfico vehicular no estacionario. |
| `tsw_rtm/preprocessing.py` | `spectral_whitening` | Blanqueamiento espectral para aplanamiento de potencia y mitigación de ruidos tonales industriales. |
| `tsw_rtm/preprocessing.py` | `fk_filter` | Filtrado bidimensional direccional en el plano $F-K$ en abanico ($800 - 3500\text{ m/s}$). |
| `tsw_rtm/interferometry.py`| `compute_cross_correlation` | Correlación cruzada espectral de ruido ambiente mediante FFT ($\mathcal{O}(N \log N)$). |
| `tsw_rtm/interferometry.py`| `time_derivative_greens` | Reconstrucción de la Función de Green empírica mediante derivación temporal ($dC/dt$). |
| `tsw_rtm/dispersion.py` | `compute_kf_grid` | Extracción de cuadrícula de potencia positiva en $(f, k)$ para la cresta de Rayleigh. |
| `tsw_rtm/dispersion.py` | `compute_fv_spectrum` | Transformación de coordenadas al dominio frecuencia-velocidad de fase ($f-v$). |
| `tsw_rtm/dispersion.py` | `extract_dispersion_and_1d_profile` | Inversión 1D de velocidad de corte $V_s(z)$ a partir de $z \approx \lambda/2$ y factor elástico $0.92$. |
| `tsw_rtm/dispersion.py` | `compute_vs30` | Cálculo del promedio armónico de tiempo de viaje $V_{s30}$ y clasificación geotécnica NEHRP. |
| `tsw_rtm/dispersion.py` | `compute_fs_spectrum` | Transformación al dominio de frecuencia-lentitud de fase ($f-s$). |
| `tsw_rtm/velocity_flood.py` | `generate_velocity_flood_models` | Construcción de los modelos de inundación lateral continua de sedimento ($v_s$) y roca ($v_g$). |
| `tsw_rtm/wave_solver.py` | `WaveSolver2D` | Interfaz ctypes con la biblioteca compilada en C, validación CFL y condiciones de frontera. |
| `tsw_rtm/c_engine/wave2d.c`| `acoustic_2d_step` | Simulador FDTD en C con OpenMP, diferencias finitas de 4º orden en espacio y 2º orden en tiempo. |
| `tsw_rtm/c_engine/wave2d.c`| `build_cerjan_sponge` | Esponja absorbente gaussiana de Cerjan en los límites del dominio numérico. |
| `tsw_rtm/c_engine/wave2d.c`| `correlate_imaging_condition` | Condición de imagen RTM por correlación cruzada de tiempo cero acumulativa al vuelo. |
| `tsw_rtm/rtm_engine.py` | `TSWRTMEngine` | Orquestación multi-fuente, compensación de iluminación y filtrado Laplaciano 2D. |
| `tsw_rtm/visualizer.py` | Múltiples funciones de ploteo | Renderizado de figuras de alta resolución (200 DPI) y exportación de matrices completas a formato CSV. |
| `tsw_rtm/doc_builder.py` | `build_das_guide_docx` | Compilación automática del reporte técnico profesional en formato Word (`.docx`). |

---

## 7. SÍNTESIS FINAL Y CONCLUSIONES

El pipeline integrado en este proyecto representa un flujo de trabajo geofísico moderno y autosuficiente de **sismología pasiva de ultra-alta resolución**:
1. Extrae señales coherentes diminutas a partir del ruido caótico superficial registrado por infraestructura de fibra óptica preexistente (DAS).
2. Caracteriza las propiedades dispersivas del subsuelo determinando con rigor el perfil de rigidez elástica $V_s(z)$ y el parámetro geotécnico de sitio $V_{s30}$.
3. Supera las limitaciones de resolución de la tomografía tradicional mediante la **migración TSW-RTM con inundación dual de velocidad**, colapsando la energía de las ondas transmitidas exactamente sobre la discontinuidad de la falla tectónica.
4. Ejecuta todo el flujo numérico con alta eficiencia computacional mediante algoritmos paralelizados en C con OpenMP, produciendo resultados reproducibles, auditables y respaldados por datos numéricos exportados en CSV y documentos ejecutivos.
