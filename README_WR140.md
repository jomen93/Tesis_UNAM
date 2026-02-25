# Simulación de WR 140: Dinámica de Vientos en Colisión

Este proyecto está configurado para simular la interacción hidrodinámica de los vientos estelares en el sistema binario masivo **WR 140** utilizando el código **Walicxe3D**.

## 1. El Sistema Físico: WR 140

WR 140 es el arquetipo de los "Colliding Wind Binaries" (CWB) de largo periodo. Es famoso por su alta excentricidad y su producción periódica de polvo.

### Parámetros Estelares Simulados
La simulación modela dos estrellas masivas con vientos supersónicos que chocan:

| Parámetro | Estrella 1 (Wolf-Rayet WC7) | Estrella 2 (O5.5fc) |
| :--- | :--- | :--- |
| **Masa ($M$)** | $20 M_{\odot}$ | $50 M_{\odot}$ |
| **Pérdida de Masa ($\dot{M}$)** | $4.3 \times 10^{-5} M_{\odot}/\text{yr}$ | $1.5 \times 10^{-6} M_{\odot}/\text{yr}$ |
| **Velocidad Viento ($v_{\infty}$)** | $2860$ km/s | $3000$ km/s |
| **Temperatura Viento** | $10^5$ K | $10^5$ K |

### Parámetros Orbitales
*   **Periodo ($P$):** 7.93 años (~2896 días).
*   **Excentricidad ($e$):** 0.881 (Órbita muy elíptica).
*   **Distancia en Periastron:** ~13 UA (El choque es muy fuerte e inestable).
*   **Distancia en Apoastron:** ~200 UA (El choque es adiabático y estable).

## 2. Resultados Esperados

Al ejecutar esta simulación, deberías observar los siguientes fenómenos físicos:

1.  **Región de Choque de Vientos (WCR):** Una estructura en forma de cono de choque que se curva debido al movimiento orbital (Efecto Coriolis), creando una espiral de Arquímedes.
2.  **Inestabilidades Hidrodinámicas:** Especialmente cerca del periastron, la zona de choque debería volverse turbulenta debido a inestabilidades de *Kelvin-Helmholtz* y *Thin-Shell* (NTSI) si el enfriamiento es eficiente.
3.  **Variabilidad en Rayos X:**
    *   La luminosidad de rayos X ($L_X \propto 1/D$) debería aumentar drásticamente en el periastron (cuando $D$ es mínima).
    *   Sin embargo, justo antes del periastron, deberías ver una caída súbita en los rayos X blandos. Esto es la **absorción** por el viento denso de la Wolf-Rayet pasando por delante de la región de choque.

## 3. Instrucciones de Ejecución

### Requisitos Previos (En tu computadora Ryzen/Linux)
Asegúrate de tener instalados los compiladores y librerías MPI:
```bash
sudo apt update
sudo apt install build-essential gfortran libopenmpi-dev openmpi-bin
```

### Paso 1: Configurar el Compilador
Edita el archivo `Makefile`. Busca las líneas de `COMPILER` y asegúrate de usar `mpif90` con flags de `gfortran`.

**Ejemplo de configuración en Makefile:**
```makefile
# ... (al inicio del archivo)
COMPILER = mpif90
# ...
USER_FLAGS = -O3 -fbacktrace -ffree-line-length-none
# ...
```
*(Nota: Comenta las líneas que hacen referencia a `ifort`)*

### Paso 2: Compilar
Limpia compilaciones anteriores y genera el nuevo ejecutable:
```bash
make clean
make
```
Esto generará un ejecutable llamado `raytracing_test` (o el nombre definido en `PROGRAM` en el Makefile).

### Paso 3: Ejecutar la Simulación
Para correr la simulación utilizando 16 núcleos (optimizado para Ryzen 7 9800):

```bash
mpirun -np 16 ./raytracing_test
```
*   **Salida:** Los datos se guardarán en la carpeta `data/` (asegúrate de que exista: `mkdir -p data`).
*   **Logs:** El progreso se mostrará en pantalla o en archivos `.log` en `data/`.

### Paso 4: Visualización y Análisis
Una vez terminada la simulación (o mientras corre), usa los scripts de Python para ver los resultados:

1.  **Mapas de Densidad/Temperatura:**
    Usa `plot_coldens.py` (necesitarás adaptarlo) o herramientas como **VisIt** o **ParaView** para abrir los archivos `.vtk` generados.

2.  **Curva de Luz de Rayos X:**
    Ejecuta el script de post-proceso:
    ```bash
    python3 xray_curve.py
    ```
    Esto leerá los archivos binarios de salida y generará la gráfica de luminosidad vs. fase orbital.

## Notas Importantes
*   **Memoria:** La simulación está configurada para usar ~24 GB de RAM. Cierra navegadores web y otras aplicaciones pesadas antes de correrla.
*   **Tiempo:** Simular 8 años de tiempo físico puede tardar varias horas o días de cómputo, dependiendo de la resolución (`maxlev`) elegida en `parameters.f90`.
