# CLAUDE.md — Tesis UNAM: Emisión de Rayos X en WR 140

## Qué es este proyecto

Tesis doctoral en astrofísica computacional (UNAM). El objetivo es simular la interacción hidrodinámica de los vientos estelares en el sistema binario masivo **WR 140** (arquetipo de *Colliding Wind Binary*) y calcular curvas de luz de rayos X para comparar con observaciones (JWST 2024-2025, Monnier et al. 2021).

**Código base**: [Walicxe3D](https://github.com/meithan/walicxe3d) — hidrodinámica 3D con AMR, escrito en Fortran, paralelizado con MPI. Autores originales: Juan C. Toledo y Alejandro Esquivel (ICN-UNAM).

## Compilar y correr

```bash
# Compilar simulación hidrodinámica principal
make clean && make

# Compilar postproceso de rayos X
make raytracing_c

# Correr simulación (8 procesos MPI ≈ 24 GB RAM)
mpirun -np 8 ./walicxe3d

# Verificar calidad de snapshot después de cada prueba
python3 verificar_snapshot_1.py
```

El ejecutable principal se llama `walicxe3d`. Los datos se guardan en `data/`.

## Archivos que modificamos (los demás son del código base, no tocar)

| Archivo | Qué controla |
|---|---|
| `source/parameters.f90` | Parámetros numéricos (solver, CFL, viscosidad, resolución) y tiempo de simulación |
| `source/user.f90` | Condiciones iniciales, parámetros de vientos, fase orbital inicial |
| `source/winds.f90` | Inyección de vientos — **Enfoque B implementado aquí** (zona de transición suave) |
| `source/orbits.f90` | Masas, período y excentricidad orbital |
| `source/raytracing.f90` | Postproceso de emisión de rayos X |

## Parámetros activos (estado actual)

**Física** (`source/user.f90` y `source/orbits.f90`):
- WR (WC7): Ṁ=1×10⁻⁵ M☉/yr, v∞=2860 km/s, radio_inyección=1.5 AU
- O5.5fc: Ṁ=1.5×10⁻⁶ M☉/yr, v∞=3000 km/s, radio_inyección=1.5 AU
- M1=10 M☉, M2=30 M☉, P=7.93 yr, e=0.90

**Numérico** (`source/parameters.f90`):
- Solver: HLLC, Limiter: Minmod, CFL: 0.1, visc_eta: 2e-2, cooling_limit: 0.05
- RAM_per_proc: 3000 MB, nProcs: 8, dowarm: .false.

## Estado del proyecto

**Simulación funcionando correctamente** (verificado 2026-05-12).

**Bug crítico resuelto**: Los scripts Python leían los binarios de Walicxe3D con `reshape(order='C')` (row-major) en lugar de `order='F'` (Fortran column-major). Esto intercambiaba ecuaciones con posiciones espaciales y generaba valores completamente no-físicos. La "inestabilidad" observada anteriormente era este bug de lectura.

**Valores físicos reales (snapshot 0001 de WR 140)**:
- ρ_max = 2.44×10⁻¹⁵ g/cm³ (zona de choque, físicamente razonable)
- v_max = 2998 km/s ≈ v∞ de la estrella O ✓
- T_max = 7.6×10⁷ K (gas chocado en colisión de vientos) ✓
- 100% de celdas válidas

**Regla de oro**: Siempre leer binarios Walicxe3D con `numpy.reshape(..., order='F')`.

**Test de viento esférico**: `tests/run_wind_test.sh` — pasa los 4 checks de validación física.

**Próximos pasos**:
1. Correr simulación completa de WR 140 y analizar evolución temporal
2. Calcular curvas de luz de rayos X con `make raytracing_c`
3. Comparar con observaciones (JWST 2024-2025)

## Estructura de directorios

```
source/                  # Fortran — solo modificar los 5 archivos listados arriba
data/                    # Salida de simulación (Blocks*.bin, State*.dat, Grid*.vtk)
tables/                  # Tablas de enfriamiento radiativo (no modificar)
scripts_analisis/        # Python: visualización, curvas de luz, lectura de binarios
scripts_automatizacion/  # Bash: compilación y monitoreo
documentacion_diagnosticos/  # Historial completo de diagnósticos y soluciones
logs_historicos/         # Logs de ejecuciones anteriores
k_0/ k_25/ k_250/ k_1000/ k_2500/  # Resultados de raytracing por kappa
```

## Scripts de análisis principales

```bash
python3 scripts_analisis/cortes_2d.py --snap N           # Cortes 2D (XY/XZ/YZ) de densidad, T, v
python3 scripts_analisis/cortes_2d.py --snap N --zoom 20 # Zoom en la región central ±20 AU
python3 scripts_analisis/xray_curve.py                   # Curva de luz de rayos X
python3 scripts_analisis/read_binary.py                  # Leer archivos binarios
./tests/run_wind_test.sh                                 # Validación: viento esférico simple
```

## Documentación relevante

- `README_WR140.md` — física del sistema y parámetros observados
- `documentacion_diagnosticos/DIAGNOSTICO_HIDRODINAMICO.md` — análisis del problema de inestabilidad
- `documentacion_diagnosticos/COMPARACION_PARAMETROS_WR140.md` — parámetros vs observaciones
- `AJUSTES_NUMERICOS_ESTABILIDAD.md` — justificación de cada ajuste numérico
- `PROGRESO_ENFOQUE_B.md` — estado actual y opciones disponibles

## Notas importantes

- La simulación usa ~24 GB de RAM. Cerrar navegadores antes de correr.
- `data/` contiene archivos binarios grandes — no commitear.
- El warm start (`State.*.dat`) está desactivado (`dowarm = .false.`) porque el último estado guardado estaba corrupto.
- El periastron de WR 140 (1.36 AU) con radios de inyección de 1.5 AU implica solapamiento — el Enfoque B en `winds.f90` está diseñado para manejarlo mediante mezcla gradual (Hermite cúbica).
