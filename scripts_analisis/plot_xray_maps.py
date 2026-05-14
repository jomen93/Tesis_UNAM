#!/usr/bin/env python3
"""
plot_xray_maps.py — Genera imágenes 2D de emisión de rayos X
a partir de los binarios de raytracing de Walicxe3D.

Uso:
  python3 scripts_analisis/plot_xray_maps.py [opciones]

Opciones:
  --kappa DIR      Directorio con binarios XrayX_raytracing.*.bin (default: resultados/k_0)
  --out DIR        Directorio de salida (default: resultados/imagenes_xray/)
  --domain SIZE    Tamaño físico del dominio en AU (default: 80)
  --vmin VAL       Valor mínimo de escala (default: percentil 1 de todos los snaps)
  --vmax VAL       Valor máximo de escala (default: percentil 99.9)
  --cmap NAME      Colormap (default: inferno)
  --dpi N          DPI de salida (default: 150)
  --min-fill FRAC  Fracción mínima de píxeles no-cero para incluir un snap (default: 0.001)
"""

import argparse
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

AU = 1.496e13   # cm

# Parámetros orbitales de WR 140
PE_YR   = 7.93          # período orbital [yr]
DTOUT_YR = 0.119        # tiempo entre snapshots del run de raytracing [yr]
                        # inferido de las imágenes: snap 10 ↔ fase 0.150


def load_map(fpath, nx=256, ny=256):
    """Lee un mapa de raytracing (nx×ny float64, Fortran order)."""
    raw = np.frombuffer(open(fpath, 'rb').read(), dtype='float64')
    expected = nx * ny
    if len(raw) < expected:
        return None
    # Fortran column-major: x varía más rápido → data[ix, iz]
    return raw[:expected].reshape((nx, ny), order='F')


def is_valid(data, min_fill=0.001):
    """True si el mapa tiene suficientes píxeles con emisión real."""
    if data is None:
        return False
    n_pos = np.sum(data > 0)
    return n_pos >= min_fill * data.size


def orbital_phase(snap, dtout_yr=DTOUT_YR, pe_yr=PE_YR):
    """Fase orbital ∈ [0, 1) a partir del número de snapshot."""
    return (snap * dtout_yr / pe_yr) % 1.0


def make_image(data, snap, extent_au, vmin, vmax, cmap, outpath, dpi=150):
    """Genera y guarda una imagen PNG de un mapa de rayos X."""
    phase = orbital_phase(snap)

    # Reemplaza ceros/negativos por vmin para escala logarítmica
    img = np.where(data > 0, data, np.nan)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        img.T,                     # transponer: eje X horizontal, Z vertical
        origin='lower',
        extent=extent_au,          # [xmin, xmax, zmin, zmax] en AU
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        cmap=cmap,
        aspect='equal',
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'Emisión de Rayos X (erg s$^{-1}$ cm$^{-2}$)', fontsize=10)
    ax.set_xlabel('X (AU)', fontsize=11)
    ax.set_ylabel('Z (AU)', fontsize=11)
    ax.set_title(
        f'WR 140 — Rayos X  |  Snap {snap:04d}  |  Fase {phase:.3f}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    p = argparse.ArgumentParser()
    p.add_argument('--kappa',    default='resultados/k_0',         help='Directorio de raytracing')
    p.add_argument('--out',      default='resultados/imagenes_xray', help='Directorio de salida')
    p.add_argument('--domain',   type=float, default=80.0, help='Tamaño del dominio en AU')
    p.add_argument('--vmin',     type=float, default=None)
    p.add_argument('--vmax',     type=float, default=None)
    p.add_argument('--cmap',     default='inferno')
    p.add_argument('--dpi',      type=int,   default=150)
    p.add_argument('--min-fill', type=float, default=0.001,
                   help='Fracción mínima de píxeles >0 para incluir snap')
    args = p.parse_args()

    kappa_dir = args.kappa if os.path.isabs(args.kappa) else os.path.join(root, args.kappa)
    out_dir   = args.out   if os.path.isabs(args.out)   else os.path.join(root, args.out)

    # Buscar binarios
    files = sorted(glob.glob(os.path.join(kappa_dir, 'XrayX_raytracing.*.bin')))
    if not files:
        sys.exit(f'No se encontraron archivos en {kappa_dir}')

    # Determinar dimensiones del mapa desde el primer archivo no-vacío
    nx = ny = 256
    for f in files:
        raw = np.frombuffer(open(f, 'rb').read(), dtype='float64')
        n = int(len(raw) ** 0.5 + 0.5)
        if n * n == len(raw) and n > 0:
            nx = ny = n
            break

    extent_au = [0, args.domain, 0, args.domain]

    print(f'Directorio  : {kappa_dir}')
    print(f'Mapa        : {nx}×{ny} px  →  {args.domain:.0f} AU × {args.domain:.0f} AU')
    print(f'Archivos    : {len(files)} encontrados')

    # Cargar todos los mapas válidos
    valid = []
    for fpath in files:
        snap = int(os.path.basename(fpath).split('.')[-2])
        data = load_map(fpath, nx, ny)
        if is_valid(data, args.min_fill):
            valid.append((snap, data))

    if not valid:
        sys.exit('No hay snapshots válidos (todos ceros o casi vacíos).')

    print(f'Válidos     : {len(valid)} de {len(files)}  '
          f'(snaps: {valid[0][0]:04d} – {valid[-1][0]:04d})')

    # Escala global consistente entre todos los snapshots válidos
    if args.vmin is None or args.vmax is None:
        all_pos = np.concatenate([d[d > 0].ravel() for _, d in valid])
        global_vmin = args.vmin or float(np.percentile(all_pos, 1))
        global_vmax = args.vmax or float(np.percentile(all_pos, 99.9))
    else:
        global_vmin, global_vmax = args.vmin, args.vmax

    print(f'Escala      : vmin={global_vmin:.2e}, vmax={global_vmax:.2e} erg/s/cm²')

    # Generar imágenes
    os.makedirs(out_dir, exist_ok=True)
    for snap, data in valid:
        outpath = os.path.join(out_dir, f'xray_{snap:04d}.png')
        make_image(data, snap, extent_au, global_vmin, global_vmax,
                   args.cmap, outpath, args.dpi)
        phase = orbital_phase(snap)
        print(f'  ✓ snap {snap:04d}  fase={phase:.3f}  '
              f'max={data.max():.2e}  →  {os.path.basename(outpath)}')

    print(f'\nListo. {len(valid)} imágenes en: {out_dir}/')


if __name__ == '__main__':
    main()
