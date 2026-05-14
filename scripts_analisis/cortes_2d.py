#!/usr/bin/env python3
"""
cortes_2d.py — 2D cross-section cuts of Walicxe3D AMR binary output.

Reads binary snapshot files (float64, conserved variables) and reconstructs
the density, temperature, and velocity magnitude on three orthogonal slices
(XY at z=center, XZ at y=center, YZ at x=center).

Usage:
  python3 scripts_analisis/cortes_2d.py [options]

Options:
  --data DIR        Data directory  (default: data/)
  --snap N          Snapshot index  (default: last available)
  --nprocs N        MPI process count (default: auto-detected)
  --domain SIZE_AU  Domain size in AU (default: auto from data/)
  --test            Use test data (tests/single_wind/data/, 20 AU, 4 procs)
  --out DIR         Output directory for PNGs (default: resultados/imagenes_xray/)
  --show            Display plots interactively (default: save to file)
"""

import sys
import os
import math
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants (CGS)
# ──────────────────────────────────────────────────────────────────────────────
AU    = 1.496e13          # cm
AMU   = 1.66053906e-24   # g
KB    = 1.380649e-16     # erg/K
GAMMA = 5.0 / 3.0
MUI   = 0.61             # ionized mean mol. weight
MU0   = 1.3              # neutral mean mol. weight (code density scale)

# ──────────────────────────────────────────────────────────────────────────────
# Code unit scalings  (from parameter.dat / Walicxe3D conventions)
# ──────────────────────────────────────────────────────────────────────────────
D_SC  = MU0 * AMU        # g/cm³
V_SC  = 1.0e5            # cm/s
P_SC  = D_SC * V_SC**2   # dyne/cm²
L_SC  = AU               # cm

# ──────────────────────────────────────────────────────────────────────────────
# AMR block ID decoding  (mirrors admesh.f90: offsets / bcoords / getRefCorner)
# ──────────────────────────────────────────────────────────────────────────────

def _offsets(level, nrx, nry, nrz):
    """Blocks in all levels before `level` (mirrors Fortran offsets())."""
    result = 0
    for lv in range(2, level + 1):
        result += nrx * nry * nrz * 8 ** (lv - 2)
    return result


def decode_block(bID, maxlev, nrx, nry, nrz,
                 xphystot, yphystot, zphystot, ncells):
    """
    Return (xx, yy, zz, dx, dy, dz, level) for block bID.
    Coordinates in CGS; xx/yy/zz are the lower-corner of the block.
    """
    # Determine level
    off = 0
    blk_per_lev = nrx * nry * nrz
    for lv in range(1, maxlev + 1):
        n_at_lev = blk_per_lev * 8 ** (lv - 1)
        lo = off + 1
        hi = off + n_at_lev
        if lo <= bID <= hi:
            level = lv
            break
        off = hi
    else:
        raise ValueError(f"bID={bID} out of range for maxlev={maxlev}")

    off_lv = _offsets(level, nrx, nry, nrz)
    nx = nrx * 2 ** (level - 1)
    ny = nry * 2 ** (level - 1)
    nz = nrz * 2 ** (level - 1)
    localID = bID - off_lv

    x = localID % nx
    if x == 0:
        x = nx
    y = math.ceil(localID / nx) % ny
    if y == 0:
        y = ny
    z = math.ceil(localID / (nx * ny)) % nz
    if z == 0:
        z = nz

    dx = xphystot / (ncells * nrx * 2 ** (level - 1))
    dy = yphystot / (ncells * nry * 2 ** (level - 1))
    dz = zphystot / (ncells * nrz * 2 ** (level - 1))
    xx = (x - 1) * ncells * dx
    yy = (y - 1) * ncells * dy
    zz = (z - 1) * ncells * dz
    return xx, yy, zz, dx, dy, dz, level


# ──────────────────────────────────────────────────────────────────────────────
# Binary reader
# ──────────────────────────────────────────────────────────────────────────────

def read_snapshot(datadir, snap_idx, nprocs, neqtot=5, ncells=16):
    """
    Read all process files for snapshot `snap_idx`.
    Returns list of (bID, data[NEQTOT, NCELLS, NCELLS, NCELLS]).
    Data is in code units (float64).
    """
    blocks = []
    for proc in range(nprocs):
        fpath = os.path.join(datadir, f"Blocks{proc:03d}.{snap_idx:04d}.bin")
        if not os.path.exists(fpath):
            print(f"  WARNING: missing {os.path.basename(fpath)}")
            continue
        with open(fpath, "rb") as f:
            nblocks = np.frombuffer(f.read(4), dtype=np.int32)[0]
            for _ in range(nblocks):
                bID = np.frombuffer(f.read(4), dtype=np.int32)[0]
                n   = neqtot * ncells ** 3
                raw = np.frombuffer(f.read(n * 8), dtype=np.float64)
                if len(raw) != n:
                    break
                # Fortran writes column-major: eq varies fastest, then ix, iy, iz.
                # order='F' tells numpy to interpret the flat array as Fortran-ordered,
                # giving data[eq, ix, iy, iz] with correct correspondence.
                data = raw.reshape((neqtot, ncells, ncells, ncells), order='F')
                blocks.append((bID, data))
    return blocks


def find_last_snapshot(datadir, nprocs):
    pattern = os.path.join(datadir, "Blocks000.????.bin")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return int(os.path.basename(files[-1]).split(".")[1])


def find_all_snapshots(datadir):
    """Return sorted list of all available snapshot indices."""
    files = sorted(glob.glob(os.path.join(datadir, "Blocks000.????.bin")))
    return [int(os.path.basename(f).split(".")[1]) for f in files]


def autodetect_nprocs(datadir, snap_idx):
    count = 0
    for p in range(64):
        if os.path.exists(os.path.join(datadir, f"Blocks{p:03d}.{snap_idx:04d}.bin")):
            count += 1
        else:
            break
    return count if count > 0 else 8


# ──────────────────────────────────────────────────────────────────────────────
# Primitive variables (one cell)
# ──────────────────────────────────────────────────────────────────────────────

def to_prim_cgs(cell):
    """
    Convert conserved variables (code units) to CGS primitives.
    Returns (rho_cgs, vel_cgs_mag, temp_K) or None if unphysical.
    """
    rho_code = cell[0]
    if not (np.isfinite(rho_code) and rho_code > 0.0):
        return None
    vx = cell[1] / rho_code
    vy = cell[2] / rho_code
    vz = cell[3] / rho_code
    e_int = cell[4] - 0.5 * rho_code * (vx**2 + vy**2 + vz**2)
    pres = (GAMMA - 1.0) * e_int
    if not (np.isfinite(pres) and pres > 0.0):
        return None
    rho_cgs  = rho_code * D_SC
    vel_cgs  = np.sqrt(vx**2 + vy**2 + vz**2) * V_SC
    temp_K   = (pres * P_SC) / rho_cgs * (MUI * AMU / KB)
    return rho_cgs, vel_cgs, temp_K


# ──────────────────────────────────────────────────────────────────────────────
# 2D slice reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def build_slice(blocks, axis, slice_pos_cgs,
                xphystot, yphystot, zphystot,
                maxlev, nrx, nry, nrz, ncells,
                resolution=512):
    """
    Build a 2D slice perpendicular to `axis` (0=X, 1=Y, 2=Z) at `slice_pos_cgs`.

    Returns (rho_2d, vel_2d, temp_2d, coord1_edges, coord2_edges)
    where coord1/coord2 are the two axes spanning the slice plane (CGS).
    """
    phys = [xphystot, yphystot, zphystot]
    # Determine the two in-plane axes
    axes = [0, 1, 2]
    axes.remove(axis)
    ax1, ax2 = axes

    size1 = phys[ax1]
    size2 = phys[ax2]

    rho_grid  = np.full((resolution, resolution), np.nan)
    vel_grid  = np.full((resolution, resolution), np.nan)
    temp_grid = np.full((resolution, resolution), np.nan)
    lev_grid  = np.zeros((resolution, resolution), dtype=int)

    for bID, data in blocks:
        try:
            corners = decode_block(bID, maxlev, nrx, nry, nrz,
                                   xphystot, yphystot, zphystot, ncells)
        except ValueError:
            continue
        xx, yy, zz, dx, dy, dz = corners[:6]
        level = corners[6]

        block_lo = [xx, yy, zz]
        block_d  = [dx, dy, dz]
        block_hi = [xx + ncells * dx, yy + ncells * dy, zz + ncells * dz]

        # Check if slice plane intersects this block
        if not (block_lo[axis] <= slice_pos_cgs < block_hi[axis]):
            continue

        # Which cell index along the slice axis contains slice_pos?
        k_slice = int((slice_pos_cgs - block_lo[axis]) / block_d[axis])
        k_slice = min(k_slice, ncells - 1)

        # Indices in data: data[eq, k_z, k_y, k_x]  (C-order, z=axis 0 in array)
        # Walicxe3D layout: data[NEQTOT, NCELLS_z, NCELLS_y, NCELLS_x]
        # axis 0=X → loop over k_x = k_slice, varying k_y, k_z
        # axis 1=Y → loop over k_y = k_slice, varying k_x, k_z
        # axis 2=Z → loop over k_z = k_slice, varying k_x, k_y

        for i1 in range(ncells):
            for i2 in range(ncells):
                # data[eq, ix, iy, iz] after F-order reshape.
                # i1 indexes the first in-plane axis, i2 the second.
                if axis == 0:        # slice in X (ix=k_slice): ax1=Y, ax2=Z
                    cell = data[:, k_slice, i1, i2]  # data[eq, ix=k, iy=i1, iz=i2]
                    c1_lo = yy + i1 * dy;  c1_hi = c1_lo + dy
                    c2_lo = zz + i2 * dz;  c2_hi = c2_lo + dz
                elif axis == 1:      # slice in Y (iy=k_slice): ax1=X, ax2=Z
                    cell = data[:, i1, k_slice, i2]  # data[eq, ix=i1, iy=k, iz=i2]
                    c1_lo = xx + i1 * dx;  c1_hi = c1_lo + dx
                    c2_lo = zz + i2 * dz;  c2_hi = c2_lo + dz
                else:                # slice in Z (iz=k_slice): ax1=X, ax2=Y
                    cell = data[:, i1, i2, k_slice]  # data[eq, ix=i1, iy=i2, iz=k]
                    c1_lo = xx + i1 * dx;  c1_hi = c1_lo + dx
                    c2_lo = yy + i2 * dy;  c2_hi = c2_lo + dy

                result = to_prim_cgs(cell)
                if result is None:
                    continue

                # Fill the rectangle of pixels covered by this cell
                p1_lo = max(0, int(c1_lo / size1 * resolution))
                p1_hi = min(resolution - 1, int(c1_hi / size1 * resolution))
                p2_lo = max(0, int(c2_lo / size2 * resolution))
                p2_hi = min(resolution - 1, int(c2_hi / size2 * resolution))

                # Only overwrite pixels at coarser or equal level
                mask = lev_grid[p2_lo:p2_hi+1, p1_lo:p1_hi+1] <= level
                lev_sub = lev_grid[p2_lo:p2_hi+1, p1_lo:p1_hi+1]
                lev_sub[mask] = level
                lev_grid[p2_lo:p2_hi+1, p1_lo:p1_hi+1] = lev_sub

                rho_grid[p2_lo:p2_hi+1, p1_lo:p1_hi+1]  = np.where(mask, result[0], rho_grid[p2_lo:p2_hi+1, p1_lo:p1_hi+1])
                vel_grid[p2_lo:p2_hi+1, p1_lo:p1_hi+1]  = np.where(mask, result[1], vel_grid[p2_lo:p2_hi+1, p1_lo:p1_hi+1])
                temp_grid[p2_lo:p2_hi+1, p1_lo:p1_hi+1] = np.where(mask, result[2], temp_grid[p2_lo:p2_hi+1, p1_lo:p1_hi+1])

    c1_edges = np.linspace(0, size1 / AU, resolution + 1)
    c2_edges = np.linspace(0, size2 / AU, resolution + 1)
    return rho_grid, vel_grid, temp_grid, c1_edges, c2_edges


# ──────────────────────────────────────────────────────────────────────────────
# Global color scale
# ──────────────────────────────────────────────────────────────────────────────

def compute_global_scales(all_rho, all_vel, all_temp):
    """
    Compute consistent vmin/vmax for density, velocity, and temperature
    from lists of 2D grids (one per snapshot).  Uses percentiles so that
    a handful of extreme cells don't dominate the scale.
    """
    def pct(grids, lo, hi):
        vals = np.concatenate([g[np.isfinite(g) & (g > 0)].ravel() for g in grids])
        if len(vals) == 0:
            return (1e-30, 1.0)
        return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))

    rho_min,  rho_max  = pct(all_rho,  1,   99.9)
    temp_min, temp_max = pct(all_temp, 1,   99.9)
    _,        vel_max  = pct(all_vel,  0,   99.9)   # vel starts at 0

    return {
        'rho':  (rho_min,  rho_max),
        'temp': (temp_min, temp_max),
        'vel':  (0.0,      vel_max / 1e5),           # stored in km/s
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_slice(rho, vel, temp, c1_edges, c2_edges,
               plane_name, ax1_name, ax2_name,
               snap_idx, outdir, scales, show=False):
    """Generate and save a figure with three panels: density, temperature, velocity.

    `scales` must be a dict with keys 'rho', 'temp', 'vel', each a (vmin, vmax) tuple.
    Passing a fixed scales dict keeps the colorbar identical across all snapshots.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Snapshot {snap_idx:04d} — {plane_name} slice", fontsize=13)

    def panel(ax, data, title, cmap, unit_label, vmin, vmax, log=True):
        d = np.where(np.isfinite(data) & (data > 0), data, np.nan)
        norm = (mcolors.LogNorm(vmin=vmin, vmax=vmax) if log
                else mcolors.Normalize(vmin=vmin, vmax=vmax))
        im = ax.pcolormesh(c1_edges, c2_edges, d,
                           cmap=cmap, norm=norm, shading='flat')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(unit_label, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{ax1_name} (AU)", fontsize=9)
        ax.set_ylabel(f"{ax2_name} (AU)", fontsize=9)
        ax.set_aspect('equal')

    panel(axes[0], rho,       "Density",     'inferno', r'$\rho$ (g cm$^{-3}$)',
          vmin=scales['rho'][0],  vmax=scales['rho'][1],  log=True)
    panel(axes[1], temp,      "Temperature", 'plasma',  r'$T$ (K)',
          vmin=scales['temp'][0], vmax=scales['temp'][1], log=True)
    panel(axes[2], vel / 1e5, "Velocity",    'viridis', r'$|v|$ (km s$^{-1}$)',
          vmin=scales['vel'][0],  vmax=scales['vel'][1],  log=False)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"slice_{plane_name.lower()}_{snap_idx:04d}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fname}")
    if show:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    parser = argparse.ArgumentParser(description="2D cross-section cuts of Walicxe3D output")
    parser.add_argument("--data",    default=None,   help="Data directory")
    parser.add_argument("--snap",    type=int, default=None, help="Snapshot index")
    parser.add_argument("--nprocs",  type=int, default=None, help="MPI process count")
    parser.add_argument("--domain",  type=float, default=None, help="Domain size in AU")
    parser.add_argument("--ncells",  type=int, default=16, help="Cells per block (default: 16)")
    parser.add_argument("--maxlev",  type=int, default=5,  help="Max AMR levels (default: 5)")
    parser.add_argument("--res",     type=int, default=512, help="Output grid resolution (default: 512)")
    parser.add_argument("--test",      action="store_true",  help="Use single-wind test data")
    parser.add_argument("--out",       default=None,  help="Output directory")
    parser.add_argument("--show",      action="store_true",  help="Show plots interactively")
    parser.add_argument("--axis",      default="all", choices=["XY","XZ","YZ","all"],
                        help="Which slice plane (default: all)")
    parser.add_argument("--zoom",      type=float, default=None,
                        help="Show only central ±ZOOM AU around domain center (e.g. --zoom 15)")
    parser.add_argument("--all-snaps", action="store_true",
                        help="Process all snapshots with a shared global colorbar scale")
    args = parser.parse_args()

    # ── Defaults based on --test flag ─────────────────────────────────────────
    if args.test:
        datadir  = args.data  or os.path.join(root, "tests", "single_wind", "data")
        domain   = args.domain or 20.0
        nprocs   = args.nprocs or 4
        outdir   = args.out   or os.path.join(root, "tests", "single_wind", "images")
    else:
        datadir  = args.data  or os.path.join(root, "data")
        domain   = args.domain or 80.0
        nprocs   = args.nprocs or None  # auto-detect
        outdir   = args.out   or os.path.join(root, "resultados", "imagenes_xray")

    # ── Resolve snapshot list ──────────────────────────────────────────────────
    if args.all_snaps:
        snaps = find_all_snapshots(datadir)
        if not snaps:
            sys.exit(f"No snapshot files found in {datadir}")
        if args.snap is not None:
            sys.exit("--snap and --all-snaps are mutually exclusive")
    else:
        snap = args.snap
        if snap is None:
            snap = find_last_snapshot(datadir, nprocs or 0)
            if snap is None:
                sys.exit(f"No snapshot files found in {datadir}")
        snaps = [snap]

    if nprocs is None:
        nprocs = autodetect_nprocs(datadir, snaps[0])

    xphystot = yphystot = zphystot = domain * AU

    print(f"Data dir   : {datadir}")
    print(f"Snapshots  : {snaps}")
    print(f"MPI procs  : {nprocs}")
    print(f"Domain     : {domain:.0f} AU × {domain:.0f} AU × {domain:.0f} AU")
    print(f"Output dir : {outdir}")
    print()

    center = xphystot / 2.0
    slice_planes = {
        "XY": (2, center, "XY plane (z=center)", "X", "Y"),
        "XZ": (1, center, "XZ plane (y=center)", "X", "Z"),
        "YZ": (0, center, "YZ plane (x=center)", "Y", "Z"),
    }
    planes = list(slice_planes.keys()) if args.axis == "all" else [args.axis]

    def crop_zoom(rho, vel, temp, c1, c2):
        """Apply --zoom crop (same fraction for all snaps → consistent extent)."""
        if args.zoom is None:
            return rho, vel, temp, c1, c2
        half = args.zoom / domain
        lo_f = max(0.0, 0.5 - half)
        hi_f = min(1.0, 0.5 + half)
        n = rho.shape[0]
        i_lo = int(lo_f * n);  i_hi = int(hi_f * n)
        return (rho[i_lo:i_hi, i_lo:i_hi],
                vel[i_lo:i_hi, i_lo:i_hi],
                temp[i_lo:i_hi, i_lo:i_hi],
                c1[i_lo:i_hi+1],
                c2[i_lo:i_hi+1])

    # ── Pass 1: read all snapshots and build slices ────────────────────────────
    print(f"Pass 1/2 — reading {len(snaps)} snapshot(s) ...")
    all_slices = {}   # snap -> {plane_name: (rho, vel, temp, c1, c2)}
    for sn in snaps:
        print(f"  snap {sn:04d} ...", end=" ", flush=True)
        blocks = read_snapshot(datadir, sn, nprocs, neqtot=5, ncells=args.ncells)
        print(f"{len(blocks)} blocks", end="")
        all_slices[sn] = {}
        for name in planes:
            axis_idx, pos, _, _, _ = slice_planes[name]
            rho, vel, temp, c1, c2 = build_slice(
                blocks, axis_idx, pos,
                xphystot, yphystot, zphystot,
                args.maxlev, 1, 1, 1, args.ncells,
                resolution=args.res
            )
            rho, vel, temp, c1, c2 = crop_zoom(rho, vel, temp, c1, c2)
            all_slices[sn][name] = (rho, vel, temp, c1, c2)
        print()

    # ── Compute global color scale from all snapshots and planes ───────────────
    print("\nComputing global color scale ...")
    all_rho  = [all_slices[sn][n][0] for sn in snaps for n in planes]
    all_vel  = [all_slices[sn][n][1] for sn in snaps for n in planes]
    all_temp = [all_slices[sn][n][2] for sn in snaps for n in planes]
    scales = compute_global_scales(all_rho, all_vel, all_temp)
    print(f"  rho  : {scales['rho'][0]:.2e} – {scales['rho'][1]:.2e} g/cm³")
    print(f"  temp : {scales['temp'][0]:.2e} – {scales['temp'][1]:.2e} K")
    print(f"  vel  : {scales['vel'][0]:.1f} – {scales['vel'][1]:.1f} km/s")

    # ── Pass 2: render all images with the shared scale ────────────────────────
    print(f"\nPass 2/2 — rendering {len(snaps) * len(planes)} image(s) ...")
    os.makedirs(outdir, exist_ok=True)
    for sn in snaps:
        for name in planes:
            _, _, _, ax1_name, ax2_name = slice_planes[name]
            rho, vel, temp, c1, c2 = all_slices[sn][name]
            valid = np.sum(np.isfinite(rho))
            total = rho.size
            print(f"  snap {sn:04d} {name}  valid={100*valid/total:.1f}%", end=" ")
            plot_slice(rho, vel, temp, c1, c2, name, ax1_name, ax2_name,
                       sn, outdir, scales=scales, show=args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
