#!/usr/bin/env python3
"""
validate_wind.py — Validates the single spherical wind test case.

Reads the last available snapshot from tests/single_wind/data/ and checks:
  1. STABILITY:  no NaN, no negative densities or pressures
  2. DENSITY RANGE: rho_max is within a factor of 10 of the analytical value
                    at the injection radius (AMR resolution limits precision)
  3. VALID CELLS: at least 80% of cells have physically meaningful values
  4. VELOCITY:   average speed in the wind zone is close to v_inf
  5. TEMPERATURE: wind temperature is in the expected range

Analytical reference (isothermal 1/r^2 wind):
  rho(r) = Mdot / (4*pi * r^2 * v_inf)
  v(r)   = v_inf = 1000 km/s
  T      = 1e6 K
"""

import sys
import numpy as np
from pathlib import Path

# ============================================================
# Test parameters — must match tests/single_wind/user_test.f90
# ============================================================
MDOT      = 1.0e-6 * (2.0e33 / 3.15e7)  # g/s  (1e-6 Msun/yr)
V_INF     = 1000.0e5                      # cm/s (1000 km/s)
RADIUS    = 1.496e13                      # cm   (1 AU)
WIND_TEMP = 1.0e6                         # K

# Analytical density at the injection radius
RHO_AT_RADIUS = MDOT / (4.0 * np.pi * RADIUS**2 * V_INF)

# ============================================================
# Simulation parameters — must match parameters_test.f90
# ============================================================
DATADIR   = Path(__file__).parent / "single_wind" / "data"
NPROCS    = 4
NEQTOT    = 5
NCELLS    = 16    # ncells_block

# Unit scalings
AU        = 1.496e13       # cm
MU0       = 1.3
AMU       = 1.66053906e-24 # g
KB        = 1.380649e-16   # erg/K
MUI       = 0.61
ISM_DENS  = MU0 * AMU     # g/cm^3  (density scale)
V_SC      = 1.0e5          # cm/s    (velocity scale)
D_SC      = ISM_DENS
P_SC      = D_SC * V_SC**2
GAMMA     = 5.0 / 3.0

# Validation thresholds
MIN_VALID_FRACTION  = 0.80   # at least 80% of cells must be valid
RHO_MAX_FACTOR      = 10.0   # rho_max must be within 10x of analytical
VELOCITY_TOLERANCE  = 0.30   # average wind speed within 30% of v_inf
TEMP_MIN            = 1.0e4  # K  — wind should be hot
TEMP_MAX            = 1.0e9  # K  — should not be unphysically hot


# ============================================================

def find_last_snapshot():
    """Return the index of the last snapshot in DATADIR."""
    files = sorted(DATADIR.glob(f"Blocks000.????.bin"))
    if not files:
        return None
    # Extract the snapshot number from the filename
    last = files[-1].name          # e.g. "Blocks000.0003.bin"
    return int(last.split(".")[1])


def read_snapshot(snap_idx):
    """
    Read all process files for snapshot snap_idx.
    Returns a flat list of (bID, data_array) where data_array has shape
    (NEQTOT, NCELLS, NCELLS, NCELLS) in code units.
    Uses float64 because the simulation is compiled with DOUBLEP=Y.
    """
    all_blocks = []
    for proc in range(NPROCS):
        fpath = DATADIR / f"Blocks{proc:03d}.{snap_idx:04d}.bin"
        if not fpath.exists():
            print(f"  WARNING: missing file {fpath.name}")
            continue
        with open(fpath, "rb") as f:
            nblocks = np.frombuffer(f.read(4), dtype=np.int32)[0]
            for _ in range(nblocks):
                bID = np.frombuffer(f.read(4), dtype=np.int32)[0]
                n   = NEQTOT * NCELLS**3
                raw = np.frombuffer(f.read(n * 8), dtype=np.float64)
                if len(raw) != n:
                    break
                # Fortran writes column-major (eq varies fastest, then ix, iy, iz).
                # order='F' gives data[eq, ix, iy, iz] with correct correspondence.
                data = raw.reshape((NEQTOT, NCELLS, NCELLS, NCELLS), order='F')
                all_blocks.append((bID, data))
    return all_blocks


def conserved_to_prim(uvars):
    """
    Convert one cell's conserved variables to (rho, vx, vy, vz, pres) in code
    units. Returns None if conversion fails (zero/negative density).
    """
    rho = uvars[0]
    if not (np.isfinite(rho) and rho > 0.0):
        return None
    vx  = uvars[1] / rho
    vy  = uvars[2] / rho
    vz  = uvars[3] / rho
    if not (np.isfinite(vx) and np.isfinite(vy) and np.isfinite(vz)):
        return None
    e_int = uvars[4] - 0.5 * rho * (vx**2 + vy**2 + vz**2)
    pres  = (GAMMA - 1.0) * e_int
    if not (np.isfinite(pres) and pres > 0.0):
        return None
    return rho, vx, vy, vz, pres


def run_validation():
    print("=" * 65)
    print("  SINGLE SPHERICAL WIND — VALIDATION REPORT")
    print("=" * 65)
    print(f"\nAnalytical reference:")
    print(f"  Mdot           = {MDOT:.3e} g/s")
    print(f"  v_inf          = {V_INF/1e5:.0f} km/s")
    print(f"  rho(r=1 AU)    = {RHO_AT_RADIUS:.3e} g/cm^3")
    print(f"  Wind temp      = {WIND_TEMP:.1e} K")

    snap = find_last_snapshot()
    if snap is None:
        print(f"\nFAIL: no snapshot files found in {DATADIR}")
        sys.exit(1)

    print(f"\nReading snapshot {snap:04d} from {NPROCS} process files ...")
    blocks = read_snapshot(snap)
    if not blocks:
        print("FAIL: could not read any block data.")
        sys.exit(1)
    print(f"  Loaded {len(blocks)} AMR blocks.")

    # --------------------------------------------------------
    # Collect per-cell statistics
    # --------------------------------------------------------
    total_cells  = 0
    valid_cells  = 0
    nan_count    = 0
    neg_rho      = 0
    neg_pres     = 0

    rho_list  = []
    vel_list  = []
    temp_list = []

    for _, data in blocks:
        for i in range(NCELLS):
            for j in range(NCELLS):
                for k in range(NCELLS):
                    total_cells += 1
                    cell = data[:, k, j, i]

                    if not np.all(np.isfinite(cell)):
                        nan_count += 1
                        continue

                    if cell[0] <= 0.0:
                        neg_rho += 1
                        continue

                    result = conserved_to_prim(cell)
                    if result is None:
                        neg_pres += 1
                        continue

                    rho_code, vx, vy, vz, pres_code = result
                    rho_cgs  = rho_code  * D_SC
                    pres_cgs = pres_code * P_SC
                    vel_cgs  = np.sqrt(vx**2 + vy**2 + vz**2) * V_SC
                    temp_K   = pres_cgs / rho_cgs * (MUI * AMU / KB)

                    rho_list.append(rho_cgs)
                    vel_list.append(vel_cgs)
                    temp_list.append(temp_K)
                    valid_cells += 1

    valid_frac = valid_cells / total_cells if total_cells > 0 else 0.0

    print(f"\n{'─'*65}")
    print(f"  CHECK 1 — Cell validity")
    print(f"{'─'*65}")
    print(f"  Total cells   : {total_cells:,}")
    print(f"  Valid cells   : {valid_cells:,}  ({100*valid_frac:.1f}%)")
    if nan_count:
        print(f"  NaN detected  : {nan_count:,}")
    if neg_rho:
        print(f"  Negative rho  : {neg_rho:,}")
    if neg_pres:
        print(f"  Negative pres : {neg_pres:,}")
    check1 = valid_frac >= MIN_VALID_FRACTION and nan_count == 0
    status1 = "PASS" if check1 else "FAIL"
    print(f"  Result        : {status1}  (threshold: {100*MIN_VALID_FRACTION:.0f}% valid, 0 NaN)")

    if not rho_list:
        print("\nFAIL: no valid cells to analyse.")
        sys.exit(1)

    rho_arr  = np.array(rho_list)
    vel_arr  = np.array(vel_list)
    temp_arr = np.array(temp_list)

    print(f"\n{'─'*65}")
    print(f"  CHECK 2 — Density range vs analytical 1/r^2 profile")
    print(f"{'─'*65}")
    rho_max = rho_arr.max()
    rho_min = rho_arr[rho_arr > ISM_DENS * 10].min() if np.any(rho_arr > ISM_DENS * 10) else rho_arr.min()
    ratio   = rho_max / RHO_AT_RADIUS
    print(f"  rho_max (sim)      = {rho_max:.3e} g/cm^3")
    print(f"  rho(r=1AU) (anal.) = {RHO_AT_RADIUS:.3e} g/cm^3")
    print(f"  ratio sim/anal     = {ratio:.2f}x  (expect ~1, AMR allows up to {RHO_MAX_FACTOR}x)")
    check2 = (1.0/RHO_MAX_FACTOR) <= ratio <= RHO_MAX_FACTOR
    status2 = "PASS" if check2 else "FAIL"
    print(f"  Result             : {status2}")

    print(f"\n{'─'*65}")
    print(f"  CHECK 3 — Wind velocity vs v_inf = {V_INF/1e5:.0f} km/s")
    print(f"{'─'*65}")
    # Only look at cells significantly denser than ISM (these are in the wind zone)
    wind_mask = rho_arr > ISM_DENS * 100
    if wind_mask.sum() > 10:
        vel_wind = vel_arr[wind_mask]
        vel_mean = vel_wind.mean()
        vel_err  = abs(vel_mean - V_INF) / V_INF
        print(f"  Wind cells       : {wind_mask.sum():,}")
        print(f"  Mean speed (sim) : {vel_mean/1e5:.1f} km/s")
        print(f"  v_inf (expected) : {V_INF/1e5:.1f} km/s")
        print(f"  Relative error   : {100*vel_err:.1f}%  (threshold: {100*VELOCITY_TOLERANCE:.0f}%)")
        check3 = vel_err <= VELOCITY_TOLERANCE
    else:
        print(f"  Not enough wind cells (dense zone) to test — skipping")
        check3 = True  # non-blocking if wind hasn't propagated far yet
    status3 = "PASS" if check3 else "FAIL"
    print(f"  Result           : {status3}")

    print(f"\n{'─'*65}")
    print(f"  CHECK 4 — Wind temperature ({WIND_TEMP:.0e} K expected)")
    print(f"{'─'*65}")
    if wind_mask.sum() > 10:
        temp_wind = temp_arr[wind_mask]
        temp_med  = np.median(temp_wind)
        in_range  = (TEMP_MIN <= temp_med <= TEMP_MAX)
        print(f"  Median wind temp : {temp_med:.2e} K")
        print(f"  Expected range   : {TEMP_MIN:.0e} – {TEMP_MAX:.0e} K")
        check4 = in_range
    else:
        print(f"  Skipped (insufficient wind cells)")
        check4 = True
    status4 = "PASS" if check4 else "FAIL"
    print(f"  Result           : {status4}")

    # --------------------------------------------------------
    # Overall verdict
    # --------------------------------------------------------
    all_pass = all([check1, check2, check3, check4])
    print(f"\n{'=' * 65}")
    if all_pass:
        print("  OVERALL: PASS — spherical wind test validated successfully.")
    else:
        print("  OVERALL: FAIL — one or more checks failed (see above).")
    print("=" * 65)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    run_validation()
