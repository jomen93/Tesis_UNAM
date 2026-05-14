#!/usr/bin/env bash
# run_wind_test.sh — Compile, run, and validate the single spherical wind test.
#
# Usage:
#   ./tests/run_wind_test.sh            # run with 4 MPI processes
#   ./tests/run_wind_test.sh --np 8     # override process count
#
# The script:
#   1. Backs up source/user.f90 and source/parameters.f90
#   2. Installs the test versions
#   3. Compiles walicxe3d
#   4. Runs the simulation into tests/single_wind/data/
#   5. Validates the output with validate_wind.py
#   6. Restores original source files (even on failure)
#
# Exit codes: 0 = pass, 1 = fail

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$SCRIPT_DIR")"
SOURCE="$PROJECT/source"
TEST_DIR="$SCRIPT_DIR/single_wind"
DATA_DIR="$TEST_DIR/data"

# ── Options ───────────────────────────────────────────────────────────────────
NP=4
SKIP_VALIDATE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --np)             NP="$2"; shift 2 ;;
    --skip-validate)  SKIP_VALIDATE=1; shift ;;
    *)    echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[test] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

# ── Cleanup on exit (always restores originals) ───────────────────────────────
cleanup() {
  local ec=$?
  log "Restoring original source files ..."
  [[ -f "$SOURCE/user.f90.orig"       ]] && mv "$SOURCE/user.f90.orig"       "$SOURCE/user.f90"
  [[ -f "$SOURCE/parameters.f90.orig" ]] && mv "$SOURCE/parameters.f90.orig" "$SOURCE/parameters.f90"
  if [[ $ec -eq 0 ]]; then
    log "Recompiling production binary ..."
    cd "$PROJECT" && make -s clean && make -s
    log "Production binary restored."
  else
    log "Test failed — production binary NOT recompiled automatically."
    log "Run 'make clean && make' manually to restore."
  fi
}
trap cleanup EXIT

# ── 0. Pre-flight checks ──────────────────────────────────────────────────────
cd "$PROJECT"
[[ -f "$TEST_DIR/user_test.f90"       ]] || fail "Missing: tests/single_wind/user_test.f90"
[[ -f "$TEST_DIR/parameters_test.f90" ]] || fail "Missing: tests/single_wind/parameters_test.f90"
[[ -f "$SCRIPT_DIR/validate_wind.py"  ]] || fail "Missing: tests/validate_wind.py"

mkdir -p "$DATA_DIR"
rm -f "$DATA_DIR"/*.bin "$DATA_DIR"/*.dat "$DATA_DIR"/*.log "$DATA_DIR"/*.vtk 2>/dev/null || true

# ── 1. Swap source files ──────────────────────────────────────────────────────
log "Installing test source files ..."
cp "$SOURCE/user.f90"       "$SOURCE/user.f90.orig"
cp "$SOURCE/parameters.f90" "$SOURCE/parameters.f90.orig"
cp "$TEST_DIR/user_test.f90"       "$SOURCE/user.f90"
cp "$TEST_DIR/parameters_test.f90" "$SOURCE/parameters.f90"

# ── 2. Compile ────────────────────────────────────────────────────────────────
log "Compiling ..."
make -s clean
make -s 2>&1 | tee "$DATA_DIR/build.log"
[[ -f "$PROJECT/walicxe3d" ]] || fail "Compilation failed. See $DATA_DIR/build.log"
log "Compilation OK."

# ── 3. Run ────────────────────────────────────────────────────────────────────
log "Running simulation with $NP MPI processes ..."
log "(domain: 20 AU, 128^3, tfin=0.2 yr — expect ~30-60 minutes)"
START=$(date +%s)

mpirun -np "$NP" "$PROJECT/walicxe3d" 2>&1 | tee "$DATA_DIR/run.log"

END=$(date +%s)
ELAPSED=$(( END - START ))
log "Simulation finished in ${ELAPSED}s."

# ── 4. Quick log scan for fatal errors ───────────────────────────────────────
if grep -q "MPI_ABORT\|Nan in FC\|Nan in GC\|Nan in HC\|zero density" "$DATA_DIR/run.log" 2>/dev/null; then
  echo ""
  echo "  Fatal errors detected in run.log:"
  grep "MPI_ABORT\|Nan in\|zero density" "$DATA_DIR/run.log" | head -10
  fail "Simulation aborted — see $DATA_DIR/run.log"
fi

# Check that at least one snapshot was written
N_SNAPS=$(find "$DATA_DIR" -name "Blocks000.????.bin" | wc -l)
[[ "$N_SNAPS" -gt 0 ]] || fail "No snapshot files generated."
log "Found $N_SNAPS snapshot(s)."

# ── 5. Validate ───────────────────────────────────────────────────────────────
if [[ $SKIP_VALIDATE -eq 1 ]]; then
  log "Skipping physical validation (--skip-validate)."
else
  log "Running physical validation ..."
  python3 "$SCRIPT_DIR/validate_wind.py"
fi
