!===============================================================================
! parameters_test.f90 — Parameters for single spherical wind validation test
!
! Differences from production parameters.f90:
!   - Small domain (20 AU), low resolution (128^3), 4 MPI processes
!   - Short run time (0.02 yr = ~7 days physical)
!   - No radiative cooling (avoids needing cooling tables)
!   - Standard numerical parameters (not the extreme stability settings of WR 140)
!   - Data written to tests/single_wind/data/
!===============================================================================

module parameters
  use constants
#ifdef MPIP
  use mpi
#endif

  implicit none

  ! ============================================
  ! Execution parameters
  ! ============================================

  real, parameter :: tfin  = 0.15 * YR   !< ~3 half-crossing times (5 AU / 1000 km/s ≈ 0.047 yr → steady collision by 0.15 yr)
  real, parameter :: dtout = 0.003 * YR  !< 50 output snapshots

  logical, parameter :: dowarm = .false.
  character(*), parameter :: warm_file = ""

  integer, parameter :: nProcs = 4
  real, parameter :: RAM_per_proc = 500.0

  ! ============================================
  ! Adaptive Mesh parameters
  ! ============================================

  real, parameter :: xphystot = 20.0 * AU
  real, parameter :: yphystot = 20.0 * AU
  real, parameter :: zphystot = 20.0 * AU

  integer, parameter :: mesh_method = MESH_AUTO

  integer, parameter :: p_maxcells_x = 128
  integer, parameter :: p_maxcells_y = 128
  integer, parameter :: p_maxcells_z = 128

  integer, parameter :: p_nbrootx = 1
  integer, parameter :: p_nbrooty = 1
  integer, parameter :: p_nbrootz = 1
  integer, parameter :: p_maxlev  = 5

  integer, parameter :: ncells_block = 16

  real, parameter :: refineThres = 0.3
  real, parameter :: coarseThres = 0.03

  ! ============================================
  ! Boundary conditions (free outflow on all faces)
  ! ============================================

  integer, parameter :: bc_left   = BC_FREE_FLOW
  integer, parameter :: bc_right  = BC_FREE_FLOW
  integer, parameter :: bc_front  = BC_FREE_FLOW
  integer, parameter :: bc_back   = BC_FREE_FLOW
  integer, parameter :: bc_bottom = BC_FREE_FLOW
  integer, parameter :: bc_top    = BC_FREE_FLOW

  ! ============================================
  ! Data output
  ! ============================================

  logical, parameter :: output_bin = .true.
  logical, parameter :: output_vtk = .false.

  integer, parameter :: output_mode = OUT_SIMULT
  integer, parameter :: units_type  = PHYS_UNITS

  character(*), parameter :: datadir   = &
    "/home/johan/Documentos/Personal/Tesis_UNAM/tests/single_wind/data"
  character(*), parameter :: blockstpl = "BlocksXXX.YYYY"
  character(*), parameter :: gridtpl   = "Grid.YYYY"
  character(*), parameter :: statetpl  = "State.YYYY"
  character(*), parameter :: paramfile = datadir//"/parameter.dat"

  logical, parameter :: logged = .true.
  character(*), parameter :: logdir = datadir

  ! ============================================
  ! Solver — standard settings (not WR 140 extreme)
  ! ============================================

  integer, parameter :: solver_type  = SOLVER_HLLC
  integer, parameter :: limiter_type = LIMITER_VANLEER
  integer, parameter :: nghost       = 2
  integer, parameter :: npassive     = 0

  real, parameter :: CFL      = 0.3
  real, parameter :: visc_eta = 5.0E-3

  ! ============================================
  ! Radiative Cooling — disabled for the test
  ! (avoids requiring cooling table files)
  ! ============================================

  integer, parameter :: cooling_type  = COOL_NONE
  character(*), parameter :: cooling_file = ""
  real, parameter :: cooling_limit = 0.1

  ! ============================================
  ! ISM (ambient medium) — low density, cold
  ! ============================================

  real, parameter :: gamma    = 5.0/3.0
  real, parameter :: mu0      = 1.3
  real, parameter :: mui      = 0.61
  real, parameter :: ion_thres = 1.0e4

  real, parameter :: ism_mu0  = mu0
  real, parameter :: ism_dens = 1.0 * ism_mu0 * AMU
  real, parameter :: ism_temp = 100.0
  real, parameter :: ism_vx   = 0.0
  real, parameter :: ism_vy   = 0.0
  real, parameter :: ism_vz   = 0.0
  real, parameter :: ism_metal = 1.0

#ifdef PASBP
  real, parameter :: ism_bx = 0.0
  real, parameter :: ism_by = 0.0
  real, parameter :: ism_bz = 0.0
#endif

  ! ============================================
  ! Unit scalings
  ! ============================================

  real, parameter :: l_sc   = 1.0*AU
  real, parameter :: d_sc   = ism_dens
  real, parameter :: v_sc   = 1.0e5
  real, parameter :: pas_sc = d_sc

  ! ============================================================================
  !                    Derived parameters — do not modify
  ! ============================================================================

  integer, parameter :: neqhydro = 5
#ifdef PASBP
  integer, parameter :: neqmhd = 3
#else
  integer, parameter :: neqmhd = 0
#endif
  integer, parameter :: metalpas = neqhydro + neqmhd + min(npassive,1)
  integer, parameter :: neqtot   = neqhydro + neqmhd + npassive
  integer, parameter :: firstpas = neqhydro + neqmhd + 1

#ifdef DOUBLEP
  integer, parameter :: bytes_per_real = 8
#else
  integer, parameter :: bytes_per_real = 4
#endif

  integer, parameter :: block_ram_size = &
    (ncells_block+2*nghost)**3 * neqtot * bytes_per_real

  integer, parameter :: nbMaxProc = &
    int((RAM_per_proc*1024*1024 - 6*block_ram_size)*0.95/(block_ram_size*3))

  integer, parameter :: nbMaxGlobal = nbMaxProc*nProcs

  integer, parameter :: ncells_x = ncells_block
  integer, parameter :: ncells_y = ncells_block
  integer, parameter :: ncells_z = ncells_block

  integer, parameter :: nxmin = 1-nghost
  integer, parameter :: nxmax = ncells_x+nghost
  integer, parameter :: nymin = 1-nghost
  integer, parameter :: nymax = ncells_y+nghost
  integer, parameter :: nzmin = 1-nghost
  integer, parameter :: nzmax = ncells_z+nghost

#ifdef MPIP
#ifdef DOUBLEP
  integer, parameter :: mpi_real_kind = MPI_DOUBLE_PRECISION
#else
  integer, parameter :: mpi_real_kind = MPI_REAL
#endif
#endif

  real, parameter :: CV = 1.0/(gamma-1.0)
  integer, parameter :: master = 0

  real, parameter :: p_sc = d_sc*v_sc**2
  real, parameter :: e_sc = p_sc
  real, parameter :: t_sc = l_sc/v_sc

  integer, parameter :: param = 11

  contains

  subroutine writeparameters()
    implicit none
    open(unit=param, file=paramfile, status="replace")
    write(param,'(1x,a,es12.5)') "x-size", xphystot
    write(param,'(1x,a,i0)')     "Max-levelCellsAlong_x ", p_maxcells_x
    write(param,'(1x,a,i0)')     "Max-levelCellsAlong_y ", p_maxcells_y
    write(param,'(1x,a,i0)')     "Max-levelCellsAlong_z ", p_maxcells_z
    write(param,'(1x,a,f6.3)')   "CFL_parameter ", CFL
    write(param,'(1x,a,es12.5)') "Length ", l_sc
    write(param,'(1x,a,es12.5)') "Density ", d_sc
    write(param,'(1x,a,es12.5)') "Velocity ", v_sc
    write(param,'(1x,a,es12.5)') "Pressure ", p_sc
    write(param,'(1x,a,es12.5)') "Time ", t_sc
    write(param,'(1x,a,es12.5)') "AMU ", AMU, " g"
    close(param)
  end subroutine writeparameters

end module parameters
