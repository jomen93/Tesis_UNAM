!===============================================================================
! user_test.f90 — Two static spherical winds (no orbital motion)
!
! Physical setup:
!   Wind 1 (WR-like, strong): x = 4 AU, centered in Y and Z
!   Wind 2 (O-like,  weak  ): x = 16 AU, centered in Y and Z
!   Separation: 12 AU — collision front expected near x ~ 13 AU
!
!   Mdot ratio = 10:1 with equal v_inf → stagnation at r2 ~ 2.9 AU from wind 2
!   Ram-pressure balance: Mdot1/r1^2 = Mdot2/r2^2  (same v_inf)
!===============================================================================

module userconds

  use parameters
  use globals
  use winds
  implicit none

  type(spherical_wind_type) :: wind1, wind2

  ! Wind 1 — strong (WR-like)
  real, parameter :: w1_mdot   = 1.0e-5 * MSUN/YR   ! 1e-5 Msun/yr
  real, parameter :: w1_vinf   = 1000.0 * KPS        ! 1000 km/s
  real, parameter :: w1_radius = 1.0 * AU             ! 1 AU injection radius
  real, parameter :: w1_temp   = 1.0e6               ! 1 MK

  ! Wind 2 — weak (O-like)
  real, parameter :: w2_mdot   = 1.0e-6 * MSUN/YR   ! 1e-6 Msun/yr  (ratio 10:1)
  real, parameter :: w2_vinf   = 1000.0 * KPS        ! 1000 km/s
  real, parameter :: w2_radius = 1.0 * AU
  real, parameter :: w2_temp   = 1.0e6

contains

  subroutine userInitialCondition(uvars)
    implicit none
    real, intent(inout) :: uvars(nbMaxProc, neqtot, &
                           nxmin:nxmax, nymin:nymax, nzmin:nzmax)

    ! Wind 1 — left side of domain (x = 4 AU)
    wind1%xc     = 4.0 * AU
    wind1%yc     = yphystot / 2.0
    wind1%zc     = zphystot / 2.0
    wind1%vx     = 0.0;  wind1%vy = 0.0;  wind1%vz = 0.0
    wind1%radius = w1_radius
    wind1%mdot   = w1_mdot
    wind1%vinf   = w1_vinf
    wind1%temp   = w1_temp
    wind1%mu     = mui

    ! Wind 2 — right side of domain (x = 16 AU)
    wind2%xc     = 16.0 * AU
    wind2%yc     = yphystot / 2.0
    wind2%zc     = zphystot / 2.0
    wind2%vx     = 0.0;  wind2%vy = 0.0;  wind2%vz = 0.0
    wind2%radius = w2_radius
    wind2%mdot   = w2_mdot
    wind2%vinf   = w2_vinf
    wind2%temp   = w2_temp
    wind2%mu     = mui

    call imposeSphericalWind(wind1, uvars)
    call imposeSphericalWind(wind2, uvars)

    write(logu,*) ""
    write(logu,'(1x,a)') "=== TWO STATIC WINDS TEST ==="
    write(logu,'(1x,a,f5.1,a)') "Wind 1 (WR-like)  x = ", 4.0, " AU"
    write(logu,'(1x,a,es10.3,a,f7.1,a)') &
      "  Mdot=", w1_mdot, " g/s  vinf=", w1_vinf/1e5, " km/s"
    write(logu,'(1x,a,f5.1,a)') "Wind 2 (O-like )  x = ", 16.0, " AU"
    write(logu,'(1x,a,es10.3,a,f7.1,a)') &
      "  Mdot=", w2_mdot, " g/s  vinf=", w2_vinf/1e5, " km/s"
    write(logu,'(1x,a,f5.2,a)') "Expected stagnation near x ~ ", 13.1, " AU"
    write(logu,'(1x,a)') "============================="

  end subroutine userInitialCondition

  subroutine userBoundary(uvars)
    implicit none
    real, intent(inout) :: uvars(nbMaxProc, neqtot, &
                           nxmin:nxmax, nymin:nymax, nzmin:nzmax)

    ! Re-impose both winds every timestep (positions fixed — no orbital motion)
    call imposeSphericalWind(wind1, uvars)
    call imposeSphericalWind(wind2, uvars)

  end subroutine userBoundary

end module userconds
