!===============================================================================
!> @file windsource.f90
!> @brief Boundary condition module: spherical wind source
!> @author Alejandro Esquivel and Juan C. Toledo
!> @date 17/Apr/2013

! Copyright (c) 2014 Juan C. Toledo and Alejandro Esquivel
!
! This file is part of Walicxe3D.
!
! Walicxe3D is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.

! You should have received a copy of the GNU General Public License
! along with this program.  If not, see http://www.gnu.org/licenses/.

!===============================================================================

!> @brief Imposes a spherical wind source
!> @details This user-defined module establishes the parameters, 
!! variables and subroutines required to impose a spherical wind source.
!!
!! How to use this module.
!! The module currently supports
!! 1) The parameters of each wind must be defined by creating an object of
!!    the appropriate type. The currently supported wind types are:
!!      type_spherical_wind : a spherical r^-2 wind
!!      type_plane_wind : a planar wind (to be used as in inflow condition)
!!    Then, the user must specify the corresponding wind parameters.
!! 2) The user must modify the userBC() subroutine in the user.f90 source
!!    file, adding a call to the corresponding subroutine: 

module winds

  implicit none

  !===============================
  ! Spherical wind source parameters
  !
  ! ALL VALUES IN CGS
  ! xc, yc, zc: position of the source center
  ! vx, vy, vz: systematic velocity components of the wind source
  ! radius: radius of the wind source
  ! mdot: mass loss rate
  ! vinf: terminal speed
  ! temp: temperature
  ! mu: gas mean atomic mass (in amus)
  ! -- Optional fields --
  ! bx, by, bz: magnetic field components (for runs with passive B field)
  ! metal: metallicity of the gas (for runs with metallicity-dependent cooling)
  type spherical_wind_type
    real :: xc, yc, zc
    real :: vx, vy, vz
    real :: radius
    real :: mdot
    real :: vinf
    real :: temp
    real :: mu
    real :: bx = 0.0
    real :: by = 0.0
    real :: bz = 0.0
    real :: metal = 1.0
  end type spherical_wind_type
  !===============================

  !===============================
  ! Plane wind source definition
  !
  ! ALL VALUES IN CGS
  ! plane: the plane upon which the wind enters the simulation.
  ! Must be one of the following constants:
  !   PLANE_LEFT: the YZ plane crossing at X=0
  !   PLANE_RIGHT: the YZ plane crossing at X=size_x
  !   PLANE_FRONT: the XZ plane crossing at Y=0
  !   PLANE_BACK: the XZ plane crossing at Y=size_y
  !   PLANE_BOTTOM: the XY plane crossing at Z=0
  !   PLANE_TOP: the XY plane crossing at Z=size_z
  ! rho: density of the flow to be imposed on the plane
  ! vel: velocity *magnitude* of the flow (the sign will be calculated
  !      based on the plane of entry)
  ! temps: temperature of the flow
  ! mu: gas mean atomic mass (in amus)
  ! -- Optional fields --
  ! bx, by, bz: magnetic field components (for runs with passive B field)
  ! metal: metallicity of the gas (for runs with metallicity-dependent cooling)
  integer, parameter :: PLANE_LEFT   = 1
  integer, parameter :: PLANE_RIGHT  = 2
  integer, parameter :: PLANE_FRONT  = 3
  integer, parameter :: PLANE_BACK   = 4
  integer, parameter :: PLANE_BOTTOM = 5
  integer, parameter :: PLANE_TOP    = 6
  type plane_wind_type
    integer :: plane
    real :: rho
    real :: vel
    real :: temp
    real :: mu
    real :: bx = 0.0
    real :: by = 0.0
    real :: bz = 0.0
    real :: metal = 1.0
  end type plane_wind_type
  !===============================
  
contains

  ! ============================================
  !> @brief Imposes a spherical wind source on the simulation
  !> @details Simulates a spherical wind source by setting the flow
  !! conditions in a region of given radius centered at (xc,yc,zc), 
  !! in which a steady-state wind solution is imposed with the given
  !! mass-loss rate, terminal speed and temperature.
  !> @param wind_params A spherical_wind_type variable containing the
  !! wind parameters. See the module documentation for further details.
  !> @param uvars Flow variables array to be modified
  subroutine imposeSphericalWind (wind_params, uvars)
 
    use parameters
    use globals
    implicit none

    type(spherical_wind_type), intent(in) :: wind_params
    real, intent(inout) :: uvars(nbMaxProc, neqtot, &
                           nxmin:nxmax, nymin:nymax, nzmin:nzmax)

    integer :: nb, bID, i, j, k
    real :: xc, yc, zc, vwx, vwy, vwz, radius, mdot, vinf, temp
    real :: dens, vx, vy, vz, pres, x, y, z, dist, mu, metal
    real :: dist_min, safe_dist  ! For limiting 1/r^2 divergence
    real :: r_inner, r_outer, blend_weight, t  ! For smooth transition zone
    real :: primit_wind(neqtot), primit_current(neqtot), primit_mixed(neqtot)
    real :: primit(neqtot)
    real :: zone(6)
    integer :: zlevel
    integer :: eq, istat

    ! Unpack wind source parameters
    xc = wind_params%xc
    yc = wind_params%yc
    zc = wind_params%zc
    vwx = wind_params%vx
    vwy = wind_params%vy
    vwz = wind_params%vz
    
    radius = wind_params%radius
    mdot = wind_params%mdot
    vinf = wind_params%vinf
    temp = wind_params%temp
    mu = wind_params%mu
    metal = wind_params%metal

    ! Define smooth transition zone (Enfoque B)
    ! Inner radius: 70% of injection radius (pure wind zone)
    ! Outer radius: 100% of injection radius (transition boundary)
    r_inner = 0.7 * radius
    r_outer = radius

    ! Report parameters and refine the zone around the wind source only once
    if (it.eq.0) then
      write(logu,*) ""
      write(logu,'(1x,a)') "> Imposing spherical wind source ..."
      write(logu,'(1x,a,es12.5)') "Mdot = ", mdot
      write(logu,'(1x,a,es12.5)') "vinf = ", vinf
      write(logu,'(1x,a,es12.5)') "Radius = ", radius
      write(logu,'(1x,a,es12.5)') "Rho(R) = ", mdot/vinf/radius/radius/(4*PI)
      write(logu,'(1x,a,es12.5)') "Temp = ", temp
      write(logu,'(1x,a,es12.5,1x,es12.5,1x,es12.5)') "Location: ", xc, yc, zc
      zone(1) = xc - radius
      zone(2) = xc + radius
      zone(3) = yc - radius
      zone(4) = yc + radius
      zone(5) = zc - radius
      zone(6) = zc + radius
      zlevel = maxlev
      call refineZone (zone, zlevel)
    end if

    ! Impose flow conditions, where applicable
    do nb=1,nbMaxProc
      bID = localBlocks(nb)
      if (bID.ne.-1) then

        do i=nxmin,nxmax
          do j=nymin,nymax
            do k=nzmin,nzmax

              call cellPos (bID, i, j, k, x, y, z)
              x = x*l_sc;  y = y*l_sc;  z = z*l_sc
              dist = sqrt((x-xc)**2+(y-yc)**2+(z-zc)**2)   ! phys units

              ! ENFOQUE B: SMOOTH TRANSITION ZONE
              ! Only process cells within outer radius (includes transition zone)
              if (dist.lt.r_outer) then

                  ! === 1. Calculate WIND primitives ===
                  ! CRITICAL FIX: Limit minimum distance to avoid 1/r^2 divergence
                  dist_min = 0.5 * r_inner  ! Minimum distance = 50% of inner radius
                  safe_dist = max(dist, dist_min)

                  dens = mdot/vinf/safe_dist/safe_dist/(4.0*PI)
                  vx = vinf*(x-xc)/safe_dist + vwx
                  vy = vinf*(y-yc)/safe_dist + vwy
                  vz = vinf*(z-zc)/safe_dist + vwz
                  pres= dens/(mu*AMU)*KB*temp

                  ! Scale wind primitives
                  primit_wind(1) = dens/d_sc
                  primit_wind(2) = vx/v_sc
                  primit_wind(3) = vy/v_sc
                  primit_wind(4) = vz/v_sc
                  primit_wind(5) = pres/p_sc
#ifdef PASBP
                  ! Magnetic field
                  primit_wind(6) = 0.0
                  primit_wind(7) = 0.0
                  primit_wind(8) = 0.0
#endif
                  ! Passive scalar for metalicity
                  if (cooling_type.eq.COOL_TABLE_METAL) then
                    primit_wind(metalpas) = metal*primit_wind(1)
                  end if

                  ! === 2. Safety check and get CURRENT primitives ===
                  ! CRITICAL: Check if current state has valid density BEFORE calling flow2prim
                  ! (flow2prim aborts if density is zero)
                  if (uvars(nb,1,i,j,k).le.0.0) then
                    ! Current density is zero or negative - use 100% wind
                    blend_weight = 1.0
                  else
                    ! Get current primitives from cell
                    call flow2prim(uvars(nb,:,i,j,k), primit_current, istat)

                    ! === 3. Calculate blending weight ===
                    ! SAFETY: If conversion failed (istat != 0), use pure wind
                    if (istat.ne.0) then
                      ! Current state conversion failed - use 100% wind
                      blend_weight = 1.0
                    else if (dist.lt.r_inner) then
                      ! Pure wind zone (inside inner radius)
                      blend_weight = 1.0
                    else
                      ! Transition zone (between r_inner and r_outer)
                      ! Use smooth cubic Hermite function: weight(t) = 3t^2 - 2t^3
                      ! where t goes from 0 (at r_inner) to 1 (at r_outer)
                      t = (dist - r_inner) / (r_outer - r_inner)
                      ! Invert: weight=1 at r_inner, weight=0 at r_outer
                      blend_weight = 1.0 - (3.0*t*t - 2.0*t*t*t)
                      blend_weight = max(0.0, min(1.0, blend_weight))
                    end if
                  end if  ! Close the else block for valid density

                  ! === 4. Blend primitives ===
                  if (blend_weight.ge.0.9999) then
                    ! Pure wind (or nearly so) - no need to blend
                    do eq=1,neqtot
                      primit_mixed(eq) = primit_wind(eq)
                    end do
                  else
                    ! Blend: primit_mixed = weight * wind + (1 - weight) * current
                    do eq=1,neqtot
                      primit_mixed(eq) = blend_weight * primit_wind(eq) + &
                                         (1.0 - blend_weight) * primit_current(eq)
                    end do
                  end if

                  ! === 5. Apply blended solution to cell ===
                  call prim2flow(primit_mixed, uvars(nb,:,i,j,k))

              end if

            end do
          end do
        end do

      end if
    end do

  end subroutine imposeSphericalWind

! =============================================

  !> @brief Imposes a planar wind on one of the edges of the simulation box.
  !> @details Simulates a planar wind entering the simulation box
  !! by setting the flow conditions in the appropriate boundary layer. 
  !> @params wind_params A plane_winds_type variable containing the wind
  !! parameters. See the module documentation for further details.
  !> @param uvars Flow variables array to be modified
  subroutine imposePlaneWind (wind_params, uvars)
 
    use parameters
    use globals
    implicit none

    type(plane_wind_type), intent(in) :: wind_params
    real, intent(inout) :: uvars(nbMaxProc, neqtot, &
                           nxmin:nxmax, nymin:nymax, nzmin:nzmax)

    integer :: nb, bID, i, j, k, plane
    real :: dens, vx, vy, vz, pres, temp, vel, mu, metal
    real :: primit(neqtot)
    integer :: neighType
    integer :: neighList(4)

    write(logu,*) ""
    write(logu,'(1x,a)') "> Refreshing plane wind ..."

    ! Unpack wind source parameters
    plane = wind_params%plane
    dens  = wind_params%rho
    vel   = wind_params%vel
    temp  = wind_params%temp
    mu    = wind_params%mu
    metal = wind_params%metal

    ! Impose flow conditions on ghost cells of TOP simulation boundary
    do nb=1,nbMaxProc
      bID = localBlocks(nb)
      if (bID.ne.-1) then

        ! Determine if this block is at the TOP boundary
        call neighbors (bID, TOP, neighType, neighList)

        ! If so, set TOP *ghost* cells
        if (neighType.eq.NEIGH_BOUNDARY) then

          do i=nxmin,nxmax
            do j=nymin,nymax
              do k=ncells_z,nzmax

                ! Compute velocity components and pressure
                vx = 0.0
                vy = 0.0
                vz = 0.0
                select case(plane)
                  case (PLANE_LEFT)
                    vx = vel
                  case (PLANE_RIGHT)
                    vx = -vel
                  case (PLANE_FRONT)
                    vy = vel
                  case (PLANE_BACK)
                    vy = -vel
                  case (PLANE_BOTTOM)
                    vz = vel
                  case (PLANE_TOP)
                    vz = -vel
                end select
                pres = dens/(mu*AMU)*KB*temp

                ! Scale primitives
                primit(1) = dens/d_sc
                primit(2) = vx/v_sc
                primit(3) = vy/v_sc
                primit(4) = vz/v_sc
                primit(5) = pres/p_sc

                ! Passive scalar for metalicity
                if (cooling_type.eq.COOL_TABLE_METAL) then
                  primit(metalpas) = metal*primit(1)
                end if

                ! Convert primitives and set flow vars for this cell
                call prim2flow( primit, uvars(nb,:,i,j,k) )

              end do
            end do
          end do

        end if

      end if
    end do

  end subroutine imposePlaneWind


! =============================================

end module winds
