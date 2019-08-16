!
!  cart_pole.f90
!  cart-pole
!
!  Created by Jacopo Canton on 31/01/19
!  Copyright (c) 2019 Jacopo Canton. All rights reserved.
!
!  This file contains the cart pole test case for Smarties in Fortran.
!   - The first module, `class_cartPole`, defines the environment and the
!   simulation parameters.
!   - The second module, `fortran_smarties`, contains the interface to the C++
!   code as well as the main function `fortran_app_main` called by Smarties.
!
!==============================================================================

module class_cartPole

  implicit none
  private
  real,    parameter :: pi = 3.1415926535897931d0
  logical, parameter :: SWINGUP=.false.
  real,    parameter :: mp = 0.1
  real,    parameter :: mc = 1.
  real,    parameter :: l  = 0.5
  real,    parameter :: g  = 9.81
  integer, parameter :: STATE_SIZE = 6


  type, public :: cartPole
    ! Main class containing the simulation data and procedures
    real :: dt = 1e-4
    integer :: nsteps = 50
    integer :: info=1, step=0
    real, dimension(4) :: u
    real :: F=0, t=0
    !
    contains
      procedure :: reset     => reset
      procedure :: is_over   => is_over
      procedure :: advance   => advance
      procedure :: getState  => getState
      procedure :: getReward => getReward
  end type cartPole

  
contains
!------------------------------------------------------------------------------
! Definition of the procedures for cartPole

  subroutine reset(this) ! reset the simulation
    class(cartPole) :: this
    ! initialize the state to a random vector
    call random_number(this%u) ! random_number is an intrinsic [0, 1)
    if (SWINGUP) then
      this%u = (this%u-0.5)*2.*1. ! [-1, 1)
    else
      this%u = (this%u-0.5)*2.*0.05 ! [-0.05, 0.05)
    end if
    this%F = 0
    this%t = 0
    this%step = 0
    this%info=1
  end subroutine reset


  function is_over(this) result(answer) ! is the simulation over?
    class(cartPole), intent(in) :: this
    logical :: answer
    answer=.false.
    !
    if (SWINGUP) then
      if (this%step>=500 .or. abs(this%u(1))>2.4) answer = .true.
    else
      if (this%step>=500 .or. abs(this%u(1))>2.4 .or. abs(this%u(3))>pi/15 ) answer = .true.
    endif
  end function is_over


  function advance(this, action) result(terminated) ! advance the simulation in time
    use, intrinsic :: iso_c_binding
    class(cartPole) :: this
    real(c_double), dimension(:) :: action
    logical(c_bool) :: terminated
    integer :: i
    !
    this%F = action(1)
    this%step = this%step+1
    do i = 1, this%nsteps
      this%u = rk46_nl(this%t, this%dt, this%F, this%u) ! time discretization function defined below
      this%t = this%t + this%dt
      if (this%is_over()) then
        terminated = .true.
        return
      end if
    end do
    terminated = .false.
  end function advance


  function getState(this) result(state) ! get the full state (with additional values)
    class(cartPole), intent(in) :: this
    real, dimension(STATE_SIZE) :: state
    !
    state(1:4) = this%u
    state(5)   = cos(this%u(3))
    state(6)   = sin(this%u(3))
  end function getState


  function getReward(this) result(reward) ! assign a reward associated with the current state
    class(cartPole), intent(in) :: this
    real :: angle
    real :: reward
    reward = 0
    !
    if (SWINGUP) then
      angle = mod(this%u(3), 2*pi)
      if (angle<0) angle = angle + 2*pi
      if (abs(angle-pi)<pi/6) reward = 1
    else
      if (abs(this%u(3))<=pi/15 .and. abs(this%u(1))<=2.4) reward = 1
    end if
  end function getReward


  function rk46_nl(t0, dt, F, u0) result(u) ! time discretization
    real, intent(in) :: t0, dt, F
    real, dimension(4), intent(in) :: u0
    real, dimension(4) :: u
    integer :: i
    real :: t
    real, dimension(4) :: w=0
    integer, parameter :: s = 6
    real, dimension(6), parameter :: a = (/0.000000000000, -0.737101392796, -1.634740794341, &
        -0.744739003780, -1.469897351522, -2.813971388035/)
    real, dimension(6), parameter :: b = (/0.032918605146,  0.823256998200,  0.381530948900, &
         0.200092213184,  1.718581042715,  0.270000000000/)
    real, dimension(6), parameter :: c = (/0.000000000000,  0.032918605146,  0.249351723343, &
         0.466911705055,  0.582030414044,  0.847252983783/)
    !
    u = u0
    do i = 1, s
      t = t0 + dt*c(i)
      w = w*a(i) + Diff(u, t, F)*dt
      u = u + w*b(i)
    end do
  end function rk46_nl


  function Diff(u, t, F) result(res) ! part of time discretization
    real, dimension(4), intent(in) :: u
    real, intent(in) :: t, F
    real, dimension(4) :: res
    !
    real :: cosy, siny, w
    !
    real :: fac1, fac2, totMass, F1
    !
    res = 0
    cosy = cos(u(3))
    siny = sin(u(3))
    w = u(4)
    if (SWINGUP) then
      fac1 = 1./(mc + mp*siny*siny)
      fac2 = fac1/l
      res(2) = fac1*(F + mp*siny*(l*w*w + g*cosy))
      res(4) = fac2*(-F*cosy - mp*l*w*w*cosy*siny - (mc+mp)*g*siny)
    else
      totMass = mp + mc
      fac2 = l*(4./3. - mp*cosy*cosy/totMass)
      F1 = F + mp*l*w*w*siny
      res(4) = (g*siny - F1*cosy/totMass)/fac2
      res(2) = (F1 - mp*l*res(4)*cosy)/totMass
    end if
    res(1) = u(2)
    res(3) = u(4)
  end function


end module class_cartPole

!==============================================================================

module fortran_smarties
  ! Module interfacing with Smarties
  
  implicit none

  include 'mpif.h'

  private
  integer, parameter :: NUM_ACTIONS = 1
  integer, parameter :: STATE_SIZE  = 6

  ! Just to be clear on our intention for this procedure to be called from
  ! outside the module.
  public fortran_app_main


  interface
    ! Interfaces to the C++ side which send and receive data from Smarties.
    ! There should be no need to edit these unless Smarties is changed.
    subroutine rlcomm_update_state_action_dims(rlcomm, sdim, adim) &
        bind(c, name='rlcomm_update_state_action_dims')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: rlcomm
      integer(c_int),  intent(in), value :: sdim
      integer(c_int),  intent(in), value :: adim
    end subroutine rlcomm_update_state_action_dims

    subroutine rlcomm_set_action_scales(rlcomm, upper_action_bound, lower_action_bound, array_len, bounded) &
        bind(c, name='rlcomm_set_action_scales')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: rlcomm
      type(c_ptr),     intent(in), value :: upper_action_bound, lower_action_bound
      integer(c_int),  intent(in), value :: array_len
      logical(c_bool), intent(in), value :: bounded
    end subroutine rlcomm_set_action_scales

    subroutine rlcomm_set_state_scales(rlcomm, upper_state_bound, lower_state_bound, array_len) &
        bind(c, name='rlcomm_set_state_scales')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: rlcomm
      type(c_ptr),     intent(in), value :: upper_state_bound, lower_state_bound
      integer(c_int),  intent(in), value :: array_len
    end subroutine rlcomm_set_state_scales

    subroutine rlcomm_set_state_observable(rlcomm, b_observable, array_len) &
        bind(c, name='rlcomm_set_state_observable')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: rlcomm
      type(c_ptr),    intent(in), value :: b_observable
      integer(c_int), intent(in), value :: array_len
    end subroutine rlcomm_set_state_observable

    subroutine rlcomm_sendInitState(rlcomm, state, array_len) &
        bind(c, name='rlcomm_sendInitState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: rlcomm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: array_len
    end subroutine rlcomm_sendInitState

    subroutine rlcomm_sendTermState(rlcomm, state, array_len, reward) &
        bind(c, name='rlcomm_sendTermState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: rlcomm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: array_len
      real(c_double), intent(in), value :: reward
    end subroutine rlcomm_sendTermState

    subroutine rlcomm_sendState(rlcomm, state, array_len, reward) &
        bind(c, name='rlcomm_sendState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: rlcomm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: array_len
      real(c_double), intent(in), value :: reward
    end subroutine rlcomm_sendState

    subroutine rlcomm_recvAction(rlcomm, action, array_len) &
        bind(c, name='rlcomm_recvAction')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), intent(in), value :: rlcomm
      type(c_ptr), intent(in), value :: action
      integer(c_int), intent(in), value :: array_len
    end subroutine rlcomm_recvAction

  end interface


contains

  function fortran_app_main(rlcomm, f_mpicomm) result(result_value) &
    bind(c, name='fortran_app_main')
    ! This is the main function called from Smarties when the training starts.
  
    use, intrinsic :: iso_c_binding
    use class_cartPole
    implicit none
  
    type(c_ptr), intent(in), value :: rlcomm ! this is the pointer to the Smarties communicator
    integer,     intent(in), value :: f_mpicomm ! this is the MPI_COMM_WORLD handle initialized by Smarties
    !
    integer :: result_value
    !
    type(cartPole) :: env ! define one instance of the environment class
    !
    ! definition of the parameters used with Smarties
    logical(c_bool) :: bounded
    real(c_double),  dimension(NUM_ACTIONS), target :: upper_action_bound, lower_action_bound
    logical(c_bool), dimension(STATE_SIZE),  target :: b_observable
    real(c_double),  dimension(STATE_SIZE),  target :: upper_state_bound, lower_state_bound
    real(c_double),  dimension(NUM_ACTIONS), target :: action
    logical(c_bool) :: terminated
    real(c_double), dimension(STATE_SIZE), target :: state
    real(c_double) :: reward

    integer :: rank, numProcs, mpiIerr
  

    write(*,*) 'Fortran side begins'

    ! initialize MPI ranks and check that things are working
    call mpi_comm_rank(f_mpicomm, rank, mpiIerr)
    call mpi_comm_size(f_mpicomm, numProcs, mpiIerr)
    write(*,*) 'rank #', rank, ' of ', numProcs, ' is alive in Fortran'


    ! inform Smarties about the size of the state and the number of actions it can take
    call rlcomm_update_state_action_dims(rlcomm, STATE_SIZE, NUM_ACTIONS)
  
    ! OPTIONAL: aciton bounds
    bounded = .true.
    upper_action_bound = (/ 10/)
    lower_action_bound = (/-10/)
    call rlcomm_set_action_scales(rlcomm, c_loc(upper_action_bound), c_loc(lower_action_bound), NUM_ACTIONS, bounded)
  
    ! OPTIONAL: hide state variables.
    ! e.g. show cosine/sine but not angle
    b_observable = (/.true., .true., .true., .false., .true., .true./)
    call rlcomm_set_state_observable(rlcomm, c_loc(b_observable), STATE_SIZE)
  
    ! OPTIONAL: set space bounds
    upper_state_bound = (/ 1,  1,  1,  1,  1,  1/)
    lower_state_bound = (/-1, -1, -1, -1, -1, -1/)
    call rlcomm_set_state_scales(rlcomm, c_loc(upper_state_bound), c_loc(lower_state_bound), STATE_SIZE)
  
    ! train loop
    do while (.true.)
  
      ! reset environment
      call env%reset()
      
      ! send initial state to Smarties
      state = env%getState()
      call rlcomm_sendInitState(rlcomm, c_loc(state), STATE_SIZE)
  
      ! simulation loop
      do while (.true.)
  
        ! get the action
        call rlcomm_recvAction(rlcomm, c_loc(action), NUM_ACTIONS)
  
        ! advance the simulation
        terminated = env%advance(action)
  
        ! get the current state and reward
        state = env%getState()
        reward = env%getReward()
  
        if (terminated) then
          ! tell Smarties that this is a terminal state
          call rlcomm_sendTermState(rlcomm, c_loc(state), STATE_SIZE, reward)
          exit
        else
          call rlcomm_sendState(rlcomm, c_loc(state), STATE_SIZE, reward)
        end if
  
      end do ! simulation loop
  
    end do ! train loop
  
  
    result_value = 0
    write(*,*) 'Fortran side ends'

  end function fortran_app_main

end module fortran_smarties
