module smarties

  implicit none

  interface
    subroutine smarties_sendInitState( &
        ptr2comm, state, state_dim, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: state_dim
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_sendInitState
  end interface

  interface
    subroutine smarties_sendState( &
        ptr2comm, state, state_dim, reward, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: state_dim
      real(c_double), intent(in), value :: reward
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_sendState
  end interface

  interface
    subroutine smarties_sendTermState( &
        ptr2comm, state, state_dim, reward, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: state_dim
      real(c_double), intent(in), value :: reward
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_sendTermState
  end interface

  interface
    subroutine smarties_sendLastState( &
        ptr2comm, state, state_dim, reward, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: state_dim
      real(c_double), intent(in), value :: reward
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_sendLastState
  end interface

  interface
    subroutine smarties_recvAction( &
        ptr2comm, action, action_dim, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: action
      integer(c_int), intent(in), value :: action_dim
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_recvAction
  end interface

  interface
    subroutine smarties_setNumAgents( &
        ptr2comm, num_agents)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), value :: num_agents
    end subroutine smarties_setNumAgents
  end interface

  interface
    subroutine smarties_setStateActionDims( &
        ptr2comm, state_dim, action_dim, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), value :: state_dim
      integer(c_int), intent(in), value :: action_dim
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_setStateActionDims
  end interface


  interface
    subroutine smarties_setActionScales( &
        ptr2comm, upper_act_bound, lower_act_bound, &
        bounded, action_dim, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: ptr2comm
      type(c_ptr),     intent(in), value :: upper_act_bound
      type(c_ptr),     intent(in), value :: lower_act_bound
      logical(c_bool), intent(in), value :: bounded
      integer(c_int),  intent(in), value :: action_dim
      integer(c_int),  intent(in), optional :: agentID
    end subroutine smarties_setActionScales
  end interface

  interface
    subroutine smarties_setActionScalesBounds( &
        ptr2comm, upper_act_bound, lower_act_bound, &
        bounded, action_dim, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: ptr2comm
      type(c_ptr),     intent(in), value :: upper_act_bound
      type(c_ptr),     intent(in), value :: lower_act_bound
      type(c_ptr),     intent(in), value :: bounded
      integer(c_int),  intent(in), value :: action_dim
      integer(c_int),  intent(in), optional :: agentID
    end subroutine smarties_setActionScalesBounds
  end interface


  interface
    subroutine smarties_setActionOptions( &
        ptr2comm, num_options, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: ptr2comm
      integer(c_int),  intent(in), value :: num_options
      integer(c_int),  intent(in), optional :: agentID
    end subroutine smarties_setActionOptions
  end interface

  interface
    subroutine smarties_setActionOptionsPerDim( &
        ptr2comm, num_options, action_dim, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: ptr2comm
      type(c_ptr),     intent(in), value :: num_options
      integer(c_int),  intent(in), value :: action_dim
      integer(c_int),  intent(in), optional :: agentID
    end subroutine smarties_setActionOptionsPerDim
  end interface

  interface
    subroutine smarties_setStateObservable( &
        ptr2comm, b_observable, state_dim, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: b_observable
      integer(c_int), intent(in), value :: state_dim
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_setStateObservable
  end interface

  interface
    subroutine smarties_setStateScales( &
        ptr2comm, upper_state_bound, lower_state_bound, state_dim, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: upper_state_bound
      type(c_ptr),    intent(in), value :: lower_state_bound
      integer(c_int), intent(in), value :: state_dim
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_setStateScales
  end interface

  interface
    subroutine smarties_setIsPartiallyObservable(ptr2comm, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_setIsPartiallyObservable
  end interface

  interface
    subroutine smarties_finalizeProblemDescription(ptr2comm)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
    end subroutine smarties_finalizeProblemDescription
  end interface

  interface
    subroutine smarties_envHasDistributedAgents(ptr2comm)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
    end subroutine smarties_envHasDistributedAgents
  end interface

  interface
    subroutine smarties_agentsDefineDifferentMDP(ptr2comm)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
    end subroutine smarties_agentsDefineDifferentMDP
  end interface

  interface
    subroutine smarties_disableDataTrackingForAgents( &
        ptr2comm, agentStart, agentEnd)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), value :: agentStart
      integer(c_int), intent(in), value :: agentEnd
    end subroutine smarties_disableDataTrackingForAgents
  end interface

  interface
    subroutine smarties_setPreprocessingConv2d( &
        ptr2comm, input_width, input_height, input_features, &
        kernels_num, filters_size, stride, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), value :: input_width
      integer(c_int), intent(in), value :: input_height
      integer(c_int), intent(in), value :: input_features
      integer(c_int), intent(in), value :: kernels_num
      integer(c_int), intent(in), value :: filters_size
      integer(c_int), intent(in), value :: stride
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_setPreprocessingConv2d
  end interface

  interface
    subroutine smarties_setNumAppendedPastObservations( &
        ptr2comm, n_appended, agentID)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), value :: n_appended
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_setNumAppendedPastObservations
  end interface

  interface
    subroutine smarties_getUniformRandom( &
        ptr2comm, range_begin, range_end, sampled)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in),  value    :: ptr2comm
      real(c_double), intent(in),  optional :: range_begin
      real(c_double), intent(in),  optional :: range_end
      real(c_double), intent(out), optional :: sampled(*)
    end subroutine smarties_getUniformRandom
  end interface

  interface
    subroutine smarties_getNormalRandom( &
        ptr2comm, mean, stdev, sampled)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in),  value    :: ptr2comm
      real(c_double), intent(in),  optional :: mean
      real(c_double), intent(in),  optional :: stdev
      real(c_double), intent(out), optional :: sampled(*)
    end subroutine smarties_getNormalRandom
  end interface

end module smarties
  
module app_main_module
    implicit none
    include 'mpif.h'
    public app_main
  contains
    function app_main(smarties_comm, f_mpicomm) result(result_value) &
         bind(c, name='app_main')      
      use, intrinsic :: iso_c_binding
      use smarties
      implicit none

      type(c_ptr),    intent(in), value :: smarties_comm
      integer(c_int), intent(in), value :: f_mpicomm
      integer(c_int) :: result_value
      logical(c_bool) :: bounded
      integer, parameter :: NUM_ACTIONS = 1
      integer, parameter :: STATE_SIZE = 6
      integer, parameter :: AGENT_ID = 0
      real(c_double),  dimension(NUM_ACTIONS), target :: upper_action_bound, lower_action_bound
      logical(c_bool), dimension(STATE_SIZE),  target :: b_observable
      real(c_double),  dimension(STATE_SIZE),  target :: upper_state_bound, lower_state_bound
      real(c_double),  dimension(NUM_ACTIONS), target :: action
      logical :: terminated
      real(c_double), dimension(STATE_SIZE), target :: state
      real(c_double) :: reward
      double precision :: getReward
      logical :: advance

      integer :: rank, numProcs, mpiIerr
      write(6,*) 'Fortran side begins'
      call mpi_comm_rank(f_mpicomm, rank, mpiIerr)
      call mpi_comm_size(f_mpicomm, numProcs, mpiIerr)
      write(6,*) 'rank #', rank, ' of ', numProcs, ' is alive in Fortran'
      ! inform Smarties about the size of the state and the number of actions it can take
      call smarties2_setstateactiondims(smarties_comm, STATE_SIZE, NUM_ACTIONS, AGENT_ID)

      ! OPTIONAL: aciton bounds
      bounded = .true.
      upper_action_bound = (/ 10/)
      lower_action_bound = (/-10/)
      call smarties_setactionscales(smarties_comm, &
           c_loc(upper_action_bound), c_loc(lower_action_bound), &
           bounded, NUM_ACTIONS)

      ! OPTIONAL: hide state variables.
      ! e.g. show cosine/sine but not angle
      b_observable = (/.true., .true., .true., .false., .true., .true./)
      call smarties_setStateObservable(smarties_comm, c_loc(b_observable), STATE_SIZE)

      ! OPTIONAL: set space bounds
      upper_state_bound = (/ 1,  1,  1,  1,  1,  1/)
      lower_state_bound = (/-1, -1, -1, -1, -1, -1/)
      call smarties_setStateScales(smarties_comm, c_loc(upper_state_bound), c_loc(lower_state_bound), STATE_SIZE)

      do while (.true.)
         call reset()
         call getState(state)
         call smarties_sendInitState(smarties_comm, c_loc(state), STATE_SIZE)
         do while (.true.)
            call smarties_recvAction(smarties_comm, c_loc(action), NUM_ACTIONS)
            terminated = advance(action)
            call getState(state)
            reward = getReward()
            if (terminated) then
               call smarties_sendTermState(smarties_comm, c_loc(state), STATE_SIZE, reward)
               exit
            else
               call smarties_sendState(smarties_comm, c_loc(state), STATE_SIZE, reward)
            end if
         end do
      end do
      result_value = 0
      write(6,*) 'Fortran side ends'
    end function app_main
  end module app_main_module

  subroutine reset()
    integer :: step
    double precision, dimension(4) :: u
    double precision :: F
    double precision :: t
    common /global/ u, step, F, t
    call random_number(u)
    u = (u-0.5)*2.*0.05 ! [-0.05, 0.05)
    F = 0
    t = 0
    step = 0
  end subroutine reset

  function is_over() result(answer)
    logical :: answer
    double precision, parameter :: pi = 3.1415926535897931d0
    integer :: step
    double precision, dimension(4) :: u
    double precision :: F
    double precision :: t
    common /global/ u, step, F, t
    answer=.false.
    if (step>=500 .or. abs(u(1))>2.4 .or. abs(u(3))>pi/15 ) answer = .true.
  end function is_over


    subroutine getState(state)
      integer,          parameter :: STATE_SIZE = 6
      double precision, dimension(STATE_SIZE) :: state
      integer :: step
      double precision, dimension(4) :: u
      double precision :: F
      double precision :: t
      common /global/ u, step, F, t
      state(1:4) = u
      state(5)   = cos(u(3))
      state(6)   = sin(u(3))
    end subroutine getState

    subroutine rk46_nl(t0, dt, F, u0)
      double precision, intent(in) :: t0, dt, F
      double precision, dimension(4), intent(inout) :: u0
      double precision, dimension(4) :: u
      integer :: i
      double precision :: t
      double precision, dimension(4) :: w=0
      integer, parameter :: s = 6
      double precision, dimension(6), parameter :: a = (/0.000000000000, -0.737101392796, -1.634740794341, &
           -0.744739003780, -1.469897351522, -2.813971388035/)
      double precision, dimension(6), parameter :: b = (/0.032918605146,  0.823256998200,  0.381530948900, &
           0.200092213184,  1.718581042715,  0.270000000000/)
      double precision, dimension(6), parameter :: c = (/0.000000000000,  0.032918605146,  0.249351723343, &
           0.466911705055,  0.582030414044,  0.847252983783/)
      double precision, dimension(4) :: res
      !
      u = u0
      do i = 1, s
         t = t0 + dt*c(i)
         call Diff(u, F, res)
         w = w*a(i) + res*dt
         u = u + w*b(i)
      end do
      u0 = u
    end subroutine rk46_nl

    subroutine Diff(u, F, res)
      double precision, dimension(4), intent(in) :: u
      double precision, intent(in) :: F
      double precision, dimension(4) :: res
      double precision :: cosy, siny, w
      double precision :: fac1, fac2, totMass, F1
      double precision, parameter :: mp = 0.1
      double precision, parameter :: mc = 1.
      double precision, parameter :: l  = 0.5
      double precision, parameter :: g  = 9.81
      res = 0
      cosy = cos(u(3))
      siny = sin(u(3))
      w = u(4)
      totMass = mp + mc
      fac2 = l*(4./3. - mp*cosy*cosy/totMass)
      F1 = F + mp*l*w*w*siny
      res(4) = (g*siny - F1*cosy/totMass)/fac2
      res(2) = (F1 - mp*l*res(4)*cosy)/totMass
      res(1) = u(2)
      res(3) = u(4)
    end subroutine Diff

    function getReward() result(reward)
      double precision :: angle
      double precision :: reward
      double precision, parameter :: pi = 3.1415926535897931d0
      integer :: step
      double precision, dimension(4) :: u
      double precision :: F
      double precision :: t
      common /global/ u, step, F, t
      reward = 0
      if (abs(u(3))<=pi/15 .and. abs(u(1))<=2.4) reward = 1
    end function getReward

    function advance(action) result(terminated)
      use, intrinsic :: iso_c_binding
      integer, parameter :: NUM_ACTIONS = 1
      real(c_double), dimension(NUM_ACTIONS) :: action
      logical :: terminated
      logical :: is_over
      integer :: i
      double precision :: dt = 4e-4
      integer :: nsteps = 50
      integer :: step
      double precision, dimension(4) :: u
      double precision :: F
      double precision :: t
      common /global/ u, step, F, t
      F = action(1)
      step = step+1
      do i = 1, nsteps
         call rk46_nl(t, dt, F, u)
         t = t + dt
         if (is_over()) then
            terminated = .true.
            return
         end if
      end do
      terminated = .false.
    end function advance
