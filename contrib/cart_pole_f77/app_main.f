      function app_main(comm, mpicomm)
      implicit none
      integer, parameter :: AGENT_ID = 0
      integer, parameter :: NUM_ACTIONS = 1
      integer, parameter :: STATE_SIZE = 6
      integer*8,    intent(in), value :: comm
      integer, intent(in), value :: mpicomm

      double precision, dimension(NUM_ACTIONS) :: action
      double precision, dimension(NUM_ACTIONS) :: lower_action
      double precision, dimension(NUM_ACTIONS) :: upper_action
      double precision, dimension(STATE_SIZE) :: lower_state
      double precision, dimension(STATE_SIZE) :: state
      double precision, dimension(STATE_SIZE) :: upper_state
      double precision :: getReward
      double precision :: reward
      integer :: app_main
      integer :: mpiIerr
      integer :: numProcs
      integer :: rank
      logical :: advance
      logical :: bounded
      logical, dimension(STATE_SIZE) :: b_observable
      logical :: terminated

      write(6,*) 'Fortran side begins'
      call mpi_comm_rank(mpicomm, rank, mpiIerr)
      call mpi_comm_size(mpicomm, numProcs, mpiIerr)
      write(6,*) 'rank #', rank, ' of ', numProcs,
     +      ' is alive in Fortran'
      call smarties_setstateactiondims(comm, STATE_SIZE, NUM_ACTIONS,
     + AGENT_ID)
      bounded = .true.
      upper_action = (/ 10/)
      lower_action = (/-10/)
      call smarties_setactionscales(comm,
     +      upper_action, lower_action,
     +      bounded, NUM_ACTIONS, AGENT_ID)
      b_observable = (/.true., .true., .true., .false., .true., .true./)
      call smarties_setStateObservable(comm, b_observable, STATE_SIZE,
     +                                 AGENT_ID)
      upper_state = (/ 1,  1,  1,  1,  1,  1/)
      lower_state = (/-1, -1, -1, -1, -1, -1/)
      call smarties_setStateScales(comm, upper_state, lower_state,
     +                             STATE_SIZE, AGENT_ID)
      do while (.true.)
      call reset()
      call getState(state)
      call smarties_sendInitState(comm, state, STATE_SIZE, AGENT_ID)
      do while (.true.)
         call smarties_recvAction(comm, action, NUM_ACTIONS, AGENT_ID)
         terminated = advance(action)
         call getState(state)
         reward = getReward()
         if (terminated) then
            call smarties_sendTermState(comm, state, STATE_SIZE,
     +                                 reward, AGENT_ID)
            exit
         else
            call smarties_sendState(comm, state, STATE_SIZE,
     +       reward, AGENT_ID)
         end if
       end do
      end do
      app_main = 0
      write(6,*) 'Fortran side ends'
      end function app_main

      subroutine reset()
      implicit none
      integer :: step
      double precision, dimension(4) :: u
      double precision :: F
      double precision :: t
      common /global/ u, step, F, t
      call random_number(u)
      u = (u-0.5)*2.*0.05       ! [-0.05, 0.05)
      F = 0
      t = 0
      step = 0
      end subroutine reset

      function is_over()
      implicit none
      logical :: is_over
      double precision, parameter :: pi = 3.1415926535897931d0
      integer :: step
      double precision, dimension(4) :: u
      double precision :: F
      double precision :: t
      common /global/ u, step, F, t

      is_over = .false.
      if (step>=500 .or. abs(u(1))>2.4 .or. abs(u(3))>pi/15 )
     +     is_over = .true.
      end function is_over

      subroutine getState(state)
      implicit none
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
      implicit none
      double precision, intent(in) :: t0
      double precision, intent(in) :: dt
      double precision, intent(in) :: F
      double precision, dimension(4), intent(inout) :: u0
      double precision, dimension(4) :: u
      integer :: i
      double precision :: t
      double precision, dimension(4) :: w=0
      integer, parameter :: s = 6
      double precision, dimension(6), parameter :: a =
     + (/0.000000000000, -0.737101392796, -1.634740794341,
     + -0.744739003780, -1.469897351522, -2.813971388035/)
      double precision, dimension(6), parameter :: b =
     + (/0.032918605146,  0.823256998200,  0.381530948900,
     + 0.200092213184,  1.718581042715,  0.270000000000/)
      double precision, dimension(6), parameter :: c =
     + (/0.000000000000,  0.032918605146,  0.249351723343,
     + 0.466911705055,  0.582030414044,  0.847252983783/)
      double precision, dimension(4) :: res

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
      implicit none
      double precision, dimension(4), intent(in) :: u
      double precision, intent(in) :: F
      double precision, dimension(4) :: res
      double precision :: cosy
      double precision :: siny
      double precision :: w
      double precision :: fac2
      double precision :: totMass
      double precision :: F1
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

      function getReward()
      implicit none
      double precision :: getReward
      double precision, parameter :: pi = 3.1415926535897931d0
      integer :: step
      double precision, dimension(4) :: u
      double precision :: F
      double precision :: t
      common /global/ u, step, F, t
      getReward = 0
      if (abs(u(3))<=pi/15 .and. abs(u(1))<=2.4) getReward = 1
      end function getReward

      function advance(action)
      implicit none
      integer, parameter :: NUM_ACTIONS = 1
      double precision, dimension(NUM_ACTIONS) :: action
      logical :: advance
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
         advance = .true.
         return
      end if
      end do
      advance = .false.
      end function advance
