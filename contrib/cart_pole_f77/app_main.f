      function app_main(comm, mpicomm)
      implicit none
      integer AGENT_ID
      integer NUM_ACTIONS
      integer STATE_SIZE
      parameter (AGENT_ID = 0)
      parameter (NUM_ACTIONS = 1)
      parameter (STATE_SIZE = 6)
      double precision action(NUM_ACTIONS)
      double precision getReward
      double precision lower_action(NUM_ACTIONS)
      double precision lower_state(STATE_SIZE)
      double precision reward
      double precision state(STATE_SIZE)
      double precision upper_action(NUM_ACTIONS)
      double precision upper_state(STATE_SIZE)
      integer*8 comm
      integer app_main
      integer   mpicomm
      integer mpiIerr
      integer numProcs
      integer rank
      logical advance
      logical b_observable(STATE_SIZE)
      logical bounded
      logical terminated

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
      double precision F
      double precision t
      double precision u(4)
      integer step
      common /global/ u, step, F, t

      call random_number(u)
      u = (u-0.5)*2.*0.05       ! [-0.05, 0.05)
      F = 0
      t = 0
      step = 0
      end subroutine reset

      function is_over()
      implicit none
      double precision pi
      parameter(pi = 3.1415926535897931d0)
      double precision F
      double precision t
      double precision u(4)
      integer step
      logical is_over
      common /global/ u, step, F, t

      is_over = .false.
      if (step>=500 .or. abs(u(1))>2.4 .or. abs(u(3))>pi/15 )
     +     is_over = .true.
      end function is_over

      subroutine getState(state)
      implicit none
      integer STATE_SIZE
      parameter (STATE_SIZE = 6)
      double precision F
      double precision state(STATE_SIZE)
      double precision t
      double precision u(4)
      integer step
      common /global/ u, step, F, t

      state(1:4) = u
      state(5)   = cos(u(3))
      state(6)   = sin(u(3))
      end subroutine getState

      subroutine rk46_nl(t0, dt, F, u0)
      implicit none
      double precision t0
      double precision dt
      double precision F
      double precision u0(4)
      double precision u(4)
      integer i
      double precision t
      double precision w(4)
      integer s
      parameter(s = 6)
      double precision res(4)
      double precision a(6)
      double precision b(6)
      double precision c(6)
      a =
     + (/0.000000000000, -0.737101392796, -1.634740794341,
     +     -0.744739003780, -1.469897351522, -2.813971388035/)
      b =
     + (/0.032918605146,  0.823256998200,  0.381530948900,
     + 0.200092213184,  1.718581042715,  0.270000000000/)
      c =
     + (/0.000000000000,  0.032918605146,  0.249351723343,
     +     0.466911705055,  0.582030414044,  0.847252983783/)
      w = 0
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
      double precision mc
      double precision mp
      double precision l
      double precision g
      parameter (mp = 0.1)
      parameter (mc = 1.0)
      parameter (l  = 0.5)
      parameter (g  = 9.81)
      double precision u(4)
      double precision F
      double precision res(4)
      double precision cosy
      double precision siny
      double precision w
      double precision fac2
      double precision totMass
      double precision F1

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
      double precision getReward
      double precision pi
      parameter (pi = 3.1415926535897931d0)
      integer step
      double precision u(4)
      double precision F
      double precision t
      common /global/ u, step, F, t
      getReward = 0
      if (abs(u(3))<=pi/15 .and. abs(u(1))<=2.4) getReward = 1
      end function getReward

      function advance(action)
      implicit none
      integer NUM_ACTIONS
      parameter(NUM_ACTIONS = 1)
      double precision action(NUM_ACTIONS)
      logical advance
      logical is_over
      integer i
      double precision dt
      parameter(dt = 4e-4)
      integer nsteps
      parameter(nsteps = 50)
      integer step
      double precision u(4)
      double precision F
      double precision t
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
