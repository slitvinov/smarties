      function app_main(comm, mpicomm)
      include 'mpif.h'
      include 'SMARTIES'
      include 'SIZE'
      include 'TOTAL'

      integer app_main

      integer*8 comm
      integer mpicomm
      integer mpiIerr
      integer numProcs
      integer rank
      logical bounded
      
      character*128 cwd

      smarties_comm = comm

      write(6,*) 'Fortran side begins'
      call smarties_setstateactiondims(comm, STATE_SIZE, NUM_ACTIONS,
     + AGENT_ID)
      bounded = .true.
      call smarties_setactionscales(comm,
     +      upper_action, lower_action,
     +      bounded, NUM_ACTIONS, AGENT_ID)
      call smarties_setStateObservable(comm, b_observable, STATE_SIZE,
     +                                 AGENT_ID)
      call smarties_setStateScales(comm, upper_state, lower_state,
     +                             STATE_SIZE, AGENT_ID)

      ! copies necessary file to working directory
      open(unit=1,file='SESSION.NAME')
      call getcwd(cwd)
      write(1,'(A)') 'turbChannel' 
      write(1,'(A)') trim(cwd)//trim('/') 
      close(1)
      call system('cp ../turbChannel.re2 .')
      call system('cp ../turbChannel.ma2 .')
      call system('cp ../turbChannel.par .')
      ! training loop
      smarties_step = 0
      call nek_init(mpicomm)
      do while (.true.)
         smarties_step = smarties_step + 1
         time = 0
         etimes = dnekclock()
         istep  = 0
         call setvar       ! initialize most variables
         call setics       ! sets initial conditions
         call setprop      ! sets variable properties of arrays 
         call userchk      ! calls user defined userchk
         call setprop      ! call again because input has changed in userchk
         jp = 0  
         p0thn = p0th
         call time00       ! initalize timers to ZERO

         ! solve loop (see turbChannel.usr for details)
         call nek_solve()
         ! send final state after finishing
         call smarties_sendTermState(comm, state, STATE_SIZE,
     +                                 reward, AGENT_ID)
      end do
      ! finalize only after full loop
      call nek_end()

      app_main = 0
      write(6,*) 'Fortran side ends'
      end
