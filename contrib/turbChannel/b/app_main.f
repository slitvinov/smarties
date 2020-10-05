      function app_main(comm, mpicomm)
      include 'mpif.h'
      include 'SMARTIES'

      integer app_main

      integer*8 comm
      integer mpicomm
      integer mpiIerr
      integer numProcs
      integer rank
      logical bounded
      integer status

      smarties_comm = comm
      status = system("/u/setup.sh")
      write(6,*) 'Fortran side begins'
      call smarties_setstateactiondims(comm, STATE_SIZE, NUM_ACTIONS,
     +     AGENT_ID)
      bounded = .true.
      call smarties_setactionscales(comm,
     +      upper_action, lower_action,
     +      bounded, NUM_ACTIONS, AGENT_ID)
      call smarties_setStateObservable(comm, b_observable, STATE_SIZE,
     +     AGENT_ID)
      call smarties_setStateScales(comm, upper_state, lower_state,
     +     STATE_SIZE, AGENT_ID)
      do while (.true.)
         call nek_init(mpicomm)
         call nek_solve()
         call smarties_sendTermState(comm, state, STATE_SIZE,
     +        reward, AGENT_ID)
         call nek_end()
      end do

      app_main = 0
      write(6,*) 'Fortran side ends'
      end