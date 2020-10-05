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
      call mpi_comm_rank(mpicomm, rank, mpiIerr)
      call mpi_comm_size(mpicomm, numProcs, mpiIerr)
      write(6,*) 'rank #', rank, ' of ', numProcs,
     +      ' is alive in Fortran'
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

      open(unit=1,file='SESSION.NAME')
      call getcwd(cwd)
      write(1,'(A)') 'turbChannel' 
      write(1,'(A)') trim(cwd) 
      write(*,*) cwd
      close(1)
      do while (.true.)
         time = 0
         call nek_init(mpicomm)
         call nek_solve()
         call smarties_sendTermState(comm, state, STATE_SIZE,
     +                                 reward, AGENT_ID)
         call nek_end()
         write(*,*) 'done with nek'
      end do

      app_main = 0
      write(6,*) 'Fortran side ends'
      end
