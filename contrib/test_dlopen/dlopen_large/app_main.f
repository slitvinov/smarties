      function app_main(comm, mpicomm, first)
      implicit none
      intrinsic flush
      logical good

      include 'mpif.h'
      include 'SMARTIES'
      include 'SIZE'
      include 'TOTAL'

      integer app_main

      integer*8 comm
      integer counter
      integer first
      integer mpicomm
      integer mpiIerr
      integer numProcs
      integer rank
      logical bounded

      integer NUM_RE
      integer NUM_GRID
      parameter (NUM_RE=3)
      parameter (NUM_GRID=2)
      integer Re, Grid, time_seed
      real*4  r1, r2
      character*128 case_name

      character*128 cwd

      smarties_comm = comm

      write(6,*) 'app_main.f: Fortran side begins'
      if (.not. good()) then
         write(6,*) 'app_main.f: failed configuration test'
         call fail()
      end if
      write(6,*) 'app_main.f: passes configuration test'
      call mpi_comm_rank(mpicomm, rank, mpiIerr)
      if (mpiIerr .ne. mpi_success) then
         write(6, *) 'app_main.f: mpi_comm_rank failed'
         call fail()
      endif
      call mpi_comm_size(mpicomm, numProcs, mpiIerr)
      if (mpiIerr .ne. mpi_success) then
         write(6, *) 'app_main.f: mpi_comm_size failed'
         call fail()
      endif
      write(*,*) 'rank #',rank,' of ',numProcs,' is alive in Fortran'
      if (first .eq. 1) then
         call smarties_setNumAgents(comm,NUM_AGENTS)
         ! initialize all agents
         do AGENT_ID = 0,NUM_AGENTS-1
            call smarties_setStateActionDims(comm,
     +           STATE_SIZE, NUM_ACTIONS, AGENT_ID)
            bounded = .true.
            call smarties_setActionScales(comm,
     +           upper_action, lower_action,
     +           bounded, NUM_ACTIONS, AGENT_ID)
            call smarties_setStateObservable(comm,
     +           b_observable, STATE_SIZE, AGENT_ID)
            call smarties_setStateScales(comm, upper_state, lower_state,
     +           STATE_SIZE, AGENT_ID)
         end do
      end if
      if (rank.eq.0) then
         case_name = 'turbChannel'

         open(unit=1,file='SESSION.NAME')
         call getcwd(cwd)
         write(1,'(A)') case_name
         write(1,'(A)') trim(cwd)//trim('/')
         close(1)
         call system(trim('cp ../')//trim(case_name)//
     &   trim('.re2 ./turbChannel.re2'))
         call system(trim('cp ../')//trim(case_name)//
     &   trim('.ma2 ./turbChannel.ma2'))
         call system(trim('cp ../')//trim(case_name)//
     &   trim('.par ./turbChannel.par'))
         call system(trim('cp ../')//trim(case_name)//
     &   trim('.init ./turbChannel.init'))
      end if
      call nek_init(mpicomm)
      call mpi_barrier(mpicomm, mpiIerr)
      if (mpiIerr .ne. mpi_success) then
         write(6, *) 'app_main.f: mpi_barrier failed'
         call fail()
      endif
      write(6, *), 'app_main.f: passes mpi_barrier'
      call flush(6)
      call nek_solve()
      write(6,*) 'app_main.f: Fortran side ends'
      app_main = 0
      end

      function counter()
      implicit none
      integer cnt
      integer counter
      save cnt
      data cnt /42/
      cnt = cnt + 1
      counter = cnt
      end

      function good()
      implicit none
      intrinsic isnan
      integer counter
      logical good
      real x

      good = .true.
      if (counter() .ne. 42 + 1) then
         write(6, *) 'app_main.f: app_main was not reset'
         good = .false.
      end if
      x = -1.0
      if (.not. isnan(sqrt(x))) then
         write(6, *) 'app_main.f: isnan is wrong', sqrt(x)
         good = .false.
      end if
      return
      end

      subroutine fail()
      implicit none
      intrinsic abort
      intrinsic backtrace
      intrinsic flush

      call flush(6)
      call backtrace()
      call abort()
      end
