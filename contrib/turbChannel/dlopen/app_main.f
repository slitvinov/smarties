      function app_main(comm, mpicomm, first)
      implicit none
      intrinsic backtrace
      intrinsic abort
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

      character*128 cwd

      smarties_comm = comm

      write(6,*) 'app_main.f: Fortran side begins'
      if (.not. good()) then
         write(6,*) 'app_main.f: failed configuration test'
         call flush(6)
         call backtrace()
         call abort()
      end if
      write(6,*) 'app_main.f: passes configuration test'
      if (first .eq. 1) then
         call smarties_setnumagents(comm,NUM_AGENTS)
         ! initialize all agents
         do AGENT_ID = 0,NUM_AGENTS-1
            call smarties_setstateactiondims(comm,
     +           STATE_SIZE, NUM_ACTIONS, AGENT_ID)
            bounded = .true.
            call smarties_setactionscales(comm,
     +           upper_action, lower_action,
     +           bounded, NUM_ACTIONS, AGENT_ID)
            call smarties_setStateObservable(comm,
     +           b_observable, STATE_SIZE, AGENT_ID)
            call smarties_setStateScales(comm, upper_state, lower_state,
     +           STATE_SIZE, AGENT_ID)
         end do 

         ! copies necessary file to working directory
         open(unit=1,file='SESSION.NAME')
         call getcwd(cwd)
         write(1,'(A)') 'turbChannel'
         write(1,'(A)') trim(cwd)//trim('/')
         close(1)
         call system('cp ../turbChannel.re2 .')
         call system('cp ../turbChannel.ma2 .')
         call system('cp ../turbChannel.par .')
      end if

      call nek_init(mpicomm)
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
