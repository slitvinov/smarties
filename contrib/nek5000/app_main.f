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

         call opcount(1)

         call initdim         ! Initialize / set default values.
         call initdat
         call files

         call readat          ! Read .rea +map file
         call setvar          ! Initialize most variables

         instep=1             ! Check for zero steps
         if (nsteps.eq.0 .and. fintim.eq.0.) instep=0

         igeom = 2
         call setup_topo      ! Setup domain topology  

         call genwz           ! Compute GLL points, weights, etc.

         if(nio.eq.0) write(6,*) 'call usrdat'
         call usrdat
         if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat' 

         call gengeom(igeom)  ! Generate geometry, after usrdat 

         if (ifmvbd) call setup_mesh_dssum ! Set mesh dssum (needs geom)

         if(nio.eq.0) write(6,*) 'call usrdat2'
         call usrdat2
         if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat2' 

         call fix_geom
         call geom_reset(1)    ! recompute Jacobians, etc.

         call vrdsmsh          ! verify mesh topology
         call mesh_metrics     ! print some metrics

         call setlog(.true.)   ! Initalize logical flags

         if (ifneknekc) call neknek_setup

         call bcmask  ! Set BC masks for Dirichlet boundaries.

         if (fintim.ne.0.0 .or. nsteps.ne.0) 
     $      call geneig(igeom) ! eigvals for tolerances

         call dg_setup ! Setup DG, if dg flag is set.

         if (ifflow.and.iftran) then ! Init pressure solver 
            if (fintim.ne.0 .or. nsteps.ne.0) call prinit
         endif

         if(ifcvode) call cv_setsize

         if(nio.eq.0) write(6,*) 'call usrdat3'
         call usrdat3
         if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat3'

#ifdef CMTNEK
         call nek_cmt_init
#endif   

         call setics
         call setprop

         if (instep.ne.0) then
            if (ifneknekc) call neknek_exchange
            if (ifneknekc) call chk_outflow

            if (nio.eq.0) write(6,*) 'call userchk'
            call userchk
            if(nio.eq.0) write(6,'(A,/)') ' done :: userchk' 
         endif

         call setprop      ! call again because input has changed in userchk

         if (ifcvode .and. nsteps.gt.0) call cv_init

         call comment
         call sstest (isss) 

         call dofcnt

         jp = 0  ! Set perturbation field count to 0 for baseline flow
         p0thn = p0th

         call in_situ_init()

         call time00       !     Initalize timers to ZERO
         call opcount(2)

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
