      function app_main(comm, mpicomm, first)
      include 'mpif.h'
      include 'SMARTIES'
      include 'SIZE'
      include 'TOTAL'

      integer app_main

      integer*8 comm
      integer first
      integer mpicomm
      integer mpiIerr
      integer numProcs
      integer rank
      logical bounded
      
      character*128 cwd

      smarties_comm = comm

      write(6,*) 'Fortran side begins'
      write(6,*) 'first: ', first
      if (first .eq. 1) then
         call smarties_setstateactiondims(comm, STATE_SIZE, NUM_ACTIONS,
     +        AGENT_ID)
         bounded = .true.
         call smarties_setactionscales(comm,
     +        upper_action, lower_action,
     +        bounded, NUM_ACTIONS, AGENT_ID)
         call smarties_setStateObservable(comm, b_observable, STATE_SIZE,
     +        AGENT_ID)
         call smarties_setStateScales(comm, upper_state, lower_state,
     +        STATE_SIZE, AGENT_ID)
      end if

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

      do while (.true.)
         ! initialize
         smarties_step = smarties_step + 1
         call init_common()
         call nek_init(mpicomm)
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


      subroutine init_common()
      include 'SIZE'
      include 'TOTAL'
C SOLN 
      bq=0
      vxlag=0 
      vylag=0 
      vzlag=0 
      tlag=0 
      pr=0 
      prlag=0
      dp0thdt=0
      gamma0=0
      p0thlag=0
      v1mask=0
      v2mask=0
      v3mask=0
      pmask=0
      tmask=0
      omask=0
      vmult=0
      tmult=0
      b1mask=0
      b2mask=0
      b3mask=0
      bpmask=0
      vxp=0 
      vyp=0 
      vzp=0 
      prp=0
      tp=0
      bqp=0
      bfxp=0
      bfyp=0
      bfzp=0
      vxlagp=0
      vylagp=0
      vzlagp=0
      prlagp=0
      tlagp=0
      exx1p=0
      exy1p=0
      exz1p=0
      exx2p=0
      exy2p=0
      exz2p=0
      vgradt1p=0
      vgradt2p=0
      jp=0
C TSTEP
      time=0
      timef=0
      fintim=0
      timeio=0
      timeioe=0
      dt=0
      dtlag=0
      dtinit=0
      dtinvm=0
      courno=0
      ctarg=0
      ab=0
      bd=0
      abmsh=0
      avdiff=0
      avtran=0
      volfld=0
      tolrel=0
      tolabs=0
      tolhdf=0
      tolpdf=0
      tolev=0
      tolnl=0
      prelax=0
      tolps=0
      tolhs=0
      tolhr=0
      tolhv=0
      tolht=0
      tolhe=0
      iesolv=0
      ifalgn=0 
      ifrsxy=0
      volel=0    
      return
      end
