c- constants -----------------------------------------------------------

#define tSTATSTART uparam(1) /* start time for averaging */
#define tSTATFREQ  uparam(2) /* output frequency for statistics */

c data extraction along wall normal direction
#define INTP_NMAX 200 /* number of sample points */
#define XCINT 1.0     /* x coordinate of 1D line*/
#define ZCINT 1.0     /* z coordinate of 1D line */

c mesh dimensions
#define BETAM 2.4     /* wall normal stretching parameter */
#define PI (4.*atan(1.))
#define XLEN (2.*PI)
#define ZLEN PI
#define NUMBER_ELEMENTS_X 8
#define NUMBER_ELEMENTS_Y 6
#define NUMBER_ELEMENTS_Z 4
#define DPDX 0.00239166557 

c-----------------------------------------------------------------------
      subroutine uservp (ix,iy,iz,ieg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'

      integer e

      utrans = 1.
      udiff  = param(2)

      if (ifield .eq. 2) then
         e = gllel(ieg)
         udiff = param(8)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine userf  (ix,iy,iz,ieg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'

      ffx = DPDX 
      ffy = 0.0
      ffz = 0.0

      return
      end
c-----------------------------------------------------------------------
      subroutine userq  (ix,iy,iz,ieg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'

      qvol =  0.0

      return
      end
c-----------------------------------------------------------------------
      subroutine userchk
      include 'SIZE'
      include 'TOTAL'
      include 'SMARTIES'

      real x0(3)
      data x0 /0.0, 0.0, 0.0/ 
      save x0

      integer icalld
      save    icalld
      data    icalld /0/

      real atime,timel
      save atime,timel

      integer ntdump
      save    ntdump

      real    rwk(INTP_NMAX,ldim+1) ! r, s, t, dist2
      integer iwk(INTP_NMAX,3)      ! code, proc, el 
      save    rwk, iwk

      integer nint, intp_h
      save    nint, intp_h

      logical iffpts
      save iffpts

      real xint(INTP_NMAX),yint(INTP_NMAX),zint(INTP_NMAX)
      save xint, yint, zint
      save igs_x, igs_z

      parameter(nstat=9)
      real ravg(lx1*ly1*lz1*lelt,nstat)
      real stat(lx1*ly1*lz1*lelt,nstat)
      real stat_y(INTP_NMAX*nstat)
      save ravg, stat, stat_y

      save dragx_avg

      logical ifverbose
      common /gaaa/    wo1(lx1,ly1,lz1,lelv)
     &              ,  wo2(lx1,ly1,lz1,lelv)
     &              ,  wo3(lx1,ly1,lz1,lelv)

      real tplus
      real tmn, tmx

      integer bIDs(1)
      save iobj_wall

      n     = nx1*ny1*nz1*nelv
      nelx  = NUMBER_ELEMENTS_X
      nely  = NUMBER_ELEMENTS_Y
      nelz  = NUMBER_ELEMENTS_Z

      if (istep.eq.0) then
         call getStateReward()
         write(*,*) state
         write(*,*) smarties_comm, state, STATE_SIZE, AGENT_ID 
         call smarties_sendInitState(smarties_comm, state, 
     &                               STATE_SIZE, AGENT_ID)
         write(*,*) state
      end if 

      call smarties_recvAction(smarties_comm, action, 
     &                         NUM_ACTIONS, AGENT_ID)
      call getStateReward()
      call smarties_sendState(smarties_comm, state, STATE_SIZE,
     &       reward, AGENT_ID)


      if (istep.eq.0) then
         bIDs(1) = 1
         call create_obj(iobj_wall,bIDs,1)
         nm = iglsum(nmember(iobj_wall),1)
         if(nid.eq.0) write(6,*) 'obj_wall nmem:', nm 
         call prepost(.true.,'  ')
      endif

      ubar = glsc2(vx,bm1,n)/volvm1
      e2   = glsc3(vy,bm1,vy,n)+glsc3(vz,bm1,vz,n)
      e2   = e2/volvm1
      if (nfield.gt.1) then
        tmn  = glmin(t,n)
        tmx  = glmax(t,n)
      endif
      if(nid.eq.0) write(6,2) time,ubar,e2,tmn,tmx
   2               format(1p5e13.4,' monitor')

      if (time.lt.tSTATSTART) return

c
c     What follows computes some statistics ...
c

      if(ifoutfld) then
        if (ldimt.ge.2) call lambda2(t(1,1,1,1,2))
        if (ldimt.ge.3) call comp_vort3(t(1,1,1,1,3),wo1,wo2,vx,vy,vz)
      endif

      if(icalld.eq.0) then
        if(nid.eq.0) write(6,*) 'Start collecting statistics ...'

        nxm = 1 ! mesh is linear
        call interp_setup(intp_h,0.0,nxm,nelt)
        nint = 0
        if (nid.eq.0) then
          nint = INTP_NMAX
          call cfill(xint,XCINT,size(xint))
          do i = 1,INTP_NMAX 
             yi = (i-1.)/(INTP_NMAX-1)
             yint(i) = tanh(BETAM*(2*yi-1))/tanh(BETAM)
          enddo
          call cfill(zint,ZCINT,size(zint))
        endif
        iffpts = .true. ! dummy call to find points
        call interp_nfld(stat_y,ravg,1,xint,yint,zint,nint,
     $                   iwk,rwk,INTP_NMAX,iffpts,intp_h)
        iffpts = .false.
        call gtpp_gs_setup(igs_x,nelx     ,nely,nelz,1) ! x-avx
        call gtpp_gs_setup(igs_z,nelx*nely,1   ,nelz,3) ! z-avg

        call rzero(ravg,size(ravg))
        dragx_avg = 0
        atime     = 0
        timel     = time
        ntdump    = int(time/tSTATFREQ)

        icalld = 1
      endif

      dtime = time - timel
      atime = atime + dtime

      ! averaging over time
      if (atime.ne.0. .and. dtime.ne.0.) then
        beta      = dtime / atime
        alpha     = 1. - beta

        ifverbose = .false.
        call avg1(ravg(1,1),vx   ,alpha,beta,n,'uavg',ifverbose)
        call avg2(ravg(1,2),vx   ,alpha,beta,n,'urms',ifverbose)
        call avg2(ravg(1,3),vy   ,alpha,beta,n,'vrms',ifverbose)
        call avg2(ravg(1,4),vz   ,alpha,beta,n,'wrms',ifverbose)
        call avg3(ravg(1,5),vx,vy,alpha,beta,n,'uvmm',ifverbose)

        call avg1(ravg(1,6),t    ,alpha,beta,n,'tavg',ifverbose)
        call avg2(ravg(1,7),t    ,alpha,beta,n,'trms',ifverbose)
        call avg3(ravg(1,8),vx,t ,alpha,beta,n,'utmm',ifverbose)
        call avg3(ravg(1,9),vy,t ,alpha,beta,n,'vtmm',ifverbose)

        call torque_calc(1.0,x0,.false.,.false.) ! compute wall shear
        dragx_avg = alpha*dragx_avg + beta*dragx(iobj_wall)
      endif

      timel = time

      ! write statistics to file
      if(istep.gt.0 .and. time.gt.(ntdump+1)*tSTATFREQ) then
         ! averaging over statistical homogeneous directions (x-z)
         do i = 1,nstat
            call planar_avg(wo1      ,ravg(1,i),igs_x)
            call planar_avg(stat(1,i),wo1      ,igs_z)
         enddo
         stat(:,2) = stat(:,2)-stat(:,1)**2

         if (nfield.gt.1) then
            ! evaluate d<T>/dy at the lower wall
            call opgrad(wo1,wo2,wo3,stat(1,6))
            call dssum(wo2,lx1,ly1,lz1)
            call col2(wo2,binvm1,n)
            call interp_nfld(stat_y,wo2,1,xint,yint,zint,nint,
     $                       iwk,rwk,INTP_NMAX,iffpts,intp_h)
            dTdy_w = stat_y(1)
         else
            dTdy_w = 1.
         endif

         ! extract data along wall normal direction (1D profile)
         call interp_nfld(stat_y,stat,nstat,xint,yint,zint,nint,
     $                    iwk,rwk,INTP_NMAX,iffpts,intp_h)

         ntdump = ntdump + 1
         if (nid.ne.0) goto 998 

         rho    = param(1)
         dnu    = param(2)
         A_w    = 2 * XLEN * ZLEN
         tw     = dragx_avg / A_w
         u_tau  = sqrt(tw / rho)
         qw     = -param(8) * dTdy_w
         t_tau  = 1/u_tau * qw
         Re_tau = u_tau / dnu
         tplus  = time * u_tau**2 / dnu

         write(6,*) 'Dumping statistics ...', Re_tau, t_tau
 
         open(unit=56,file='vel_fluc_prof.dat')
         write(56,'(A,1pe14.7)') '#time = ', time
         write(56,'(A)') 
     $    '#  y    y+    uu    vv    ww    uv    tt    ut    -vt'

         open(unit=57,file='mean_prof.dat')
         write(57,'(A,1pe14.7)') '#time = ', time
         write(57,'(A)') 
     $    '#  y    y+    Umean    Tmean'

         do i = 1,nint
            yy = 1+yint(i)
            write(56,3) 
     &           yy,
     &           yy*Re_tau,
     &           stat_y(1*nint+i)/u_tau**2,
     &           stat_y(2*nint+i)/u_tau**2,
     &           stat_y(3*nint+i)/u_tau**2,
     &           stat_y(4*nint+i)/u_tau**2,
     &           (stat_y(6*nint+i)-(stat_y(5*nint+i))**2)/t_tau**2,
     &           stat_y(7*nint+i)/qw,
     &           -stat_y(8*nint+i)/qw

            write(57,3) 
     &           yy,
     &           yy*Re_tau, 
     &           stat_y(0*nint+i)/u_tau,
     &           stat_y(5*nint+i)/t_tau

  3         format(1p15e17.9)
         enddo
         close(56)
         close(57)

 998  endif

      return
      end
c-----------------------------------------------------------------------
      subroutine userbc (ix,iy,iz,iside,ieg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      include 'SMARTIES'

      TRX = action(1)
      TRZ = action(1)
   
      return
      end
c-----------------------------------------------------------------------
      subroutine useric (ix,iy,iz,ieg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'

      integer idum
      save    idum 
      data    idum / 0 /

      real C, k, kx, ky

      Re_tau = 550 
      C      = 5.17
      k      = 0.41

      yp = (1-y)*Re_tau
      if (y.lt.0) yp = (1+y)*Re_tau
      
      ! Reichardt function
      ux  = 1/k*log(1+k*yp) + (C - (1/k)*log(k)) *
     $      (1 - exp(-yp/11) - yp/11*exp(-yp/3))
      ux  = ux * Re_tau*param(2)

      eps = 1e-2
      kx  = 23
      kz  = 13

      alpha = kx * 2*PI/XLEN
      beta  = kz * 2*PI/ZLEN 

      ! add perturbation to trigger turbulence 
      ux  = ux  + eps*beta  * sin(alpha*x)*cos(beta*z) 
      uy  =       eps       * sin(alpha*x)*sin(beta*z)
      uz  =      -eps*alpha * cos(alpha*x)*sin(beta*z)

      ! thin boundary layer at the lower wall
      gamma = 5e-6 ! initial thickness
      temp = erfc((1+y)/sqrt(1./param(8) * gamma))

      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat   ! This routine to modify element vertices
      include 'SIZE'      ! _before_ mesh is generated, which 
      include 'TOTAL'     ! guarantees GLL mapping of mesh.

      n = nelv * 2**ldim
      xmin = glmin(xc,n)
      xmax = glmax(xc,n)
      ymin = glmin(yc,n)
      ymax = glmax(yc,n)
      zmin = glmin(zc,n)
      zmax = glmax(zc,n)

      xscale = XLEN/(xmax-xmin)
      yscale = 1./(ymax-ymin)
      zscale = ZLEN/(zmax-zmin)

      do i=1,n
         xc(i,1) = xscale*xc(i,1)
         yc(i,1) = yscale*yc(i,1)
         yc(i,1) = tanh(BETAM*(2*yc(i,1)-1))/tanh(BETAM)
         zc(i,1) = zscale*zc(i,1)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat2   ! This routine to modify mesh coordinates
      include 'SIZE'
      include 'TOTAL'


      do iel=1,nelt
      do ifc=1,2*ndim
         if (cbc(ifc,iel,1) .eq. 'sh ') boundaryID(ifc,iel) = 1 
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat3
      include 'SIZE'
      include 'TOTAL'

      param(54) = 0   ! use >0 for const flowrate or <0 bulk vel
                      ! flow direction is given by (1=x, 2=y, 3=z) 
      param(55) = 0   ! flowrate/bulk-velocity 

      return
      end
c-----------------------------------------------------------------------
      subroutine getStateReward()
      include 'SMARTIES'
      include 'SIZE'
      include 'TOTAL'

      common /gaaa/    wo1(lx1,ly1,lz1,lelv)
     &              ,  wo2(lx1,ly1,lz1,lelv)
     &              ,  wo3(lx1,ly1,lz1,lelv)
      integer iel, ifc, i0, i1, j0, j1, k0, k1
      save iel, i0, j0, j1, k0
      integer igs_x, igs_z
      save igs_x, igs_z
      real dnu
 
      dnu = param(2)

      nelx  = NUMBER_ELEMENTS_X
      nely  = NUMBER_ELEMENTS_Y
      nelz  = NUMBER_ELEMENTS_Z

      if (istep.eq.0) then
         call gtpp_gs_setup(igs_x,nelx     ,nely,nelz,1) ! x-avx
         call gtpp_gs_setup(igs_z,nelx*nely,1   ,nelz,3) ! z-avg
         do iel=1,nelt
         do ifc=1,2*ndim
            if (cbc(ifc,iel,1) .eq. 'sh ') then 
               call facind (i0,i1,j0,j1,k0,k1,lx1,ly1,lz1,ifc)
               if (j0 .eq.   1) j1 = 2
               if (j0 .eq. ly1) j1 = ly1-1
               goto 1001 
            end if
         enddo
         enddo
      if (nid.eq.0) write(*,*) 'ERROR: no wall for ',nid
 1001 end if

      call planar_avg(wo1,vx,igs_x)
      call planar_avg(wo2,wo1,igs_z)

      state(1) = wo2(i0,j0,k0,iel) ! mean u at wall 
      state(2) = wo2(i0,j1,k0,iel) ! mean u off wall 
       
      call opgrad(wo1,wo2,wo3,vx)
      call dssum(wo2,lx1,ly1,lz1)
      call col2(wo2,binvm1,n)
      call planar_avg(wo1,wo2,igs_x)
      call planar_avg(wo2,wo1,igs_z)

      state(3) = wo2(i0,j0,k0,iel) ! mean dudy at wall
      state(4) = wo2(i0,j1,k0,iel) ! mean dudy off wall

      call planar_avg(wo1,vz,igs_x)
      call planar_avg(wo2,wo1,igs_z)

      state(5) = wo2(i0,j0,k0,iel) ! mean w at wall
      state(6) = wo2(i0,j1,k0,iel) ! mean w off wall

      reward = 1.0/abs((-dnu*state(3))**0.5 - (DPDX)**0.5) 
      if (isnan(reward)) reward = 1e5

      return
      end

c automatically added by makenek
      subroutine usrdat0() 

      return
      end

c automatically added by makenek
      subroutine usrsetvert(glo_num,nel,nx,ny,nz) ! to modify glo_num
      integer*8 glo_num(1)

      return
      end

c automatically added by makenek
      subroutine userqtl

      call userqtl_scig

      return
      end