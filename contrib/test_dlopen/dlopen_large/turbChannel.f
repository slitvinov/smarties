c- constants -----------------------------------------------------------

c mesh dimensions
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

      if (any(isnan(vx))) lastep = 1 ! kills Nek if NaN

      AGENT_ID = 0 ! need to change if using multiple agents

      if (istep.eq.0) then
         call getStateReward()
         call smarties_sendInitState(smarties_comm, state, 
     &                               STATE_SIZE, AGENT_ID)
         return
      end if 

      call getStateReward()
      call smarties_recvAction(smarties_comm, action, 
     &                         NUM_ACTIONS, AGENT_ID)
      call smarties_sendState(smarties_comm, state, STATE_SIZE,
     &                        reward, AGENT_ID)

      return
      end
c-----------------------------------------------------------------------
      subroutine userbc (ix,iy,iz,iside,ieg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      include 'SMARTIES'

      ! sets Neumann boundary condition to smarties action
      TRX = action(1)
      TRY = 0.
      TRZ = 0. 
   
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
         zc(i,1) = zscale*zc(i,1)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat2   ! This routine to modify mesh coordinates
      include 'SIZE'
      include 'TOTAL'

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
      real dnu, rho
 
      dnu = param(2)
      rho = param(1)

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
      call col2(wo2,binvm1,lx1*ly1*lz1*lelv)
      call planar_avg(wo1,wo2,igs_x)
      call planar_avg(wo2,wo1,igs_z)

      state(3) = wo2(i0,j0,k0,iel) ! mean dudy at wall
      state(4) = wo2(i0,j1,k0,iel) ! mean dudy off wall

      call planar_avg(wo1,vz,igs_x)
      call planar_avg(wo2,wo1,igs_z)

      state(5) = wo2(i0,j0,k0,iel) ! mean w at wall
      state(6) = wo2(i0,j1,k0,iel) ! mean w off wall

      reward = 10-3000.*abs(-dnu*state(3) - DPDX)*rho
      write(*,*) 'reward', reward
      if (isnan(reward)) reward = -10000

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
