c- constants -----------------------------------------------------------

c mesh dimensions
#define PI (4.*atan(1.))
#define XLEN (2.*PI)
#define ZLEN PI
#define NUMBER_ELEMENTS_X uparam(1)
#define NUMBER_ELEMENTS_Y uparam(2)
#define NUMBER_ELEMENTS_Z uparam(3)
#define DPDX uparam(4)

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


      return
      end
c-----------------------------------------------------------------------
      subroutine userbc (ix,iy,iz,iside,ieg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'

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
