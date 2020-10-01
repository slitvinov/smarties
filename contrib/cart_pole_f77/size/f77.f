      program size
      implicit none
      integer :: i
      double precision :: d
      integer*8 :: p
      write(*, *) "integer", sizeof(i)
      write(*, *) "double", sizeof(d)
      write(*, *) "pointer", sizeof(p)
      end program size
