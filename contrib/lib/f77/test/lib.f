      subroutine test
      intrinsic isnan
      real x
      x = -1.0
      write(6, *) sqrt(x), isnan(sqrt(x))
      end
