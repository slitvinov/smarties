#include <stdio.h>
#include <cstdint>

int main()
{
  fprintf(stderr, "integer %d\n", sizeof(int));
  fprintf(stderr, "double %d\n", sizeof(double));
  fprintf(stderr, "pointer %d\n", sizeof(uintptr_t));
}
