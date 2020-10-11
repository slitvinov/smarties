#include <stdint.h>
#include <stddef.h>
#include <smarties_f77.h>

int
main(int argc, char **argv)
{
  return smarties_dlopen_(argc, argv, "libnek5000.so", NULL);
}
