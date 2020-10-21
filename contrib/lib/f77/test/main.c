#include <dlfcn.h>
int main() {
  void *lib;
  void (*test)(void);
  lib = dlopen("./a.out", RTLD_LAZY | RTLD_LOCAL);
  test = dlsym(lib, "test_");
  test();
}
