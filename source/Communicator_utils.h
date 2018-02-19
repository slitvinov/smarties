/*************************************************************************/
/**************************   HELPER ROUTINES   **************************/
/*************************************************************************/
#include <iostream>
#include <cmath>
#include <cassert>
#include <dirent.h>

#include <netdb.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <limits>

inline void intToDoublePtr(const int i, double*const ptr)
{
  assert(i>=0);
  *ptr = (double)i+0.1;
}
inline int doublePtrToInt(const double*const ptr)
{
  return (int)*ptr;
}
inline double* _alloc(const int size)
{
    double* ret = (double*) malloc(size);
    memset(ret, 0, size);
    return ret;
}
inline void _dealloc(double* ptr)
{
  if(ptr not_eq nullptr) {
    free(ptr);
    ptr=nullptr;
  }
}

int recv_all(int fd, void *buffer, unsigned int size);

int send_all(int fd, void *buffer, unsigned int size);

int parse2(char *line, char **argv);

int cp(const char *from, const char *to);

int copy_from_dir(const std::string name);

void comm_sock(int fd, const bool bsend, double*const data, const int size);

#ifdef MPI_INCLUDED
inline int getRank(const MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}
inline int getSize(const MPI_Comm comm)
{
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}
#endif
