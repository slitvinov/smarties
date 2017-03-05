#pragma once

#include <sstream>
#include <sys/un.h>
#ifdef __MPI_CLIENT
#include <mpi.h>
#endif
#include <vector>

#define _AGENT_STATUS   int
#define _AGENT_FIRSTCOMM  1
#define _AGENT_NORMCOMM   0
#define _AGENT_LASTCOMM   2
#define _AGENT_FAILCOMM  -1

class Communicator
{
protected:
    #ifdef MPI_INCLUDED
    const MPI_Comm appComm, masterComm;
    #endif
    const int nActions, nStates, sizein, sizeout, spawner;
    const int app_rank, app_size, smarties_rank, smarties_size;
    int socket_id, msgID;
    double *datain, *dataout;

    int send_all(int fd, void *buffer, unsigned int size);
    int recv_all(int fd, void *buffer, unsigned int size);

    byte * _alloc(const int size)
    {
      #if 0
        return new byte[size];
      #else
        byte* ret = (byte*) malloc(size);
        memset(ret, 0, size);
        return ret;
      #endif
    }
    void _dealloc(byte* ptr)
    {
      if(ptr not_eq nullptr)
      {
        #if 0
          delete [] ptr;
        #else
          free(ptr);
        #endif
        ptr=nullptr;
      }
    }
    int getRank(const MPI_Comm comm)
    {
      int rank;
      MPI_Comm_rank(comm, &rank);
      return rank;
    }
    int getSize(const MPI_Comm comm)
    {
      int size;
      MPI_Comm_size(comm, &rank);
      return size;
    }

    void sendStateClient();
    void sendStateMPI();
    void recvActionClient();
    void recvActionMPI();
    void synchronizeApp();
    void printLog(const std::ostringstream o);

    void setupClient(const int iter, std::string execpath);
    void setupServer();

public:
    void sendState(int agentId,
                   int info,
                   std::vector<double>& state,
                   double reward);

    // assumption is that Action always follows a State:
    // no need to specify agent
    void recvAction(std::vector<double> & actions);

    ~Communicator()
    {
        close(Socket);
        free(datain);
        free(dataout);
    }

    //forked comm, single node job, constructor by app
    //only rule: if app gets socket=0, then it is the spawner
    Communicator::Communicator(const int _socket,const int sdim,const int adim):
    #ifdef MPI_INCLUDED
    appComm( MPI_COMM_NULL ), masterComm( MPI_COMM_NULL ),
    #endif
    nActions(adim), nStates(sdim), spawner(_socket==0), socket_id(_socket),
    sizeout(2*sizeof(int)+(1+sdim)*sizeof(double)), sizein(adim*sizeof(double)),
    app_rank(0), smarties_rank(0), app_size(0), smarties_size(0),
    dataout(_alloc(sizeout)), datain(_alloc(sizein)), msgID(0)
    {
      if (spawner) //cheap way to ensure multiple sockets can exist on same node
      {
        struct timeval clock;
        gettimeofday(&clock, NULL);
        socket_id = abs(clock.tv_usec % std::numeric_limits<int>::max());
      }
      printf("App seq: nS:%d nA:%d sin:%d sout:%d spawn:%d, socket:%d PATH=%s\n",
            nStates, nActions, sizein, sizeout, spawner, socket_id, SOCK_PATH);

      if(spawner)
        setupClient(0, std::string());
      else
        setupServer(); //app as client
    }

    //forked comm, mpi job, constructor by app
    //only rule: if app gets socket=0, then it is the spawner
    #ifdef MPI_INCLUDED
    Communicator::Communicator(const int _socket,const int sdim,const int adim,
    const MPI_Comm app): appComm(app), masterComm(MPI_COMM_NULL), nActions(adim),
    nStates(sdim), spawner(_socket==0), socket_id(_socket),
    sizeout(2*sizeof(int)+(1+sdim)*sizeof(double)), sizein(adim*sizeof(double)),
    app_rank(getRank(app)), smarties_rank(0), app_size(getSize(app)), smarties_size(0),
    dataout(_alloc(sizeout)), datain(_alloc(sizein)), msgID(0)
    {
      if (app_rank) return;

      if (spawner) //cheap way to ensure multiple sockets can exist on same node
      {
        struct timeval clock;
        gettimeofday(&clock, NULL);
        socket_id = abs(clock.tv_usec % std::numeric_limits<int>::max());
      }
      printf("App mpi: nS:%d nA:%d sin:%d sout:%d spawn:%d, socket:%d PATH=%s\n",
            nStates, nActions, sizein, sizeout, spawner, socket_id, SOCK_PATH);

      if(spawner)
        setupClient(0, std::string());
      else
        setupServer(); //app as client
    }
    #endif
};

class Communicator_smarties
{
void unpackAction(std::vector<double>& action);
void unpackState(int& iAgent, _AGENT_STATUS& status, std::vector<double>& state, double& reward);
public:
#if 0
//comm directly with master, constructor by smarties, pass this to app
Communicator::Communicator(int _sdim, int _adim, MPI_Comm smart, MPI_Comm app):
appComm(app), masterComm(smart), nActions(_adim), nStates(_sdim), msgID(0),
sizeout(2*sizeof(int) + (1+_sdim)*sizeof(double)), sizein(_adim*sizeof(double)),
app_rank(getRank(app)), smarties_rank(getRank(smart)),
app_size(getSize(app)), smarties_size(getSize(smart)),
dataout(_alloc(sizeout)), datain(_alloc(sizein))
{
#if 1
  char initd[256];
  char newd[256];

  getcwd(initd,256);
  sprintf(newd,"%s/%s", initd, jobs[i].dir);
  chdir(newd);	// go to the task private directory

  redirect_stdout_init(get_rank(app_comm));
  app_main(app_comm, largc, largv);
  redirect_stdout_finalize();

  chdir(initd);	// go up one level
#endif
  printf("nStates:%d nActions:%d sizein:%d sizeout:%d\n",
  nStates, nActions, sizein, sizeout);
}
#endif

//construtor by smarties side of fork
Communicator::Communicator(int _socket, int _sdim, int _adim, bool _spawner):
appComm(app), masterComm(smart), nActions(_adim), nStates(_sdim), msgID(0),
sizeout(2*sizeof(int) + (1+_sdim)*sizeof(double)), sizein(_adim*sizeof(double)),
app_rank(0), smarties_rank(0), app_size(0), smarties_size(0),
dataout(_alloc(sizeout)), datain(_alloc(sizein))
{
  if (spawner) //cheap way to ensure multiple sockets can exist on same node
  {
    struct timeval clock;
    gettimeofday(&clock, NULL);
    socket_id = abs(clock.tv_usec % std::numeric_limits<int>::max());
  }
  printf("App seq: nS:%d nA:%d sin:%d sout:%d spawn:%d, socket:%d PATH=%s\n",
        nStates, nActions, sizein, sizeout, spawner, socket_id, SOCK_PATH);

  if(spawner)
    setupClient(0, std::string());
  else
    setupServer(); //app as client
}
};
