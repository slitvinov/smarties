#pragma once

#include <sstream>
#include <sys/un.h>
#ifdef __MPI_CLIENT
#include <mpi.h>
#endif
#ifdef __Smarties_
#include "Settings.h"
#endif
#include <vector>

#define _AGENT_STATUS   int
#define _AGENT_FIRSTCOMM  1
#define _AGENT_NORMALCOMM 0
#define _AGENT_LASTCOMM   2
#define _AGENT_FAILCOMM   3

class Communicator
{
protected:
    #ifdef MPI_INCLUDED
    const MPI_Comm appComm, masterComm;
    #endif
    const int sizeout, sizein, nActions, nStates, spawner;
    const int app_rank, smarties_rank, app_size, smarties_size, verbose;
    double*const dataout;
    double*const datain;
    const std::string execpath, paramfile, logfile;
    int msg_id, iter, socket_id, Socket, ServerSocket;
    int fd;
    fpos_t pos;

    char SOCK_PATH[256];
    struct sockaddr_un serverAddress, clientAddress;

    void intToDoublePtr(const int i, double*const ptr) const
    {
      int*const buf = (int*)ptr;
      *buf = i;
    }
    int doublePtrToInt(const double*const ptr) const
    {
      return *((int*)ptr);
    }
    double* _alloc(const int size) const
    {
      #if 0
        return new double[size/sizeof(double)];
      #else
        double* ret = (double*) malloc(size);
        memset(ret, 0, size);
        return ret;
      #endif
    }
    void _dealloc(double* ptr) const
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
    #ifdef MPI_INCLUDED
    int getRank(const MPI_Comm comm) const
    {
      if (comm == MPI_COMM_NULL) return -1;
      int rank;
      MPI_Comm_rank(comm, &rank);
      return rank;
    }
    int getSize(const MPI_Comm comm) const
    {
      if (comm == MPI_COMM_NULL) return -1;
      int size;
      MPI_Comm_size(comm, &size);
      return size;
    }
    #endif

    //used by app in forked process paradigm
    void sendStateClient();
    void recvActionClient();
    //match actions across all ranks of app
    void synchronizeApp();

    void printLog(const std::string o);
    void printBuf(const double*const ptr, const int size);

    void redirect_stdout_stderr();
    void launch_exec(const std::string exec);
    void launch_smarties();
    void launch_app();
    void setupClient();
    void setupServer();

public:
    int getStateDim()  {return nStates;}
    int getActionDim() {return nActions;}
  //called by app to interact with smarties
    void sendState(const int iAgent, const _AGENT_STATUS status,
                    const std::vector<double> state, const double reward);
    void recvAction(std::vector<double>& actions);

  //send buffers to master
    void sendStateMPI();
    void recvActionMPI();
    void launch();

  #ifdef __Smarties_
    double* getDatain()  { return datain;  }
    double* getDataout() { return dataout; }
    void answerTerminateReq(const double answer);
    void unpackAction(std::vector<double>& action);
    void unpackState(int& iAgent, _AGENT_STATUS& status,
                     std::vector<double>& state, double& reward);
    int recvStateFromApp();
    void sendActionToApp();

    void restart(std::string fname);
    void save() const;

    void ext_app_run();
    int jobs_init(char *line, char **largv);
    void redirect_stdout_init();
    void redirect_stdout_finalize();
    //called by smarties
    Communicator(const int socket, const int sdim, const int adim,
      const bool _spawner, const std::string _exec, const MPI_Comm scom,
      const std::string log = std::string(), const int verbose = 0);

    Communicator(const int sdim, const int adim, const MPI_Comm scom,
      const MPI_Comm acom, const std::string exec, const std::string params,
      const std::string log = std::string(), const int verbose = 0);
  #endif

    Communicator(const int socket, const int sdim, const int adim,
      const std::string log = std::string(), const int verbose = 0);

    #ifdef MPI_INCLUDED
    Communicator(const int socket, const int sdim, const int adim,
      const MPI_Comm app, const std::string log = std::string(),
      const int verbose = 0);
    #endif

    ~Communicator();
};
