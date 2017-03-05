#include "Communicator.h"

#include <iostream>
#include <cmath>
#include <cassert>

#include <netdb.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <limits>

static int send_all(int fd, void *buffer, unsigned int size);
static int recv_all(int fd, void *buffer, unsigned int size);

//Called by app
void Communicator::sendState(int iAgent, _AGENT_STATUS status,
                             std::vector<double>& state, double reward)
{
    if(app_rank) return; //only rank 0 of the app sends state
    assert(state.size() == (std::size_t) nStates);

    byte * buf = dataout;
    std::ostringstream o;

    o <<"Send: "<<iAgent<<" "<<msgID++<<" "<<status<<" ";

    {
      int* ptr = (int*) buf;
      ptr[0] = iAgent;
      ptr[1] = status;
      buf += 2*sizeof(int);
    }

    {
      double* ptr = (double*) buf;
      for (int j=0; j<nStates; j++) {
          ptr[j] = state[j];
          o << state[j] << " ";
          assert(not std::isnan(state[j]));
          assert(not std::isinf(state[j]));
      }
      ptr[nStates] = reward;
      o << reward;
      assert(not std::isnan(reward));
      assert(not std::isinf(reward));
      buf += (nStates+1)*sizeof(double);
    }

    printLog(o);
    assert(buf - dataout == sizeout);

    if (smarties_rank) sendStateMPI();
    else sendStateClient();

    if (status == _AGENT_LASTCOMM) msgID = 0;
}

//Called by app
void Communicator::recvAction(std::vector<double>& actions)
{
    assert(actions.size() == (std::size_t) nActions);

    if (smarties_rank)
      recvActionMPI();
    else
      recvActionClient();

    synchronizeApp();

    double* buf = (double*) datain;
    std::ostringstream o;

    o <<"Recv: "<<agentId<<" "<<msgID<<" ";

    for (int j=0; j<nActions; j++) {
        actions[j] = buf[j];
        o << actions[j] << " ";
        assert(not std::isnan(actions[j]));
    }

    printLog(o);
}

void Communicator::sendStateClient()
{
  if(app_rank) return;

  const int bytes = send_all(Socket, dataout, sizeout);

  if(bytes <= 0) {
    printf("Lost contact with smarties, aborting...\n");
    fflush(0);
  #ifdef MPI_INCLUDED
    if(app_size)
    MPI_Abort(comm_MPI, 1);
    else
  #endif
    abort();
  }
}

void Communicator::recvActionClient()
{
  if(app_rank) return;

  int bytes = recv_all(Socket, datain, sizein);

  if (bytes <= 0) {
      printf("Lost contact with smarties, aborting..\n");
      fflush(0);
    #ifdef MPI_INCLUDED
      if(app_size)
      MPI_Abort(comm_MPI, 1);
      else
    #endif
      abort();
  }
}

void Communicator::printLog(const std::ostringstream o)
{
  if (! verbose || app_rank) return;
  fout.open(logfile.c_str(), ios::app);
  fout << o.str() << endl;
  fout.flush();
  fout.close();
}


void Communicator::synchronizeApp()
{
  if(!app_size) return;
  //send same action to all ranks of the sim
  #ifdef MPI_INCLUDED
  if(!app_rank)
    for (int i=1; i<app_size; ++i)
    MPI_Send(datain, sizein, MPI_BYTE, i, 42, appComm);
  else
    MPI_Recv(datain, sizein, MPI_BYTE, 0, 42, appComm, MPI_STATUS_IGNORE);
  #endif
}

void Communicator::recvActionMPI()
{
  if(app_rank) return;
  #ifdef MPI_INCLUDED
  MPI_Recv(datain, sizein, MPI_BYTE, 0, 0, masterComm, MPI_STATUS_IGNORE);
  #else
  abort();
  #endif
}

void Communicator::sendStateMPI()
{
  assert(!app_rank);
  #ifdef MPI_INCLUDED
  MPI_Ssend(dataout, sizeout, MPI_BYTE, 0, 2, masterComm);
  #else
  abort();
  #endif
}

void redirect_stdout_init(const int rank)
{
	fflush(stdout);
	fgetpos(stdout, &pos);
	fd = dup(fileno(stdout));
	char buf[500];
	sprintf(buf, "stdout_%03d.status", (int)rank);
	std::string numbered_stdout = std::string(buf);
	freopen(numbered_stdout.c_str(), "w", stdout);
}

void redirect_stdout_finalize()
{
	dup2(fd, fileno(stdout));
	close(fd);
	clearerr(stdout);
	fsetpos(stdout, &pos);        /* for C9X */
}

int Communicator::recvStateFromApp()
{
    int bytes = recv_all(Socket, dataout, sizeout);

    if (bytes <= 0)
    {
        if (bytes == 0) printf("socket %d hung up\n", Socket);
        else perror("(1) recv");
        close(Socket);
        close(ListenerSocket);
        {
          int* ptr = (int*) dataout;
          ptr[0] = 0;
          ptr[1] = _AGENT_FAILCOMM;
        }
        iter++;
        sendStateMPI();
    }
    else assert(bytes == sizeout);

    return bytes <= 0;
}

void Communicator::sendActionToApp()
{
    send_all(Socket, datain, sizein);
}

void Communicator::unpackAction(std::vector<double>& action)
{
    assert(action.size() == (std::size_t) nActions);
    const double* buf = (double*) datain;
    for (int j=0; j<nActions; j++) {
        action[j] = buf[j];
        assert(not std::isnan(action[j]));
        assert(not std::isinf(action[j]));
    }
    buf += nActions*sizeof(double);
    assert(buf - datain == sizein);
}

void Communicator::unpackState(int& iAgent, _AGENT_STATUS& status,
                             std::vector<double>& state, double& reward)
{
  assert(state.size() == (std::size_t) nStates);
  const byte* buf = dataout;
  std::ostringstream o;

  {
    const int* const ptr = (int*) buf;
    iAgent = ptr[0];
    status = ptr[1];
    buf += 2*sizeof(int);
  }
  o <<"Send: "<<iAgent<<" "<<msgID++<<" "<<status<<" ";
  {
    const double* const ptr = (double*) buf;
    for (int j=0; j<nStates; j++) {
        state[j] = ptr[j];
        o << state[j] << " ";
        assert(not std::isnan(state[j]));
        assert(not std::isinf(state[j]));
    }
    reward = ptr[nStates];
    o << reward;
    assert(not std::isnan(reward));
    assert(not std::isinf(reward));
    buf += (nStates+1)*sizeof(double);
  }
  printLog(o);
  assert(buf - dataout == sizeout);
}

void Communicator::setupClient(const int iter, std::string execpath)
{
  //Spawn server
  const int rf = fork();
  if (rf == 0)
  {
      char line[1024];
      //char *largv[64];
      if (execpath == std::string()) {
         execpath = "./runClient.sh";
         struct stat buffer;
         while(stat("runClient.sh", &buffer)) {
           chdir("..");
           char cwd[1024];
           if (getcwd(cwd, sizeof(cwd)) != NULL)
                printf("Current working dir: %s\n", cwd);
           else perror("getcwd() error");
         }
      } else {
        mkdir(("simulation_"+std::to_string(workerid)+"_"
                            +std::to_string(iter)+"/").c_str(),
                            S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        chdir(("simulation_"+std::to_string(workerid)+"_"
                            +std::to_string(iter)+"/").c_str());
      }

      sprintf(line, "%s", execpath.c_str());
      //parse(line, largv);     // prepare argv

      #if 1==1 //if true goes to stdout
        fflush(0);
        char output[256];
        sprintf(output, "output");
        int fd = open(output, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        dup2(fd, 1);    // make stdout go to file
        dup2(fd, 2);    // make stderr go to file
        close(fd);      // fd no longer needed
      #endif

      printf("About to exec %s.... \n",execpath.c_str());
      //int res = execlp(execpath.c_str(), execpath.c_str(), NULL);
      const int res = execlp(execpath.c_str(),
                             execpath.c_str(),
                             std::to_string(workerid).c_str(),
                             NULL);
      //int res = execvp(*largv, largv);
      if (res < 0)
      {
        fprintf(stderr,"Unable to exec file '%s'!\n", execpath.c_str());
        abort();
      }
  }

  printf("waiting for server to setup everything..\n");
  sleep(2); //pause is not safe with MPI
  printf("ok, I continue...\n");
  fflush(0);
  Socket = socket(AF_UNIX, SOCK_STREAM, 0);

  int _true = 1;
  if(setsockopt(Socket, SOL_SOCKET, SO_REUSEADDR, &_true, sizeof(int))<0) {
     perror("Sockopt failed\n");
     exit(1);
  }
  printf("Created socket\n");
  fflush(0);

  /* Specify the server */
  bzero((char *)&serverAddress, sizeof(serverAddress));
  serverAddress.sun_family = AF_UNIX;
  strcpy(serverAddress.sun_path, SOCK_PATH);
  const int servlen = sizeof(serverAddress.sun_family) + strlen(serverAddress.sun_path);

  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);
  //printf("Specify the server %s\n", hostname);
  fflush(0);
  /* Connect to the server */
  while (connect(Socket, (struct sockaddr *)&serverAddress, servlen) < 0) {
      //perror("connecting...\n");
  }
  printf("Connected to server\n");
  fflush(0);

  /*
  int check = -1;
  int bytes = recv_all(Socket, &check, sizeof(int));
  if (bytes <= 0) {
      printf("selectserver: socket hung up\n");
      fflush(0);
      abort();
  }
  if (check) {
      printf("handshake failed\n");
      fflush(0);
      abort();
  }
  */
}

void Communicator::setupServer()
{
  /* Create a socket */
  printf("Server create socket\n");
  fflush(0);
  if ((ListenerSocket = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
      perror("socket");
      exit(1);
  }
  unlink(SOCK_PATH);

  bzero(&serverAddress, sizeof(serverAddress));
  serverAddress.sun_family = AF_UNIX;
  strcpy(serverAddress.sun_path, SOCK_PATH);
  const int servlen = sizeof(serverAddress.sun_family) + strlen(serverAddress.sun_path);

  printf("Server bind listener socket\n");
  fflush(0);

  if (bind(ListenerSocket, (struct sockaddr *)&serverAddress, servlen) < 0) {
      perror("bind");
      exit(1);
  }

  int _true = 1;
  if(setsockopt(ListenerSocket, SOL_SOCKET, SO_REUSEADDR, &_true, sizeof(int))<0)
  {
      perror("Sockopt failed\n");
      exit(1);
  }

  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);
  printf("Server listen listener socket on %s\n", hostname);
  fflush(0);
  /* listen (only 1)*/
  if (listen(ListenerSocket, 1) == -1) {
      perror("listen");
      exit(1);
  }

  unsigned int addr_len = sizeof(clientAddress);
  if ((Socket = accept(ListenerSocket, (struct sockaddr*)&clientAddress, &addr_len)) == -1)
  {
      perror("accept");
      return;
  }
  else printf("selectserver: new connection from on socket %d\n", Socket);
  fflush(0);
}

/*************************************************************************/
/**************************   HELPER ROUTINES   **************************/
/*************************************************************************/

int Communicator::recv_all(int fd, void *buffer, unsigned int size)
{
    int result;
    unsigned int s=size;
    char *pos = (char*)buffer;


    do {
        result=recv(fd,pos,s,0);
        if((result!=-1)&&(result>0)) {
            s -= result;
            pos += result;
        }
        else
            return result; /*-1;*/
    } while (s>0);
    //printf("recver %f\n",*((double*)buffer));
    return size;
}

int Communicator::send_all(int fd, void *buffer, unsigned int size)
{
    int result;
    unsigned int s=size;
    char *pos = (char*)buffer;

    do {
        result=send(fd,pos,s,0);
        if((result!=-1)&&(result>0)) {
            s -= result;
            pos += result;
        }
        else return result; /*-1;*/
    } while (s>0);
    return size;
}
