#include "Communicator.h"

#include <iostream>
#include <cmath>
#include <cassert>

#include <netdb.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <fcntl.h>

static int send_all(int fd, void *buffer, unsigned int size);
static int recv_all(int fd, void *buffer, unsigned int size);

void Communicator::sendState(int& agentId, _AGENT_STATUS& info,
                             std::vector<double>& state, double& reward)
{
    assert(state.size() == nStates);
    o <<"Send: "<<agentId<<" "<<msgID++<<" "<< info<<" ";
    {int *ptr=(int*)(dataout);   *ptr=agentId;}
    {int *ptr=(int*)(dataout+1); *ptr=info;  }

    for (int j=0; j<nStates; j++) {
        if (std::isnan(state[j]) || std::isinf(state[j]) ) abort();
        *(dataout +j+2) = state[j];
        o << state[j] << " ";
    }

    *(dataout +2+nStates) = reward;
    o << reward << "\n";

    send_all(Socket, dataout, sizeout);
    if (info == 2)  msgID = 0;
}

void Communicator::recvState(int& agentId, _AGENT_STATUS& info,
                             std::vector<double>& state, double& reward)
{
    assert(state.size() == nStates);
    int bytes = 0;
    //printf("RECEIVING %d,%d\n",Socket,sizein);
    if ((bytes = recv_all(Socket, datain, sizein)) <= 0) {
        if (bytes == 0) printf("socket %d hung up\n", Socket);
        else perror("(1) recv");

        close(Socket);
        info = _AGENT_FAILCOMM;
    } else { // (bytes == nbyte)
        agentId = *((int*)  datain   );
        info = *((int*) (datain+1));
        o <<"Recv: "<<agentId<<" "<<msgID++<<" "<< info<<" ";

        int k = 2;
        for (int j=0; j<nStates; j++) {
            state[j] = datain[k++];
            o << state[j] << " ";
            assert(not std::isnan(state[j]) && not std::isinf(state[j]));
        }

        //debug3(" %f (%d)\n",datain[k],k);
        reward = datain[k++];
        o << reward << "\n";
        assert(not std::isnan(reward) && not std::isinf(reward));
        assert(k==3+nStates);
    }
}

void Communicator::recvAction(std::vector<double>& actions)
{
    assert(actions.size() == nActions);
    int bytes = recv_all(Socket, datain, sizein);
    if (bytes <= 0) {
        printf("selectserver: socket hung up\n");
        fflush(0);
        abort();
    }
    o << "Recv: ";
    for (int j=0; j<nActions; j++) {
        actions[j] = *(datain +j);
        o << actions[j] << " ";
    }
    o << "\n";
}

void Communicator::sendAction(std::vector<double>& actions)
{
    assert(actions.size() == nActions);
    o << "Sent: ";
    for (int i=0; i<nActions; i++) {
        dataout[i] = actions[i];
        o << actions[i] << " ";
        assert(not std::isnan(actions[i]) && not std::isinf(actions[i]));
    }
    o << "\n";

    send_all(Socket, dataout, sizeout);
}

Communicator::~Communicator()
{
    std::cout<<o.str()<<std::endl;
    close(Socket);
    free(datain);
    free(dataout);
}

Communicator::Communicator(int _sockID, int _statedim, int _actdim, int _server, int _issim):
workerid(_sockID==0?1:_sockID),nActions(_actdim),nStates(_statedim),
isServer(_sockID==0||_server), msgID(0)
{
    if (_issim) {
      sizeout = (3+nStates)*sizeof(double);
      sizein  =    nActions*sizeof(double);
    } else {
      sizein  = (3+nStates)*sizeof(double);
      sizeout =    nActions*sizeof(double);
    }
    sprintf(SOCK_PATH, "%s%d", "/tmp/smarties_sock_", workerid);
    printf("SOCK_PATH=->%s<-\n", SOCK_PATH);
    dataout = (double *) malloc(sizeout);
    memset(dataout, 0, sizeout);
    datain  = (double *) malloc(sizein);
    memset(datain, 0, sizein);
    printf("nStates:%d nActions:%d sizein:%d sizeout:%d\n",
          nStates, nActions, sizein, sizeout);
    if(_sockID==0)
      setupClient(0, std::string());
}


void Communicator::setupServer()
{
  /* Create a socket */
  if ((ListenerSocket = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
      perror("socket");
      exit(1);
  }
  unlink(SOCK_PATH);

  bzero(&serverAddress, sizeof(serverAddress));
  serverAddress.sun_family = AF_UNIX;
  strcpy(serverAddress.sun_path, SOCK_PATH);
  const int servlen = sizeof(serverAddress.sun_family) +
                      strlen(serverAddress.sun_path);

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

void Communicator::setupClient(const int iter, std::string execpath)
{
  //Spawn server
  const int rf = fork();
  if (rf == 0) {
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

      sprintf(line, execpath.c_str());
      //parse(line, largv);     // prepare argv

      #if 1==1 //if true goes to stdout
      char output[256];
      sprintf(output, "output");
      int fd = open(output, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
      dup2(fd, 1);    // make stdout go to file
      dup2(fd, 2);    // make stderr go to file
      close(fd);      // fd no longer needed
      #endif

      printf("About to exec.... \n");
      std::cout << execpath << std::endl; //<< *largv << endl;

      //int res = execlp(execpath.c_str(), execpath.c_str(), NULL);
      const int res = execlp(execpath.c_str(),
                             execpath.c_str(),
                             std::to_string(workerid).c_str(),
                             NULL);
      //int res = execvp(*largv, largv);

      if (res < 0) {
        fprintf(stderr,"Unable to exec file '%s'!\n", execpath.c_str());
        abort();
      }
  }

  //printf("waiting for server to setup everything..\n");
  //sleep(2); //pause is not safe with MPI
  //printf("ok, I continue...\n");

  Socket = socket(AF_UNIX, SOCK_STREAM, 0);

  int _true = 1;
  if(setsockopt(Socket, SOL_SOCKET, SO_REUSEADDR, &_true, sizeof(int))<0) {
     perror("Sockopt failed\n");
     exit(1);
  }

  /* Specify the server */
  bzero((char *)&serverAddress, sizeof(serverAddress));
  serverAddress.sun_family = AF_UNIX;
  strcpy(serverAddress.sun_path, SOCK_PATH);
  const int servlen = sizeof(serverAddress.sun_family) + strlen(serverAddress.sun_path);

  /* Connect to the server */
  while (connect(Socket, (struct sockaddr *)&serverAddress, servlen) < 0) {
      //perror("connecting...\n");
  }
}

void Communicator::closeSocket()
{
    close(ListenerSocket);
    free(datain);
    free(dataout);
}

void Communicator::dbg(double *x, int *pn)
{
    int i, n = *pn;
    int me = getpid();	/* spanwer_id : workerid */

    printf("spanwer(%d): running task with params (", me);
    for (i = 0; i < n-1; i++)
        printf("%.6lf,", x[i]);
    printf("%.6lf)\n", x[i]);
    fflush(0);
}



/*************************************************************************/
/**************************   HELPER ROUTINES   **************************/
/*************************************************************************/

static int recv_all(int fd, void *buffer, unsigned int size)
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

static int send_all(int fd, void *buffer, unsigned int size)
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
