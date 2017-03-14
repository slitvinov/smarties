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
static int parse2(char *line, char **argv);

void Communicator::sendState(const int iAgent, const _AGENT_STATUS status,
                          const std::vector<double> state, const double reward)
{
    if(app_rank) return; //only rank 0 of the app sends state
    assert(state.size() == (std::size_t) nStates);
    std::ostringstream o;

    o <<"Send: "<<iAgent<<" "<<msg_id++<<" "<<status<<" ";
    assert(iAgent>=0);
    intToDoublePtr(iAgent, dataout+0);
    intToDoublePtr(status, dataout+1);
    for (int j=0; j<nStates; j++) {
        dataout[j+2] = state[j];
        o << state[j] << " ";
        assert(not std::isnan(state[j]));
        assert(not std::isinf(state[j]));
    }
    dataout[nStates+2] = reward;
    o << reward;
    assert(not std::isnan(reward));
    assert(not std::isinf(reward));

    if (logfile != std::string()) {
      if(verbose) printLog(o.str());
      else  printBuf(dataout, sizeout);
    }

    //std::cout << o.str() << std::endl; fflush(0);
    if (smarties_rank) sendStateMPI();
    else sendStateClient();

    if (status == _AGENT_LASTCOMM) {
      //receive continue/abort
      if(!smarties_rank) {//temporary: add continue command to master
       recvActionClient();
       synchronizeApp();
       if (datain[0]<0) abort();
      }
      msg_id = 0;
    }
}

void Communicator::recvAction(std::vector<double>& actions)
{
    assert(actions.size() == (std::size_t) nActions);

    if (smarties_rank) recvActionMPI();
    else recvActionClient();

    synchronizeApp();

    double*const buf = (double*) datain;
    std::ostringstream o;

    o <<"Recv: "<<msg_id<<" ";

    for (int j=0; j<nActions; j++) {
        actions[j] = buf[j];
        o << actions[j] << " ";
        assert(not std::isnan(actions[j]));
    }

    //std::cout << o.str() << std::endl; fflush(0);
    if (logfile == std::string()) return;
    if(verbose) printLog(o.str());
    else  printBuf(datain, sizein);
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
    MPI_Abort(appComm, 1);
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
      MPI_Abort(appComm, 1);
      else
    #endif
      abort();
  }
}

void Communicator::printLog(const std::string o)
{
  if (! verbose || app_rank) return;
  const std::string fname = logfile+std::to_string(iter)+".txt";
  FILE * f = fopen(fname.c_str(), "a");
  if (f != NULL)
    fprintf(f, "%s\n", o.c_str());
  fclose(f);
}

void Communicator::printBuf(const double*const buf, const int size)
{
  if (app_rank) return;
  const std::string fname = logfile+std::to_string(iter)+".raw";
  FILE * pFile = fopen (fname.c_str(), "ab");
  fwrite (buf, sizeof(double), size/sizeof(double), pFile);
  fclose (pFile);
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
  fflush(0);
  MPI_Ssend(dataout, sizeout, MPI_BYTE, 0, 1, masterComm);
  #else
  abort();
  #endif
}

void Communicator::launch_smarties()
{
  #ifdef __Smarties_
  printf("launch_smarties\n"); fflush(0);
  abort();
  #else
  //go up til a file runClient is found: shaky
  struct stat buffer;
  while(stat("runClient.sh", &buffer))
  {
    chdir("..");
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
         printf("Current working dir: %s\n", cwd);
    else perror("getcwd() error");
  }

  redirect_stdout_stderr();

  launch_exec("./runClient.sh");
  #endif
  abort(); //if app returns: TODO
}

void Communicator::redirect_stdout_stderr()
{
  #if 1
  fflush(0);
  char output[256];
  sprintf(output, "output");
  fd = open(output, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  dup2(fd, 1);    // make stdout go to file
  dup2(fd, 2);    // make stderr go to file
  close(fd);      // fd no longer needed
  #endif
}

void Communicator::launch_exec(const std::string exec)
{
  printf("About to exec %s.... \n",exec.c_str());
  const int res = execlp(exec.c_str(),
                         exec.c_str(),
                         std::to_string(socket_id).c_str(),
                         NULL);
  //int res = execvp(*largv, largv);
  if (res < 0)
    fprintf(stderr,"Unable to exec file '%s'!\n", exec.c_str());
}

void Communicator::launch_app()
{
  #ifdef __Smarties_
  mkdir(("simulation_"+std::to_string(socket_id)+"_"
                      +std::to_string(iter)+"/").c_str(),
                      S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  chdir(("simulation_"+std::to_string(socket_id)+"_"
                      +std::to_string(iter)+"/").c_str());

  redirect_stdout_stderr();

  launch_exec(execpath);
  #else
  printf("launch_app\n");
  fflush(0);
  #endif
  abort(); //if app returns: TODO
}

void Communicator::setupClient()
{
  const int rf = fork();
  if (rf == 0) {  //child spawns server
    if (execpath == std::string())
      launch_smarties();
    else
      launch_app();
  } else {  //parent
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
    const int servlen = sizeof(serverAddress.sun_family)
                       + strlen(serverAddress.sun_path);

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
  }
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
  fflush(0);
  if ((ServerSocket = socket(AF_UNIX, SOCK_STREAM, 0)) == -1)
  {
      perror("socket");
      exit(1);
  }
  unlink(SOCK_PATH);

  bzero(&serverAddress, sizeof(serverAddress));
  serverAddress.sun_family = AF_UNIX;
  strcpy(serverAddress.sun_path, SOCK_PATH);
  const int servlen = sizeof(serverAddress.sun_family)
                    + strlen(serverAddress.sun_path);

  if (bind(ServerSocket, (struct sockaddr *)&serverAddress, servlen) < 0)
  {
      perror("bind");
      exit(1);
  }

  int _true = 1;
  if(setsockopt(ServerSocket, SOL_SOCKET, SO_REUSEADDR, &_true, sizeof(int))<0)
  {
      perror("Sockopt failed\n");
      exit(1);
  }

  /* listen (only 1)*/
  if (listen(ServerSocket, 1) == -1)
  {
      perror("listen");
      exit(1);
  }

  unsigned int addr_len = sizeof(clientAddress);
  if((Socket=accept(ServerSocket,(struct sockaddr*)&clientAddress,&addr_len))==-1)
  {
      perror("accept");
      return;
  }
  else printf("selectserver: new connection from on socket %d\n", Socket);
  fflush(0);
}

void Communicator::launch()
{
  if (spawner) setupClient();
  else setupServer();
}

//forked comm, single node job, constructor by app
//only rule: if app gets socket=0, then it is the spawner
Communicator::Communicator(const int socket, const int sdim, const int adim,
                                    const std::string log, const int verb) :
  #ifdef MPI_INCLUDED
  appComm( MPI_COMM_NULL ), masterComm( MPI_COMM_NULL ),
  #endif
  sizeout((3+sdim)*sizeof(double)), sizein(adim*sizeof(double)),
  nActions(adim), nStates(sdim), spawner(socket==0), app_rank(0),
  smarties_rank(0), app_size(0), smarties_size(0), verbose(verb),
  dataout(_alloc(sizeout)), datain(_alloc(sizein)), execpath(std::string()),
  paramfile(std::string()), logfile(log), msg_id(0), iter(0), socket_id(socket)
{
  if (spawner) //cheap way to ensure multiple sockets can exist on same node
  {
    struct timeval clock;
    gettimeofday(&clock, NULL);
    socket_id = abs(clock.tv_usec % std::numeric_limits<int>::max());
  }
  sprintf(SOCK_PATH, "%s%d", "/tmp/smarties_sock_", socket_id);

  printf("App seq: nS:%d nA:%d sin:%d sout:%d spawn:%d, socket:%d PATH=%s\n",
        nStates, nActions, sizein, sizeout, spawner, socket_id, SOCK_PATH);

  launch();
}

//forked comm, mpi job, constructor by app
//only rule: if app gets socket=0, then it is the spawner
#ifdef MPI_INCLUDED
Communicator::Communicator(const int socket, const int sdim,const int adim,
                const MPI_Comm app, const std::string log, const int verb) :
  appComm(app), masterComm(MPI_COMM_NULL),
  sizeout((3+sdim)*sizeof(double)), sizein(adim*sizeof(double)),
  nActions(adim), nStates(sdim), spawner(socket==0), app_rank(getRank(app)),
  smarties_rank(0), app_size(getSize(app)), smarties_size(0), verbose(verb),
  dataout(_alloc(sizeout)), datain(_alloc(sizein)), execpath(std::string()),
  paramfile(std::string()), logfile(log), msg_id(0), iter(0), socket_id(socket)
{
  if (app_rank) return;

  if (spawner) //cheap way to ensure multiple sockets can exist on same node
  {
    struct timeval clock;
    gettimeofday(&clock, NULL);
    socket_id = abs(clock.tv_usec % std::numeric_limits<int>::max());
  }
  sprintf(SOCK_PATH, "%s%d", "/tmp/smarties_sock_", socket_id);

  printf("App mpi: nS:%d nA:%d sin:%d sout:%d spawn:%d, socket:%d PATH=%s\n",
        nStates, nActions, sizein, sizeout, spawner, socket_id, SOCK_PATH);

  launch();
}
#endif

Communicator::~Communicator()
{
    if (!smarties_rank) {
      if (spawner)
        close(ServerSocket);
      else
        close(Socket);
    } //if with forked process paradigm

    free(datain);
    free(dataout);
}

#ifdef __Smarties_
extern int app_main(Communicator*const rlcom, MPI_Comm mpicom, int argc, char**argv);

int Communicator::recvStateFromApp()
{
    int bytes = recv_all(Socket, dataout, sizeout);

    if (bytes <= 0)
    {
        if (bytes == 0) printf("socket %d hung up\n", Socket);
        else perror("(1) recv");
        close(Socket);
        close(ServerSocket);
        intToDoublePtr(0, dataout+0);
        intToDoublePtr(_AGENT_FAILCOMM, dataout+1);
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

void Communicator::answerTerminateReq(const double answer)
{
  datain[0] = answer;
  sendActionToApp();
}

void Communicator::unpackAction(std::vector<double>& action)
{
    assert(action.size() == (std::size_t) nActions);
    std::ostringstream o;
    o <<"Send: ";
    for (int j=0; j<nActions; j++) {
        action[j] = datain[j];
        o << action[j] << " ";
        assert(not std::isnan(action[j]));
        assert(not std::isinf(action[j]));
    }
    //std::cout << o.str() << std::endl;
    if (logfile == std::string()) return;
    if(verbose) printLog(o.str());
    else  printBuf(datain, sizein);
}

void Communicator::unpackState(int& iAgent, _AGENT_STATUS& status,
                             std::vector<double>& state, double& reward)
{
  assert(state.size() == (std::size_t) nStates);
  std::ostringstream o;

  iAgent = doublePtrToInt(dataout+0);
  status = doublePtrToInt(dataout+1);
  o <<"Recv: "<<iAgent<<" "<<msg_id++<<" "<<status<<" ";
  {
    for (int j=0; j<nStates; j++) {
        state[j] = dataout[j+2];
        o << state[j] << " ";
        assert(not std::isnan(state[j]));
        assert(not std::isinf(state[j]));
    }
    reward = dataout[nStates+2];
    o << reward;
    assert(not std::isnan(reward));
    assert(not std::isinf(reward));
  }

  if (logfile != std::string()) {
    if(verbose) printLog(o.str());
    else  printBuf(dataout, sizeout);
  }
  //std::cout << o.str() << std::endl;
}

void Communicator::restart(std::string fname)
{
    int wrank = getRank(MPI_COMM_WORLD);
    FILE * f = fopen(("comm_"+to_string(wrank)+".status").c_str(), "r");
    if (f == NULL) return;
    {
      int ret = -1;
      fscanf(f, "sim number: %d\n", &ret);
      if(ret>=0) iter = ret;
      printf("sim number: %d\n", iter);
    }
    {
      int ret = -1;
      fscanf(f, "message ID: %d\n", &ret);
      if(ret>=0) msg_id = ret;
      printf("message ID: %d\n", msg_id);
    }
    {
      int ret = -1;
      fscanf(f, "socket  ID: %d\n", &ret);
      if(ret>=0) socket_id = ret;
      printf("socket  ID: %d\n", socket_id);
    }
    fclose(f);
}

void Communicator::save() const
{
    int wrank = getRank(MPI_COMM_WORLD);
    FILE * f = fopen(("comm_"+to_string(wrank)+".status").c_str(), "w");
    if (f != NULL)
    {
      fprintf(f, "sim number: %d\n", iter);
      fprintf(f, "message ID: %d\n", msg_id);
      fprintf(f, "socket  ID: %d\n", socket_id);
      fclose(f);
    }
    //printf( "sim number: %d\n", env->iter);
}

//comm directly with master, constructor by smarties, pass this to app
Communicator::Communicator(const int sdim, const int adim, const MPI_Comm scom,
  const MPI_Comm acom, const std::string exec, const std::string params,
  const std::string log, const int verb):
appComm(acom), masterComm(scom),
sizeout((3+sdim)*sizeof(double)), sizein(adim*sizeof(double)),
nActions(adim), nStates(sdim), spawner(0),
app_rank(getRank(acom)), smarties_rank(getRank(scom)),
app_size(getSize(acom)), smarties_size(getSize(scom)),
verbose(verb), dataout(_alloc(sizeout)), datain(_alloc(sizein)),
execpath(exec), paramfile(params), logfile(log), msg_id(0), iter(0), socket_id(0)
{
  printf("Smarties: nS:%d nA:%d sin:%d sout:%d\n",
        nStates, nActions, sizein, sizeout);
}

//construtor by smarties side of fork
Communicator::Communicator(const int socket, const int sdim, const int adim,
const bool spawn, const std::string exe, const MPI_Comm scom,
const std::string log, const int verb):
appComm(MPI_COMM_NULL), masterComm(scom),
sizeout((3+sdim)*sizeof(double)), sizein(adim*sizeof(double)),
nActions(adim), nStates(sdim), spawner(spawn), app_rank(0),
smarties_rank(getRank(scom)), app_size(0), smarties_size(getSize(scom)),
verbose(verb), dataout(_alloc(sizeout)), datain(_alloc(sizein)), execpath(exe),
paramfile(std::string()), logfile(log), msg_id(0), iter(0), socket_id(socket)
{
  sprintf(SOCK_PATH, "%s%d", "/tmp/smarties_sock_", socket_id);
  printf("Smarties: nS:%d nA:%d sin:%d sout:%d spawn:%d, socket:%d PATH=%s\n",
        nStates, nActions, sizein, sizeout, spawner, socket_id, SOCK_PATH);
}

void Communicator::ext_app_run()
{
	char *largv[256];
	char line[1024];
  int largc = jobs_init(line, largv);

	char initd[256], newd[256];
  getcwd(initd,256);
	sprintf(newd,"%s/%s_%d_%d",initd,"simulation",getRank(MPI_COMM_WORLD),iter);
  mkdir(newd, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	chdir(newd);	// go to the task private directory
	//redirect_stdout_init();
	app_main(this, appComm, largc, largv);
	//redirect_stdout_finalize();

	chdir(initd);	// go up one level
}

int Communicator::jobs_init(char *line, char **largv)
{
	FILE * cmdfp = fopen(paramfile.c_str(), "r");

  if (cmdfp == NULL)
    die("Missing %s\n", execpath.c_str());
	if(fgets(line, 1024, cmdfp)== NULL)
    die("Empty %s\n",   execpath.c_str());
	if (strstr(line,      execpath.c_str()) == NULL)
    die("Invalid %s\n", execpath.c_str());

	fclose(cmdfp);

	return parse2(line, largv);
}

void Communicator::redirect_stdout_init()
{
	fflush(stdout);
	fgetpos(stdout, &pos);
	fd = dup(fileno(stdout));
	char buf[500];
	sprintf(buf, "output_%d_%d",getRank(MPI_COMM_WORLD),iter);
	freopen(buf, "w", stdout);
}

void Communicator::redirect_stdout_finalize()
{
	dup2(fd, fileno(stdout));
	close(fd);
	clearerr(stdout);
	fsetpos(stdout, &pos);        /* for C9X */
}
#endif

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

static int parse2(char *line, char **argv)
{
	int argc = 0;

	while (*line != '\0') {         /* if not the end of line ....... */
		while (*line == ' ' || *line == '\t' || *line == '\n')
			*line++ = '\0';         /* replace white spaces with 0 */
		*argv++ = line;         /* save the argument position */

		if (*line != '\0' && *line != ' ' && *line != '\t' && *line != '\n')
			argc++;

		while (*line != '\0' && *line != ' ' &&
			*line != '\t' && *line != '\n')
			line++; /* skip the argument until ...*/
	}
	*argv = '\0';   /* mark the end of argument list */

	return argc;
}
