#include "Communicator.h"
#include "Communicator_utils.cpp"
#define COMM_REDIRECT_OUT
//APPLICATION SIDE CONSTRUCTOR
Communicator::Communicator(const int socket, const int state_components, const int action_components, const int number_of_agents) : gen(std::mt19937(socket))
{
  if(socket<0) {
    printf("FATAL: Communicator created with socket < 0.\n");
    abort();
  }
  if(state_components<=0) {
    printf("FATAL: Cannot set negative state space dimensionality.\n");
    abort();
  }
  if(action_components<=0) {
    printf("FATAL: Cannot set negative number of action degrees of freedom.\n");
    abort();
  }
  if(number_of_agents<=0) {
    printf("FATAL: Cannot set negative number of agents.\n");
    abort();
  }
  assert(state_components>0 && action_components>0 && number_of_agents>0);
  nAgents = number_of_agents;
  update_state_action_dims(state_components, action_components);
  spawner = socket==0; // if app gets socket prefix 0, then it spawns smarties
  socket_id = socket;
  called_by_app = true;
  launch();
}

void Communicator::set_action_scales(const std::vector<double> upper,
  const std::vector<double> lower, const bool bound)
{
  assert(!sentStateActionShape);
  assert(discrete_actions == 0);
  assert(upper.size() == (size_t)nActions && lower.size() == (size_t)nActions);
  for (int i=0; i<nActions; i++) action_bounds[2*i+0] = upper[i];
  for (int i=0; i<nActions; i++) action_bounds[2*i+1] = lower[i];
  for (int i=0; i<nActions; i++) action_options[2*i+0] = 2.1;
  for (int i=0; i<nActions; i++) action_options[2*i+1] = bound ? 1.1 : 0;
}

void Communicator::set_action_options(const std::vector<int> action_option_num)
{
  assert(!sentStateActionShape);
  discrete_actions = 1;
  assert(action_option_num.size() == (size_t)nActions);
  discrete_action_values = 0;
  for (int i=0; i<nActions; i++) {
    discrete_action_values += action_option_num[i];
    action_options[2*i+0] = action_option_num[i];
    action_options[2*i+1] = 1.1;
  }

  action_bounds.resize(discrete_action_values);
  for(int i=0, k=0; i<nActions; i++)
    for(int j=0; j<action_option_num[i]; j++)
      action_bounds[k++] = j;
}

void Communicator::set_action_options(const int action_option_num)
{
  assert(!sentStateActionShape);
  if(nActions != 1) {
    printf("FATAL: Communicator::set_action_options perceived more than 1 action degree of freedom, but only one number of actions provided.\n");
    abort();
  }
  assert(1 == nActions);
  discrete_actions = 1;
  discrete_action_values = action_option_num;
  action_options[0] = action_option_num;
  action_options[1] = 1.1;

  action_bounds.resize(action_option_num);
  for(int j=0; j<action_option_num; j++) action_bounds[j] = j;
}

void Communicator::set_state_scales(const std::vector<double> upper,
  const std::vector<double> lower)
{
  assert(!sentStateActionShape);
  assert(upper.size() == (size_t)nStates && lower.size() == (size_t)nStates);
  for (int i=0; i<nStates; i++) obs_bounds[2*i+0] = upper[i];
  for (int i=0; i<nStates; i++) obs_bounds[2*i+1] = lower[i];
}

void Communicator::set_state_observable(const std::vector<bool> observable)
{
  assert(!sentStateActionShape);
  assert(observable.size() == (size_t) nStates);
  for (int i=0; i<nStates; i++) obs_inuse[i] = observable[i];
}

void Communicator::sendStateActionShape()
{
  if(sentStateActionShape) return;
  assert(obs_inuse.size() == (size_t) nStates);
  assert(obs_bounds.size() == (size_t) nStates*2);
  assert(action_bounds.size() == (size_t) nActions*2);
  assert(action_options.size() == (size_t) discrete_action_values);
  double sizes[4] = {nStates+.1, nActions+.1, discrete_actions+.1, nAgents+.1};
  comm_sock(Socket, true, sizes, 4 *sizeof(double));
  comm_sock(Socket, true, obs_inuse.data(),      nStates *1*sizeof(double));
  comm_sock(Socket, true, obs_bounds.data(),     nStates *2*sizeof(double));
  comm_sock(Socket, true, action_options.data(), nActions*2*sizeof(double));
  comm_sock(Socket, true, action_bounds.data(),  discrete_action_values*8 );
  comm_sock(Socket, false, &dump_value, sizeof(double));
  sentStateActionShape = true;
}

void Communicator::update_state_action_dims(const int sdim, const int adim)
{
  nStates = sdim;
  nActions = adim;
  discrete_action_values = 2*adim;
  obs_inuse      = std::vector<double>(1*sdim, 1);
  obs_bounds     = std::vector<double>(2*sdim, 0);
  action_options = std::vector<double>(2*adim, 0);
  action_bounds  = std::vector<double>(2*adim, 0);
  for (int i = 0; i<2*sdim; i++) obs_bounds[i]     = i%2 == 0 ? 1   : -1;
  for (int i = 0; i<2*adim; i++) action_options[i] = i%2 == 0 ? 2.1 :  0;
  for (int i = 0; i<2*adim; i++) action_bounds[i]  = i%2 == 0 ? 1   : -1;
  // agent number, initial/normal/terminal indicator, state,  reward
  size_state = (3+sdim)*sizeof(double);
  size_action = adim*sizeof(double);
  _dealloc(data_action);
  _dealloc(data_state);
  data_action = _alloc(size_action);
  data_state = _alloc(size_state);
}

#ifdef MPI_INCLUDED
//MPI APPLICATION SIDE CONSTRUCTOR
Communicator::Communicator(const int socket, const int state_components, const int action_components, const MPI_Comm app, const int number_of_agents)
{
  if(socket<0) {
    printf("FATAL: Communicator created with socket < 0.\n");
    abort();
  }
  if(state_components<=0) {
    printf("FATAL: Cannot set negative state space dimensionality.\n");
    abort();
  }
  if(action_components<=0) {
    printf("FATAL: Cannot set negative number of action degrees of freedom.\n");
    abort();
  }
  if(number_of_agents<=0) {
    printf("FATAL: Cannot set negative number of agents.\n");
    abort();
  }
  assert(state_components>0 && action_components>0 && number_of_agents>0);
  nAgents = number_of_agents;
  comm_inside_app = app;
  update_rank_size();
  update_state_action_dims(state_components, action_components);
  spawner = socket==0; // if app gets socket prefix 0, then it spawns smarties
  socket_id = socket;
  called_by_app = true;
  if (rank_inside_app == 0) //only rank 0 of the app talks with smarties
    launch();
}
#endif

void Communicator::sendState(const int iAgent, const envInfo status,
    const std::vector<double> state, const double reward)
{
  if(rank_inside_app>0) return; //only rank 0 of the app sends state
  if(!sentStateActionShape) sendStateActionShape();
  assert(state.size()==(std::size_t)nStates && data_state not_eq nullptr);
  assert(iAgent>=0 && iAgent<nAgents);

  intToDoublePtr(iAgent, data_state+0);
  intToDoublePtr(status, data_state+1);
  for (int j=0; j<nStates; j++) {
    data_state[j+2] = state[j];
    assert(not std::isnan(state[j]) && not std::isinf(state[j]));
  }
  data_state[nStates+2] = reward;
  assert(not std::isnan(reward) && not std::isinf(reward));

  if (logfile != std::string()) {
    if(verbose) printLog(data_state, size_state);
    else  printBuf(data_state, size_state);
  }

  #ifdef MPI_INCLUDED
    if (rank_learn_pool>0) send_MPI(data_state, size_state, comm_learn_pool);
    else
  #endif
    comm_sock(Socket, true, data_state, size_state);

  if (status == TERM_COMM) { seq_id++; msg_id = 0; }
}

void Communicator::recvAction(std::vector<double>& actions)
{
  assert(actions.size() == (std::size_t) nActions);
  #ifdef MPI_INCLUDED
    if(rank_inside_app <= 0)
    {
      if (rank_learn_pool>0)
        recv_MPI(data_action, size_action, comm_learn_pool, lag);
      else
  #endif
        comm_sock(Socket, false, data_action, size_action);
  #ifdef MPI_INCLUDED
      for (int i=1; i<size_inside_app; ++i)
        MPI_Send(data_action, size_action, MPI_BYTE, i, 42, comm_inside_app);
    } else {
      MPI_Recv(data_action, size_action, MPI_BYTE, 0, 42, comm_inside_app, MPI_STATUS_IGNORE);
    }
  #endif

  for (int j=0; j<nActions; j++) {
    actions[j] = data_action[j];
    assert(not std::isnan(actions[j]) && not std::isinf(actions[j]));
  }

  if(fabs(data_action[0]-_AGENT_KILLSIGNAL)<2.2e-16) abort();

  if (logfile != std::string() && rank_inside_app <= 0) {
    if(verbose) printLog(data_action, size_action);
    else  printBuf(data_action, size_action);
  }
}

void Communicator::sendCompleteTermination()
{
  if(rank_inside_app>0) return;
  intToDoublePtr(0, data_state+0);
  intToDoublePtr(GAME_OVER, data_state+1);
  #ifdef MPI_INCLUDED
    if (rank_learn_pool>0) send_MPI(data_state, size_state, comm_learn_pool);
    else
  #endif
    comm_sock(Socket, true, data_state, size_state);
}

void Communicator::printLog(const double*const buf, const int size)
{
  std::ostringstream o;
  o << "seq_id:" << seq_id << " msg_id:"  << msg_id << " ";
  for (unsigned j=0; j<size/sizeof(double); j++) o << buf[j] << " ";
  const std::string fname = logfile+std::to_string(iter)+".txt";
  FILE * f = fopen(fname.c_str(), "a");
  if (f != NULL)
    fprintf(f, "%s\n", o.str().c_str());
  fclose(f);
}

void Communicator::printBuf(const double*const buf, const int size)
{
  const std::string fname = logfile+std::to_string(iter)+".raw";
  FILE * pFile = fopen (fname.c_str(), "ab");
  fwrite (buf, sizeof(double), size/sizeof(double), pFile);
  fclose (pFile);
}

void Communicator::launch_smarties()
{
  abort();
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

void Communicator::launch()
{
  assert(rank_inside_app<1);
  if (spawner && called_by_app)
  { //cheap way to ensure multiple sockets can exist on same node
    struct timeval clock;
    gettimeofday(&clock, NULL);
    socket_id = abs(clock.tv_usec % std::numeric_limits<int>::max());
  }
  sprintf(SOCK_PATH, "%s%d", "/tmp/smarties_sock", socket_id);

  if (spawner) setupClient();
  else setupServer();
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

    launch_exec("../"+execpath);
  #else
    printf("launch_app\n");
    fflush(0);
  #endif
  abort(); //if app returns: TODO
}

void Communicator::setupClient()
{
  unlink(SOCK_PATH);
  print();
  fflush(0);
  const int rf = fork();

  if (rf == 0) {  //child spawns server
    if (execpath == std::string())
      launch_smarties();
    else
      launch_app();
  } else {  //parent
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
                      + strlen(serverAddress.sun_path)+1;

    /* Connect to the server */
    while (connect(Socket, (struct sockaddr *)&serverAddress, servlen) < 0)
      usleep(1);

    printf("Connected to server\n");
    fflush(0);
  }
}

void Communicator::setupServer()
{
  if ((ServerSocket = socket(AF_UNIX, SOCK_STREAM, 0)) == -1)
  {
    perror("socket");
    exit(1);
  }

  bzero(&serverAddress, sizeof(serverAddress));
  serverAddress.sun_family = AF_UNIX;
  strcpy(serverAddress.sun_path, SOCK_PATH);
  printf("%s %s\n",serverAddress.sun_path,SOCK_PATH);
  fflush(0);
  const int servlen = sizeof(serverAddress.sun_family)
                    + strlen(serverAddress.sun_path) +1;

  if (bind(ServerSocket, (struct sockaddr *)&serverAddress, servlen) < 0)
  {
    perror("bind");
    exit(1);
  }
  /*
  int _true = 1;
  if(setsockopt(ServerSocket, SOL_SOCKET, SO_REUSEADDR, &_true, sizeof(int))<0)
  {
    perror("Sockopt failed\n");
    exit(1);
  }
   */
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

Communicator::~Communicator()
{
  #ifdef __Smarties_
  if (rank_learn_pool>0) {
    data_action[0] = _AGENT_KILLSIGNAL;
    send_all(Socket, data_action, size_action);
  }
  #endif
  if (rank_learn_pool>0) {
    if (spawner) close(Socket);
    else   close(ServerSocket);
  } //if with forked process paradigm

  if(data_state not_eq nullptr) free(data_state);
  if(data_action not_eq nullptr) free(data_action);
}

#ifdef __Smarties_


void Communicator::getStateActionShape()
{
  double sizes[4] = {0, 0, 0, 0};
  if (rank_learn_pool==0)
    MPI_Recv(sizes, 32, MPI_BYTE, 1, 3, comm_learn_pool, MPI_STATUS_IGNORE);
  else {
    comm_sock(Socket, false, sizes, 4*sizeof(double));
    if(rank_learn_pool==1) MPI_Ssend(sizes,32, MPI_BYTE, 0,3, comm_learn_pool);
  }

  nStates = doublePtrToInt(sizes+0); nActions = doublePtrToInt(sizes+1);
  discrete_actions = doublePtrToInt(sizes+2);
  nAgents = doublePtrToInt(sizes+3);
  //printf("Discrete? %d\n",discrete_actions);
  assert(nStates>=0 && nActions>=0);
  update_state_action_dims(nStates, nActions);

  if (rank_learn_pool==0) {
    MPI_Recv(obs_inuse.data(), nStates*8, MPI_BYTE, 1, 3, comm_learn_pool, MPI_STATUS_IGNORE);
    MPI_Recv(obs_bounds.data(), nStates*16, MPI_BYTE, 1, 4, comm_learn_pool, MPI_STATUS_IGNORE);
    MPI_Recv(action_options.data(), nActions*16, MPI_BYTE, 1, 5, comm_learn_pool, MPI_STATUS_IGNORE);
  } else {
    comm_sock(Socket, false, obs_inuse.data(), nStates*8);
    comm_sock(Socket, false, obs_bounds.data(), nStates*16);
    comm_sock(Socket, false, action_options.data(), nActions*16);
    if (rank_learn_pool==1) {
      MPI_Ssend(obs_inuse.data(), nStates*8, MPI_BYTE, 0, 3, comm_learn_pool);
      MPI_Ssend(obs_bounds.data(), nStates*16, MPI_BYTE, 0, 4, comm_learn_pool);
      MPI_Ssend(action_options.data(), nActions*16, MPI_BYTE, 0, 5, comm_learn_pool);
    }
  }

  int n_vals = 0;
  for(int i=0; i<nActions; i++) n_vals += action_options[i*2];
  discrete_action_values = n_vals;

  action_bounds.resize(n_vals);
  if (rank_learn_pool==0)
    MPI_Recv(action_bounds.data(), n_vals*8, MPI_BYTE, 1, 6, comm_learn_pool, MPI_STATUS_IGNORE);
  else {
    comm_sock(Socket, false, action_bounds.data(), n_vals*8);
    comm_sock(Socket, true, &dump_value, 8);
    if (rank_learn_pool==1)
      MPI_Ssend(action_bounds.data(),n_vals*8, MPI_BYTE, 0,6, comm_learn_pool);
  }
}

Communicator::Communicator(const MPI_Comm scom, const int socket, const bool spawn)
{
  spawner = spawn;
  socket_id = socket;
  comm_learn_pool = scom;
  sentStateActionShape = true; //to avoid mpi apps sending redundant info
  update_rank_size();
}

extern int app_main(Communicator*const rlcom, MPI_Comm mpicom, int argc, char**argv);

int Communicator::recvStateFromApp()
{
  int bytes = recv_all(Socket, data_state, size_state);

  if (bytes <= 0)
  {
    if (bytes == 0) printf("socket %d hung up\n", Socket);
    else perror("(1) recv");
    close(Socket);

    intToDoublePtr(0, data_state+0);
    intToDoublePtr(FAIL_COMM, data_state+1);
    iter++;
  }
  else assert(bytes == size_state);

  if(comm_learn_pool != MPI_COMM_NULL)
    send_MPI(data_state, size_state, comm_learn_pool);

  return bytes <= 0;
}

int Communicator::sendActionToApp()
{
  //printf("I think im sending action %f\n",data_action[0]);
  if(comm_learn_pool != MPI_COMM_NULL)
    recv_MPI(data_action, size_action, comm_learn_pool, lag);

  send_all(Socket, data_action, size_action);
  if(fabs(data_action[0]-_AGENT_KILLSIGNAL)<2.2e-16) return 1;

  return 0;
}

void Communicator::answerTerminateReq(const double answer)
{
  data_action[0] = answer;
   //printf("I think im givign the goahead %f\n",data_action[0]);
  send_all(Socket, data_action, size_action);
}

void Communicator::restart(std::string fname)
{
  int wrank = getRank(MPI_COMM_WORLD);
  FILE * f = fopen(("comm_"+to_string(wrank)+".status").c_str(), "r");
  if (f == NULL) return;
  {
    long unsigned ret = 0;
    fscanf(f, "sim number: %lu\n", &ret);
    iter = ret;
    printf("sim number: %lu\n", iter);
  }
  {
    long unsigned ret = 0;
    fscanf(f, "message ID: %lu\n", &ret);
    msg_id = ret;
    printf("message ID: %lu\n", msg_id);
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
    fprintf(f, "sim number: %lu\n", iter);
    fprintf(f, "message ID: %lu\n", msg_id);
    fprintf(f, "socket  ID: %d\n", socket_id);
    fclose(f);
  }
  //printf( "sim number: %d\n", env->iter);
}

void Communicator::ext_app_run()
{
  char *largv[256];
  char line[1024];
  int largc = jobs_init(line, largv);

  char initd[256], newd[1024];
  getcwd(initd,256);
  assert(slaveGroup>=0 && rank_inside_app >= 0 && comm_inside_app != MPI_COMM_NULL);
  sprintf(newd,"%s/%s_%d_%lu",initd,"simulation",slaveGroup,iter);
  if(rank_inside_app==0) mkdir(newd, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  MPI_Barrier(comm_inside_app);
  chdir(newd);  // go to the task private directory

  //copy any additional file
  if (rank_inside_app==0)
    if (copy_from_dir("../bin") !=0 )  {
      printf("Error in copy from dir\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  MPI_Barrier(comm_inside_app);

  redirect_stdout_init();
  app_main(this, comm_inside_app, largc, largv);
  redirect_stdout_finalize();

  chdir(initd);  // go up one level
  iter++;
}

int Communicator::jobs_init(char *line, char **largv)
{
  FILE * cmdfp = fopen(paramfile.c_str(), "r");

  if (cmdfp == NULL)
    _die("Missing %s\n", paramfile.c_str());
  if(fgets(line, 1024, cmdfp)== NULL)
    _die("Empty %s\n",   paramfile.c_str());
  //if (strstr(line,       paramfile.c_str()) == NULL)
  //  _die("Invalid %s %s\n", paramfile.c_str(), line);

  fclose(cmdfp);

  return parse2(line, largv);
}

void Communicator::redirect_stdout_init()
{
  fflush(stdout);
  fgetpos(stdout, &pos);
  fd = dup(fileno(stdout));
  char buf[500];
  int wrank = getRank(MPI_COMM_WORLD);
  sprintf(buf, "output_%d_%lu",wrank,iter);
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

void Communicator::redirect_stdout_stderr()
{
  #ifdef COMM_REDIRECT_OUT
    fflush(0);
    char output[256];
    sprintf(output, "output");
    fd = open(output, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    dup2(fd, 1);    // make stdout go to file
    dup2(fd, 2);    // make stderr go to file
    close(fd);      // fd no longer needed
  #endif
}
