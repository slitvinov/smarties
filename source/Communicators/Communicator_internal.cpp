//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Communicator_internal.h"

extern int app_main(Communicator*const rlcom, MPI_Comm mpicom, int argc, char**argv);

Communicator_internal::Communicator_internal(const MPI_Comm scom, const int socket, const bool spawn) : Communicator(socket, spawn)
{
  comm_learn_pool = scom;
  update_rank_size();
}

Communicator_internal::~Communicator_internal()
{
  if (rank_learn_pool>0) {
    data_action[0] = _AGENT_KILLSIGNAL;
    send_all(Socket, data_action, size_action);
  }
}

void Communicator_internal::launch_forked()
{
  assert(not called_by_app);
  mkdir(("simulation_"+std::to_string(socket_id)+"_"
      +std::to_string(iter)+"/").c_str(),
      S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  chdir(("simulation_"+std::to_string(socket_id)+"_"
      +std::to_string(iter)+"/").c_str());

  fd = redirect_stdout_stderr();
  launch_exec("../"+execpath, socket_id);
}

int Communicator_internal::recvStateFromApp()
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

  if(comm_learn_pool != MPI_COMM_NULL) workerSend_MPI();

  return bytes <= 0;
}

int Communicator_internal::sendActionToApp()
{
  //printf("I think im sending action %f\n",data_action[0]);
  if(comm_learn_pool != MPI_COMM_NULL) workerRecv_MPI();

  bool endSignal = fabs(data_action[0]-_AGENT_KILLSIGNAL)<2.2e-16;

  send_all(Socket, data_action, size_action);

  return endSignal;
}

void Communicator_internal::answerTerminateReq(const double answer)
{
  data_action[0] = answer;
   //printf("I think im givign the goahead %f\n",data_action[0]);
  send_all(Socket, data_action, size_action);
}

void Communicator_internal::restart(std::string fname)
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

void Communicator_internal::save() const
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

void Communicator_internal::ext_app_run()
{
  char *largv[256];
  char line[1024];
  int largc = jobs_init(line, largv);

  char initd[256], newd[1024];
  getcwd(initd,256);
  assert(workerGroup>=0 && rank_inside_app >= 0 && comm_inside_app != MPI_COMM_NULL);

  while(1)
  {
    sprintf(newd,"%s/%s_%d_%lu", initd, "simulation", workerGroup, iter);

    if(rank_inside_app==0) // app's root sets up working dir
      mkdir(newd, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    MPI_Barrier(comm_inside_app);
    chdir(newd);  // go to the task private directory


    if (rank_inside_app==0 && setupfolder != "") //copy any additional file
      if (copy_from_dir(setupfolder.c_str()) !=0 )
        _die("Error in copy from dir %s\n", setupfolder.c_str());

    MPI_Barrier(comm_inside_app);

    redirect_stdout_init();
    // app only needs lower level functionalities:
    // ie. send state, recv action, specify state/action spaces properties...
    Communicator* commptr = static_cast<Communicator*>(this);
    app_main(commptr, comm_inside_app, largc, largv);
    redirect_stdout_finalize();

    chdir(initd);  // go up one level
    iter++;
  }
}

int Communicator_internal::jobs_init(char *line, char **largv)
{
  if (paramfile == "") return 0;

  FILE * cmdfp = fopen(paramfile.c_str(), "r");

  if (cmdfp == NULL)
    _die("Missing %s\n", paramfile.c_str());
  if(fgets(line, 1024, cmdfp)== NULL)
    _die("Empty %s\n",   paramfile.c_str());

  fclose(cmdfp);

  return parse2(line, largv);
}

void Communicator_internal::redirect_stdout_init()
{
  fflush(stdout);
  fgetpos(stdout, &pos);
  fd = dup(fileno(stdout));
  char buf[500];
  int wrank = getRank(MPI_COMM_WORLD);
  sprintf(buf, "output_%d_%lu",wrank,iter);
  freopen(buf, "w", stdout);
}

void Communicator_internal::redirect_stdout_finalize()
{
  dup2(fd, fileno(stdout));
  close(fd);
  clearerr(stdout);
  fsetpos(stdout, &pos);        /* for C9X */
}

void Communicator_internal::getStateActionShape()
{
  if(sentStateActionShape) die("undefined behavior");

  double sizes[4] = {0, 0, 0, 0};
  if (rank_learn_pool==0)
    MPI_Recv(sizes, 32, MPI_BYTE, 1, 3, comm_learn_pool, MPI_STATUS_IGNORE);
  else {
    comm_sock(Socket, false, sizes, 4*sizeof(double));
    if(rank_learn_pool==1) MPI_Ssend(sizes,32, MPI_BYTE, 0,3, comm_learn_pool);
  }

  nStates          = doublePtrToInt(sizes+0);
  nActions         = doublePtrToInt(sizes+1);
  discrete_actions = doublePtrToInt(sizes+2);
  nAgents          = doublePtrToInt(sizes+3);
  //printf("Discrete? %d\n",discrete_actions);
  assert(nStates>=0 && nActions>=0);
  update_state_action_dims(nStates, nActions);
  sentStateActionShape = true;

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
    if (rank_learn_pool==1)
      MPI_Ssend(action_bounds.data(),n_vals*8, MPI_BYTE, 0,6, comm_learn_pool);
  }

  print();
}
