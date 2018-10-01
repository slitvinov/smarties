//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learners/AllLearners.h"
#include "Utils/Scheduler.h"
#include "Utils/ObjectFactory.h"
using namespace std;

void runClient();
void runWorker(Settings& S);
void runMaster(Settings& S);

void runWorker(Settings& S)
{
  assert(S.workers_rank and S.workers_size>0);
  ObjectFactory factory(S);
  Environment* env = factory.createEnvironment();
  Communicator_internal comm = env->create_communicator();

  #ifndef NDEBUG
    int cpu_num; GETCPU(cpu_num); //sched_getcpu()
    printf("Worker Rank %d is running on CPU %3d\n", S.world_rank, cpu_num);
  #endif

  Worker simulation(&comm, env, S);
  simulation.run();
}

void runMaster(Settings& S)
{
  S.check();

  #ifdef INTERNALAPP //unblock creation of app comm if needed
    MPI_Comm tmp_com;
    MPI_Comm_split(S.workersComm, MPI_UNDEFINED, 0, &tmp_com);
  #endif

  ObjectFactory factory(S);
  Environment*const env = factory.createEnvironment();
  Communicator_internal comm = env->create_communicator();

  S.finalizeSeeds(); // now i know nAgents, might need more generators

  const Uint nPols = S.bSharedPol ? 1 : env->nAgentsPerRank;
  vector<Learner*> learners(nPols, nullptr);
  for(Uint i = 0; i<nPols; i++) {
    stringstream ss; ss<<"agent_"<<std::setw(2)<<std::setfill('0')<<i;
    cout << "Learner: " << ss.str() << endl;
    learners[i] = createLearner(env, S);
    learners[i]->setLearnerName(ss.str() +"_", i);
    learners[i]->restart();
  }

  #ifndef NDEBUG
    #pragma omp parallel
    {
      int cpu_num; GETCPU(cpu_num); //sched_getcpu()
      printf("Master Rank %d Thread %3d  is running on CPU %3d\n",
             S.world_rank, omp_get_thread_num(), cpu_num);
    }
  #endif

  fflush(0);
  Master master(&comm, learners, env, S);
  MPI_Barrier(S.mastersComm); // to avoid garbled output during run

  master.run();
  comm.sendTerminateReq();
}

int main (int argc, char** argv)
{
  Settings S;
  vector<ArgParser::OptionStruct> opts = S.initializeOpts();

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &S.threadSafety);
  if (S.threadSafety < MPI_THREAD_SERIALIZED)
    die("The MPI implementation does not have required thread support");
  S.bAsync = S.threadSafety>=MPI_THREAD_MULTIPLE;
  MPI_Comm_rank(MPI_COMM_WORLD, &S.world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &S.world_size);
  omp_set_dynamic(0);

  ArgParser::Parser parser(opts);
  parser.parse(argc, argv, S.world_rank == 0);
  MPI_Barrier(MPI_COMM_WORLD);

  if(not S.isServer) die("client.sh scripts are no longer supported");

  S.initRandomSeed();

  if(S.nMasters == S.world_size)
  {
    S.bMasterSpawnApp = S.nWorkers < S.world_rank;
    S.mastersComm = MPI_COMM_WORLD;
    S.workersComm = MPI_COMM_SELF;
    S.workers_rank = 0;
    S.workers_size = 1;
    S.nWorkers_own = 1;
    runMaster(S);
  }
  else
  {
    if(S.world_size not_eq S.nMasters+S.nWorkers) die(" ");
    const int learGroupSize = std::ceil( S.world_size / (Real) S.nMasters );
    const bool bIsMaster = ( S.world_rank % learGroupSize ) == 0;
    const int workerCommInd = S.world_rank / learGroupSize;
    MPI_Comm_split(MPI_COMM_WORLD, bIsMaster,     S.world_rank, &S.mastersComm);
    MPI_Comm_split(MPI_COMM_WORLD, workerCommInd, S.world_rank, &S.workersComm);
    if(not bIsMaster) {
      MPI_Comm_free(&S.mastersComm);
      S.mastersComm = MPI_COMM_NULL;
    }
    printf("Process %d is a %s part of comm %d.\n",
        S.world_rank, bIsMaster? "master" : "worker", workerCommInd);

    MPI_Comm_rank(S.workersComm, &S.workers_rank);
    MPI_Comm_size(S.workersComm, &S.workers_size);
    S.nWorkers_own = bIsMaster? S.workers_size - 1 : 1;

    MPI_Barrier(MPI_COMM_WORLD);
    if (bIsMaster) {
      runMaster(S);
      MPI_Comm_free(&S.mastersComm);
    }
    else runWorker(S);
    MPI_Comm_free(&S.workersComm);
  }

  MPI_Finalize();
  return 0;
}
