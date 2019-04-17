//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learners/AllLearners.h"
#include "Core/Scheduler.h"


inline void initialzeOMP(DistributionInfo & distrib)
{
  //omp_set_dynamic(0);
  #if 0
  #pragma omp parallel
  {
    int cpu_num; GETCPU(cpu_num); //sched_getcpu()
    char hostname[1024];
    hostname[1023] = '\0';
    gethostname(hostname, 1023);
    //#ifndef NDEBUG
      printf("Rank %d Thread %3d  is running on CPU %3d of host %s\n",
            S.world_rank, omp_get_thread_num(), cpu_num, hostname);
    //#endif
  }
  #endif
}

int main (int argc, char** argv)
{
  Settings settings;
  DistributionInfo distrib(argc, argv);

  CLI:App parser{"smarties : distributed reinforcement learning framework"};
  settings.initializeOpts(parser);
  distrib.initializeOpts(parser);
  try {
    app.parse(argc, argv);
  }
  catch (const CLI::ParseError &e) {
    if(distrib.world_rank == 0) return app.exit(e);
    else return 1;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  S.initRandomSeed();


  if(S.nMasters == S.world_size)
  {
    S.bSpawnApp = S.nWorkers > S.world_rank;
    S.mastersComm = MPI_COMM_WORLD;
    S.workersComm = MPI_COMM_NULL;
    S.workers_rank = 0;
    S.workers_size = 1;
    S.nWorkers_own =
      S.nWorkers/S.world_size + ( (S.nWorkers%S.world_size) > S.world_rank );
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
      S.bSpawnApp = 1;
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

  return 0;
}

void figureOutWorkersPattern()
{
  if(distrib.runInternalApp)
  {
    const int nWorker_processes = distrib.runInternalApp
  }
  else
  {

  }
}

void runClient();
void runWorker(Settings& S);
void runMaster(Settings& S);

void runWorker(Settings& S)
{
  assert(S.workers_rank and S.workers_size>0);
  ObjectFactory factory(S);
  Environment* env = factory.createEnvironment();
  Communicator_internal comm = env->create_communicator();

  if(not S.runInternalApp) {
    // if not running an internal app, worker has forked and will now
    // act as intermediary with master:
    Worker simulation(&comm, env, S);
    simulation.run();
  }
}

void runMaster(Settings& S)
{
  S.check();

  ObjectFactory factory(S);
  Environment*const env = factory.createEnvironment();
  Communicator_internal comm = env->create_communicator();

  S.finalizeSeeds(); // now i know nAgents, might need more generators

  const Uint nPols = S.bSharedPol ? 1 : env->nAgentsPerRank;
  vector<Learner*> learners(nPols, nullptr);
  for(Uint i = 0; i<nPols; i++) {
    stringstream ss; ss<<"agent_"<<std::setw(2)<<std::setfill('0')<<i;
    if(S.world_rank == 0) cout << "Learner: " << ss.str() << endl;
    learners[i] = createLearner(env, S);
    learners[i]->setLearnerName(ss.str() +"_", i);
    learners[i]->restart();
  }

  fflush(stdout); fflush(stderr); fflush(0);
  MPI_Barrier(S.mastersComm); // to avoid garbled output during run
  Master master(&comm, learners, env, S);

  master.run();
}
