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
void runWorker(Settings& S, MPI_Comm workersComm);
void runMaster(Settings& S, MPI_Comm workersComm, MPI_Comm mastersComm);

void runWorker(Settings& S, MPI_Comm workersComm)
{
  MPI_Comm_rank(workersComm, &S.workers_rank);
  MPI_Comm_size(workersComm, &S.workers_size);
  assert(S.workers_rank and S.workers_size>0);
  ObjectFactory factory(S);
  Environment* env = factory.createEnvironment();
  const auto comm = env->create_communicator(workersComm, S.sockPrefix, true);

  #ifndef NDEBUG
    int cpu_num; GETCPU(cpu_num); //sched_getcpu()
    printf("Worker Rank %d is running on CPU %3d\n",
      S.world_rank, cpu_num);
  #endif

  Worker simulation(&comm, env, S);
  simulation.run();
}

void runMaster(Settings& S, MPI_Comm workersComm, MPI_Comm mastersComm)
{
  S.mastersComm =  mastersComm;
  MPI_Comm_rank(workersComm, &S.workers_rank);
  MPI_Comm_size(workersComm, &S.workers_size);
  MPI_Comm_rank(mastersComm, &S.learner_rank);
  MPI_Comm_size(mastersComm, &S.learner_size);
  S.nWorkers_own = S.workers_size - 1; //minus master
  assert(S.workers_rank == 0);

  #ifdef INTERNALAPP //unblock creation of app comm if needed
    MPI_Comm tmp_com;
    MPI_Comm_split(workersComm, MPI_UNDEFINED, 0, &tmp_com);
    //no need to free this
  #endif

  ObjectFactory factory(S);
  Environment*const env = factory.createEnvironment();
  const auto comm = env->create_communicator(workersComm, S.sockPrefix, true);

  S.finalizeSeeds(); // now i know nAgents, might need more generators
  {
    const Real nLearners = S.learner_size, nWorkers = S.nWorkers;
    // each learner computes a fraction of the batch:
    S.batchSize = std::ceil(S.batchSize / nLearners) * nLearners;
    // every grad step, each worker performs a fraction of the time steps:
    S.obsPerStep = std::ceil(S.obsPerStep / nWorkers) * nWorkers;
    // each worker collects a fraction of the initial memory buffer:
    S.minTotObsNum = std::ceil(S.minTotObsNum / nWorkers) * nWorkers;
    // each learner processes a fraction of the entire dataset:
    S.maxTotObsNum = std::ceil(S.maxTotObsNum / nLearners) * nLearners;

    S.batchSize_loc = S.batchSize / S.learner_size;
    S.obsPerStep_loc = S.obsPerStep / S.nWorkers;
    S.maxTotObsNum_loc = S.maxTotObsNum / S.learner_size;
    S.minTotObsNum_loc = S.minTotObsNum / S.nWorkers;
  }

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
  Master master(workersComm, learners, env, S);
  MPI_Barrier(mastersComm); // to avoid garbled output during run

  master.run();
  master.sendTerminateReq();
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
  S.check();
  MPI_Barrier(MPI_COMM_WORLD);

  if(not S.isServer) die("client.sh scripts are no longer supported");

  S.initRandomSeed();

  MPI_Comm workersComm; // communicator for workers to talk to their master
  MPI_Comm mastersComm; // communicator for masters to talk among themselves

  if(S.nMasters == S.world_size)
  {
    S.bMasterSpawnApp = S.nWorkers < S.world_rank;
    mastersComm = MPI_COMM_WORLD;
    workersComm = MPI_COMM_SELF;

    runMaster(S, workersComm, mastersComm);
  }
  else
  {
    if(S.world_size not_eq S.nMasters+S.nWorkers) die(" ");
    const int learGroupSize = std::ceil( S.world_size / (Real) S.nMasters );
    const bool bIsMaster = ( S.world_rank % learGroupSize ) == 0;
    const int workerCommInd = S.world_rank / learGroupSize;
    MPI_Comm_split(MPI_COMM_WORLD, bIsMaster, S.world_rank, &mastersComm);
    MPI_Comm_split(MPI_COMM_WORLD, workerCommInd, S.world_rank, &workersComm);
    printf("Process %d is a %s part of comm %d.\n",
        S.world_rank, bIsMaster? "master" : "worker", workerCommInd);

    MPI_Barrier(MPI_COMM_WORLD);
    if (bIsMaster) runMaster(S, workersComm, mastersComm);
    else           runWorker(S, workersComm);
    MPI_Comm_free(&mastersComm);
    MPI_Comm_free(&workersComm);
  }

  MPI_Finalize();
  return 0;
}
