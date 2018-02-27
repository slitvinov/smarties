/*
 *  main.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "AllLearners.h"
#include "Scheduler.h"
#include "ObjectFactory.h"
using namespace std;
//TODO: enable app to give a partial sequence
//example: you have 4 agents, 2 kill each other and therefore reach a
//terminal state, the other two have not reached it, but app exits.

void runClient();
void runSlave(MPI_Comm slavesComm);
void runMaster(MPI_Comm slavesComm, MPI_Comm mastersComm);
Settings settings;

void runSlave(MPI_Comm slavesComm)
{
  MPI_Comm_rank(slavesComm, &settings.slaves_rank);
  MPI_Comm_size(slavesComm, &settings.slaves_size);
  if(settings.slaves_rank==0) die("Slave is master?\n")
  if(settings.slaves_size<=1) die("Slave has no master?\n");
  settings.nSlaves = 1;
  ObjectFactory factory(settings);
  Environment* env = factory.createEnvironment();
  Communicator comm = env->create_communicator(slavesComm, settings.sockPrefix, true);

  Slave simulation(&comm, env, settings);
  simulation.run();
}

void runMaster(MPI_Comm slavesComm, MPI_Comm mastersComm)
{
  settings.mastersComm =  mastersComm;
  MPI_Comm_rank(slavesComm, &settings.slaves_rank);
  MPI_Comm_size(slavesComm, &settings.slaves_size);
  MPI_Comm_rank(mastersComm, &settings.learner_rank);
  MPI_Comm_size(mastersComm, &settings.learner_size);
  settings.nSlaves = settings.slaves_size-1; //minus master
  assert(settings.nSlaves>=0 && settings.slaves_rank == 0);

  ObjectFactory factory(settings);
  Environment*const env = factory.createEnvironment();
  Communicator comm = env->create_communicator(slavesComm, settings.sockPrefix, true);

  const Real nLearners = settings.learner_size;
  // each learner computes a fraction of the batch:
  settings.batchSize    = std::ceil(settings.batchSize    / nLearners);
  // every grad step, each learner performs a fraction of the time steps:
  settings.obsPerStep   = std::ceil(settings.obsPerStep   / nLearners);
  // each learner contains a fraction of the memory buffer:
  settings.minTotObsNum = std::ceil(settings.minTotObsNum / nLearners);
  settings.maxTotObsNum = std::ceil(settings.maxTotObsNum / nLearners);

  if(env->mpi_ranks_per_env) { //unblock creation of app comm if needed
    MPI_Comm tmp_com;
    MPI_Comm_split(slavesComm, MPI_UNDEFINED, 0, &tmp_com);
    //no need to free this
  }

  const Uint nPols = settings.bSharedPol ? env->nAgentsPerRank : 1;
  vector<Learner*> learners(nPols, nullptr);
  for(Uint i = 0; i<nPols; i++) {
    learners[i] = createLearner(env, settings);
    stringstream ss; ss << std::setw(2) << std::setfill('0') << i;
    learners[i]->setLearnerName("agent_"+ss.str()+"_");
    learners[i]->restart();
  }

  #pragma omp parallel
  printf("Rank %d Thread %3d is running on CPU %3d\n",
    settings.world_rank, omp_get_thread_num(), sched_getcpu());

  fflush(0);
  Master master(slavesComm, learners, env, settings);
  MPI_Barrier(mastersComm); // to avoid garbled output during run

  #if 0
  if (!settings.nSlaves && !learner->nData())
  {
    printf("No slaves, just dumping the policy\n");
    learner->dumpPolicy();
    abort();
  }
  #endif

  master.run();
  master.sendTerminateReq();
}

int main (int argc, char** argv)
{
  struct timeval clock;
  gettimeofday(&clock, NULL);
  omp_set_dynamic(0);

  vector<ArgParser::OptionStruct> opts = settings.initializeOpts();

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  if (provided < MPI_THREAD_SERIALIZED)
    die("The MPI implementation does not have required thread support\n");

  MPI_Comm_rank(MPI_COMM_WORLD, &settings.world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &settings.world_size);

  ArgParser::Parser parser(opts);
  parser.parse(argc, argv, settings.world_rank == 0);
  settings.check();
  MPI_Barrier(MPI_COMM_WORLD);

  if (not settings.isServer) {
    die("You should not be running the client.sh scripts.\n");
    /*
    if (settings.sockPrefix<0)
      die("Not received a prefix for the socket\n");
    settings.generators.push_back(mt19937(settings.sockPrefix));
    printf("Launching smarties as client.\n");
    if (settings.restart == "none")
      die("smarties as client works only for evaluating policies.\n");
    settings.bTrain = 0;
    runClient();
    MPI_Finalize();
    return 0;
    */
  }

  const long MAXINT = std::numeric_limits<int>::max();
  int runSeed = abs(clock.tv_usec % MAXINT);
  MPI_Bcast(&runSeed, 1, MPI_INT, 0, MPI_COMM_WORLD);
  settings.sockPrefix = runSeed+settings.world_rank;

  if(settings.bTrain && settings.nThreads<2)
    die("must have at least 2 threads\n");
  if(!settings.bTrain && settings.nThreads<1)
    die("must have at least 1 thread even when not training.\n");

  settings.generators.reserve(omp_get_max_threads());
  settings.generators.push_back(mt19937(settings.sockPrefix));
  for(int i=1; i<omp_get_max_threads(); i++) {
    const Uint seed = settings.generators[0]();
    settings.generators.push_back(mt19937(seed));
  }

  if(settings.world_size%settings.nMasters)
    die("Number of masters not compatible with available ranks.");
  const int slavesPerMaster = settings.world_size/settings.nMasters - 1;

  MPI_Comm slavesComm; //this communicator allows slaves to talk to their master
  MPI_Comm mastersComm; //this communicator allows masters to talk among themselves

  int bIsMaster, slaveCommInd;
  // two options: either multiple learners because they are the bottleneck
  //              or multiple slaves for single master because data is expensive
  // in the second case, rank 0 will be master either away
  // in first case our objective is to maximise the spread of master ranks
  // and use processes on hyperthreaded cores to run the slaves
  // in a multi socket board usually the cpus are going to be sorted
  // as (socket-core-thread): 0-0-0 0-1-0 0-2-0 ... 1-0-0 1-1-0 1-2-0 ...
  //                          0-0-1 0-1-1 0-2-1 ... 1-0-1 1-1-1 1-2-1
  // therefore if there are more than one master per node sorting changes
  // this is al very brittle. relies on my MPI implementations sorting of ranks
  if (settings.ppn > slavesPerMaster+1) {
    if(settings.ppn % (slavesPerMaster+1)) die("Bad number of proc per node");
    const int nMastersPerNode =  settings.ppn / (slavesPerMaster+1);
    const int nodeIndx = settings.world_rank / settings.ppn;
    const int nodeRank = settings.world_rank % settings.ppn;
    // will be 1 for the first nMastersPerNode ranks of each node:
    bIsMaster = nodeRank / nMastersPerNode == 0;
    // index will be shared by every nMastersPerNode ranks:
    const int nodeMScomm = nodeRank % nMastersPerNode;
    // split communicators residing on different nodes:
    slaveCommInd = nodeMScomm + nodeIndx * nMastersPerNode;
  } else {
    bIsMaster = settings.world_rank % (slavesPerMaster+1) == 0;
    slaveCommInd = settings.world_rank / (slavesPerMaster+1);
  }

  MPI_Comm_split(MPI_COMM_WORLD, bIsMaster, settings.world_rank, &mastersComm);
  MPI_Comm_split(MPI_COMM_WORLD, slaveCommInd, settings.world_rank,&slavesComm);
  if (!bIsMaster) MPI_Comm_free(&mastersComm);
  printf("nRanks=%d, %d masters, %d slaves per master. I'm %d: %s part of comm %d.\n",
      settings.world_size,settings.nMasters,slavesPerMaster,settings.world_rank,
      bIsMaster?"master":"slave",slaveCommInd);

  MPI_Barrier(MPI_COMM_WORLD);
  if (bIsMaster) runMaster(slavesComm, mastersComm);
  else           runSlave(slavesComm);

  if (bIsMaster) MPI_Comm_free(&mastersComm);
  MPI_Comm_free(&slavesComm);
  MPI_Finalize();
  return 0;
}


/*
void runClient()
{
  settings.nSlaves = 1;
  ObjectFactory factory(settings);
  Environment* env = factory.createEnvironment();
  Communicator comm = env->create_communicator(MPI_COMM_NULL, settings.sockPrefix, false);

  Learner* learner = createLearner(MPI_COMM_WORLD, env, settings);
  if (settings.restart != "none") {
    learner->restart(settings.restart);
    //comm.restart(settings.restart);
  }
  Client simulation(learner, &comm, env, settings);
  simulation.run();
}
*/
