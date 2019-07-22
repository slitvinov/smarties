//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Utils/Warnings.h"
#include "Settings.h"
#include "CLI/CLI.hpp"
#include <cassert>

namespace smarties
{

Settings::Settings() {  }
DistributionInfo::DistributionInfo(int argc, char** argv)
{
  getcwd(initial_runDir, 1024);
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, & threadSafety);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  if (threadSafety < MPI_THREAD_SERIALIZED)
    die("The MPI implementation does not have required thread support");
  // this value will determine if we can use asynchronous mpi calls:
  bAsyncMPI = threadSafety >= MPI_THREAD_MULTIPLE;
  world_size = MPICommSize(MPI_COMM_WORLD);
  world_rank = MPICommRank(MPI_COMM_WORLD);

  if (not bAsyncMPI and world_rank == 0)
    printf("MPI implementation does not support MULTIPLE thread safety!\n");
}

DistributionInfo::~DistributionInfo()
{
  if(MPI_COMM_NULL not_eq     master_workers_comm)
          MPI_Comm_free(&     master_workers_comm);
  if(MPI_COMM_NULL not_eq workerless_masters_comm)
          MPI_Comm_free(& workerless_masters_comm);
  if(MPI_COMM_NULL not_eq     learners_train_comm)
          MPI_Comm_free(&     learners_train_comm);
  if(MPI_COMM_NULL not_eq    environment_app_comm)
          MPI_Comm_free(&    environment_app_comm);
  MPI_Finalize();
}

void DistributionInfo::initializeOpts(CLI::App & parser)
{
  parser.add_option("--nThreads",               nThreads,               COMMENT_nThreads);
  parser.add_option("--nMasters",               nMasters,               COMMENT_nMasters);
  parser.add_option("--nWorkers",               nWorkers,               COMMENT_nWorkers);
  parser.add_option("--logAllSamples",          logAllSamples,          COMMENT_logAllSamples);
  parser.add_option("--maxTotSeqNum",           maxTotSeqNum,           COMMENT_maxTotSeqNum);
  parser.add_option("--randSeed",               randSeed,               COMMENT_randSeed);
  parser.add_option("--runInternalApp",         runInternalApp,         COMMENT_runInternalApp);
  parser.add_option("--nStepPappSett",          nStepPappSett,          COMMENT_nStepPappSett);
  parser.add_option("--appSettings",            appSettings,            COMMENT_appSettings);
  parser.add_option("--launchFile",             launchFile,             COMMENT_launchFile);
  parser.add_option("--setupFolder",            setupFolder,            COMMENT_setupFolder);
  parser.add_option("--learnersOnWorkers",      learnersOnWorkers,      COMMENT_learnersOnWorkers);
  parser.add_option("--fakeMastersRanks",       fakeMastersRanks,       COMMENT_fakeMastersRanks);
  parser.add_option("--workerProcessesPerEnv",  workerProcessesPerEnv,  COMMENT_workerProcessesPerEnv);

}

inline Uint notRoundedSplitting(const Uint nSplitters,
                                const Uint nToSplit,
                                const Uint splitterRank)
{
  const Uint nPerSplitter = std::ceil( nToSplit / (Real) nSplitters );
  const Uint splitBeg = std::min( splitterRank    * nPerSplitter, nToSplit);
  const Uint splitEnd = std::min((splitterRank+1) * nPerSplitter, nToSplit);
  return splitEnd - splitBeg;
}
inline Uint indxStripedMPISplitting(const Uint nSplitters,
                                    const Uint nToSplit,
                                    const Uint indexedRank)
{
  assert(indexedRank < nSplitters + nToSplit);
  for(Uint i=0, countIndex=0; i<nSplitters; ++i) {
    const Uint nInGroup = notRoundedSplitting(nSplitters, nToSplit, i);
    countIndex += nInGroup+1; // nInGroup resources + 1 handler
    if(indexedRank < countIndex) return i;
  }
  assert(false && "logic error"); return 0;
}
inline Uint rankStripedMPISplitting(const Uint nSplitters,
                                    const Uint nToSplit,
                                    const Uint indexedRank)
{
  assert(indexedRank < nSplitters + nToSplit);
  for(Uint i=0, countIndex=0; i<nSplitters; ++i) {
    const Uint nInGroup = notRoundedSplitting(nSplitters, nToSplit, i);
    if(indexedRank < countIndex + nInGroup+1)
      return indexedRank - countIndex;
    countIndex += nInGroup+1; // nInGroup resources + 1 handler
  }
  assert(false && "logic error"); return 0;
}

void DistributionInfo::figureOutWorkersPattern()
{
  nWorker_processes = world_size - nMasters;
  bool bThereAreMasters = nMasters > 0;
  bool bThereAreWorkerProcesses = nWorker_processes > 0;
  if(nWorkers < nWorker_processes) {
    _warn("Overriding user input! Setting nWorkers to %u to have at least one environment process per worker rank", nWorker_processes);
    nWorkers = nWorker_processes;
  }

  // the rest of this method will define (or leave untouched) these entities:
  // 1) will this process run a master (in a master-worker pattern) process
  bIsMaster = false;
  // 2) for how many environments will this process have to compute actions
  //    (principally affects how many Agents and associated memory buffers
  //     are allocated)
  nOwnedEnvironments = 0;
  // 3) does this process need to fork to create child processes which will
  //    in turn run the environment application (communication over sockets)
  nForkedProcesses2spawn = 0;
  // 4) mpi communicator to send state/actions or data/parameters from a process
  //    hosting the learning algo and other proc. handling data collection
  master_workers_comm = MPI_COMM_NULL;
  // 5) mpi communicator shared by all ranks that host the learning algorithms
  //    and perform the actual parameter update steps
  learners_train_comm = MPI_COMM_NULL;
  // 6) mpi communicator given to a group of worker processes that should pool
  //    together to run an environment app which requires distributed computing
  environment_app_comm = MPI_COMM_NULL;
  // 7) mpi communicator for masters without direct link to a worker to recv
  //    training data from other masters
  workerless_masters_comm = MPI_COMM_NULL;
  // 8) tag of group of mpi ranks running a common distributed environment
  thisWorkerGroupID = -1;

  if(bThereAreMasters)
  {
    if(bThereAreWorkerProcesses)
    { // then masters talk to workers, and workers own environments
      // what is the size of the mpi communicator where we have workers?
      bIsMaster =
          rankStripedMPISplitting(nMasters, nWorker_processes, world_rank) == 0;
      Uint masterWorkerCommID =
          indxStripedMPISplitting(nMasters, nWorker_processes, world_rank);

      if(fakeMastersRanks) { // overwrite splitting if we have only fake masters
        bIsMaster = world_rank < nMasters;
        masterWorkerCommID = 0;
      }

      // TEMPORARY:
      if(nWorker_processes not_eq nWorkers) die("Mismach in number of worker processes. We do not have a way for the master to know how many environments are hosted on its workers");

      MPI_Comm_split(MPI_COMM_WORLD, bIsMaster,          world_rank, & learners_train_comm);
      MPI_Comm_split(MPI_COMM_WORLD, masterWorkerCommID, world_rank, & master_workers_comm);
      printf("Process %lu is a %s part of comm %lu.\n",
          world_rank, bIsMaster? "master" : "worker", masterWorkerCommID);

      if(bIsMaster)
      {
        nOwnedEnvironments = MPICommSize(master_workers_comm) - 1;
        _warn("master %lu owns %lu environments\n", world_rank, nOwnedEnvironments);
        if(nWorker_processes < nMasters)
             workerless_masters_comm = MPICommDup(learners_train_comm);
        else workerless_masters_comm = MPI_COMM_NULL;
        nForkedProcesses2spawn = 0;

        if(runInternalApp) { // unblock creation of environment's mpi communicator
          MPI_Comm dummy; // no need to free this
          MPI_Comm_split(master_workers_comm, MPI_UNDEFINED, 0, &dummy);
        }
      }
      else
      {
        const Uint totalWorkRank = MPICommRank(learners_train_comm);
        assert(MPICommSize(learners_train_comm) == nWorker_processes);
        nOwnedEnvironments = notRoundedSplitting(nWorker_processes, nWorkers, totalWorkRank);

        const Uint innerWorkRank = MPICommRank(master_workers_comm);
        const Uint innerWorkSize = MPICommSize(master_workers_comm);
        assert(nOwnedEnvironments==1 && innerWorkRank>0 && innerWorkSize>1);

        if(runInternalApp)
        {
          nForkedProcesses2spawn = 0;
          if( (innerWorkSize-1) % workerProcessesPerEnv not_eq 0) {
            _die("Number of worker ranks per master (%u) must be a multiple of "
            "the nr. of ranks that the environment app requires to run (%u).\n",
            innerWorkSize-1, workerProcessesPerEnv);
          }
          thisWorkerGroupID = (innerWorkRank-1) / workerProcessesPerEnv;
          MPI_Comm_split(master_workers_comm, thisWorkerGroupID, innerWorkRank, &environment_app_comm);
        } else {
          thisWorkerGroupID = 0;
          nForkedProcesses2spawn = nOwnedEnvironments;
        }

        _warn("worker %lu owns %lu environments, has rank %lu out of %lu. worker ID inside group %d.\n", world_rank, nOwnedEnvironments, innerWorkRank, innerWorkSize, thisWorkerGroupID);

        MPI_Comm_free(& learners_train_comm);
        learners_train_comm = MPI_COMM_NULL;
        workerless_masters_comm = MPI_COMM_NULL;
      }
    }
    else
    {
      bIsMaster = true;
      nOwnedEnvironments = notRoundedSplitting(nMasters, nWorkers, world_rank);
      // should also equal:
      // nWorkers/world_size + ( (nWorkers%world_size) > world_rank );
      nForkedProcesses2spawn = nOwnedEnvironments;
      if(runInternalApp) die("Cannot have zero worker ranks with an internally linked app: increase the number of worker mpi processes.");
      master_workers_comm = MPI_COMM_NULL;
      learners_train_comm = MPI_COMM_WORLD;
      if(nWorkers < nMasters) // then i need to share data
           workerless_masters_comm = MPICommDup(learners_train_comm);
      else workerless_masters_comm = MPI_COMM_NULL;
    }
  }
  else // there are no masters : workers alternate environment and learner
  {
    // only have workers, 2 cases either evaluating a policy or alternate
    // data gathering and learning algorithm iteration on same comp resources
    bIsMaster = false;
    learnersOnWorkers = true;
    if(nWorker_processes <= 0) die("Error in computation of world_size");
    if(not runInternalApp) die("Detected 0 masters : this only works if each worker also serially runs its own environment.");
    if(nWorkers not_eq nWorker_processes) die("Detected 0 masters : this only works if each worker also serially runs its own environment.");
    nOwnedEnvironments = 1;
    learners_train_comm  = MPI_COMM_WORLD;
    master_workers_comm = MPI_COMM_NULL;
    workerless_masters_comm = MPI_COMM_NULL; // all are workers implies all have data

    const Uint totalWorkRank = MPICommRank(learners_train_comm);
    const Uint totalWorkSize = MPICommSize(learners_train_comm);
    if( (totalWorkSize-1) % workerProcessesPerEnv not_eq 0) {
      _die("Number of worker ranks per master (%u) must be a multiple of "
      "the nr. of ranks that the environment app requires to run (%u).\n",
      totalWorkSize-1, workerProcessesPerEnv);
    }
    thisWorkerGroupID = (totalWorkRank-1) / workerProcessesPerEnv;
    MPI_Comm_split(learners_train_comm, thisWorkerGroupID, totalWorkRank, &environment_app_comm);
  }
}

void Settings::initializeOpts (CLI::App & parser)
{
  parser.add_option("--learner",          learner,          COMMENT_learner);
  parser.add_option("--bTrain",           bTrain,           COMMENT_bTrain);
  parser.add_option("--restart",          restart,          COMMENT_restart);
  parser.add_option("--explNoise",        explNoise,        COMMENT_explNoise);
  parser.add_option("--gamma",            gamma,            COMMENT_gamma);
  parser.add_option("--lambda",           lambda,           COMMENT_lambda);
  parser.add_option("--obsPerStep",       obsPerStep,       COMMENT_obsPerStep);
  parser.add_option("--clipImpWeight",    clipImpWeight,    COMMENT_clipImpWeight);
  parser.add_option("--penalTol",         penalTol,         COMMENT_penalTol);
  parser.add_option("--klDivConstraint",  klDivConstraint,  COMMENT_klDivConstraint);
  parser.add_option("--targetDelay",      targetDelay,      COMMENT_targetDelay);
  parser.add_option("--epsAnneal",        epsAnneal,        COMMENT_epsAnneal);
  parser.add_option("--ERoldSeqFilter",   ERoldSeqFilter,   COMMENT_ERoldSeqFilter);
  parser.add_option("--dataSamplingAlgo", dataSamplingAlgo, COMMENT_dataSamplingAlgo);
  parser.add_option("--minTotObsNum",     minTotObsNum,     COMMENT_minTotObsNum);
  parser.add_option("--maxTotObsNum",     maxTotObsNum,     COMMENT_maxTotObsNum);
  parser.add_option("--bSampleSequences", bSampleSequences, COMMENT_bSampleSequences);
  parser.add_option("--saveFreq",         saveFreq,         COMMENT_saveFreq);
  parser.add_option("--totNumSteps",      totNumSteps,      COMMENT_totNumSteps);

  parser.add_option("--encoderLayerSizes",encoderLayerSizes,COMMENT_encoderLayerSizes);
  parser.add_option("--nnLayerSizes",     nnLayerSizes,     COMMENT_nnLayerSizes);
  parser.add_option("--batchSize",        batchSize,        COMMENT_batchSize);
  parser.add_option("--nnOutputFunc",     nnOutputFunc,     COMMENT_nnOutputFunc);
  parser.add_option("--nnFunc",           nnFunc,           COMMENT_nnFunc);
  parser.add_option("--learnrate",        learnrate,        COMMENT_learnrate);
  parser.add_option("--ESpopSize",        ESpopSize,        COMMENT_ESpopSize);
  parser.add_option("--nnType",           nnType,           COMMENT_nnType);
  parser.add_option("--outWeightsPrefac", outWeightsPrefac, COMMENT_outWeightsPrefac);
  parser.add_option("--nnLambda",         nnLambda,         COMMENT_nnLambda);
  parser.add_option("--nnBPTTseq",        nnBPTTseq,        COMMENT_nnBPTTseq);
}

void Settings::defineDistributedLearning(DistributionInfo& distrib)
{
  const MPI_Comm& learnersComm = distrib.learners_train_comm;
  //const MPI_Comm& gatheringComm = distrib.master_workers_comm;
  const Uint nLearners = learnersComm==MPI_COMM_NULL? 1
                         : MPICommSize(learnersComm);
  const Real nL = nLearners;
  // each learner computes a fraction of the batch:
  if(batchSize > 1) {
    batchSize = std::ceil(batchSize / nL) * nL;
    batchSize_local = batchSize / nLearners;
  } else batchSize_local = batchSize;

  if(minTotObsNum <= 0) minTotObsNum = maxTotObsNum;
  minTotObsNum = std::ceil(minTotObsNum / nL) * nL;
  minTotObsNum_local = minTotObsNum / nLearners;
  // each learner processes a fraction of the entire dataset:
  maxTotObsNum = std::ceil(maxTotObsNum / nL) * nL;
  maxTotObsNum_local = maxTotObsNum / nLearners;

  // each worker collects a fraction of the initial memory buffer:
  const Real nOwnEnvs = distrib.nOwnedEnvironments, nTotEnvs = distrib.nWorkers;
  obsPerStep_local = nOwnEnvs * obsPerStep / nTotEnvs;
  //obsPerStep_local = obsPerStep;

  if(batchSize_local <= 0) die(" ");
  if(maxTotObsNum_local <= 0) die(" ");
}

void Settings::check()
{
  bRecurrent = nnType=="LSTM" || nnType=="RNN" || nnType == "MGU" || nnType == "GRU";

  if(bTrain == false && restart == "none") {
   printf("Did not specify path for restart files, assumed current dir.\n");
   restart = ".";
  }

  if(targetDelay<0)  die("targetDelay<0");
  if(obsPerStep<0)   die("obsPerStep<0");
  if(learnrate>1)    die("learnrate>1");
  if(learnrate<0)    die("learnrate<0");
  if(explNoise<0)    die("explNoise<0");
  if(epsAnneal<0)    die("epsAnneal<0");
  if(batchSize<=0)   die("batchSize<0");
  if(nnLambda<0)     die("nnLambda<0");
  if(gamma<0)        die("gamma<0");
  if(gamma>1)        die("gamma>1");

  if(epsAnneal>0.0001 || epsAnneal<0) {
    warn("epsAnneal should be tiny. It will be set to 5e-7 for this run.");
    epsAnneal = 5e-7;
  }
}

void DistributionInfo::initialzePRNG()
{
  if(nThreads<1) die("nThreads<1");
  if(randSeed<=0)
  {
    std::random_device rdev; const Uint rdSeed = rdev();
    randSeed = rdSeed % std::numeric_limits<Uint>::max();
    MPI_Bcast(&randSeed, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if(world_rank==0) printf("Using seed %lu\n", randSeed);
  }
  generators.resize(0);
  generators.reserve(omp_get_max_threads());
  generators.push_back(std::mt19937(randSeed));
  for(int i=1; i<omp_get_max_threads(); ++i)
    generators.push_back( std::mt19937( generators[0]() ) );
}

void DistributionInfo::finalizePRNG(const Uint nAgents_local)
{
  const Uint finalSize = generators.size() + nAgents_local;
  generators.reserve(finalSize);
  for(size_t i=generators.size(); i < finalSize; ++i)
    generators.push_back( std::mt19937( generators[0]() ) );
}

}
