//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Utils/ArgParser.h"
#include "Utils/Warnings.h"
#include "Settings.h"

#include <functional>
#include <getopt.h>

std::vector<ArgParser::OptionStruct> Settings::initializeOpts ()
{ //  //{ CHARARG_, "", TYPENUM_, COMMENT_, &, (TYPEVAL_) DEFAULT_ },
  //AVERT YOUR EYES!

  return std::vector<ArgParser::OptionStruct> ({
    // LEARNER ARGS: MUST contain all 17 mentioned above (more if modified)
    READOPT(learner), READOPT(bTrain), READOPT(clipImpWeight),
    READOPT(targetDelay), READOPT(explNoise), READOPT(ERoldSeqFilter),
    READOPT(gamma), READOPT(dataSamplingAlgo), READOPT(klDivConstraint),
    READOPT(lambda), READOPT(minTotObsNum), READOPT(maxTotObsNum),
    READOPT(obsPerStep), READOPT(penalTol), READOPT(epsAnneal),
    READOPT(bSampleSequences), READOPT(bSharedPol), READOPT(totNumSteps),

    // NETWORK ARGS: MUST contain all 15 mentioned above (more if modified)
    READOPT(nnl1), READOPT(nnl2), READOPT(nnl3), READOPT(nnl4),
    READOPT(nnl5), READOPT(nnl6), READOPT(batchSize), READOPT(appendedObs),
    READOPT(nnPdrop), READOPT(nnOutputFunc), READOPT(nnFunc),
    READOPT(learnrate), READOPT(ESpopSize), READOPT(nnType),
    READOPT(outWeightsPrefac),
    READOPT(nnLambda), READOPT(splitLayers), READOPT(nnBPTTseq),

    // SMARTIES ARGS: MUST contain all 10 mentioned above (more if modified)
    READOPT(nThreads), READOPT(nMasters), READOPT(nWorkers),
    READOPT(isServer), READOPT(sockPrefix), READOPT(samplesFile),
    READOPT(restart), READOPT(maxTotSeqNum),
    READOPT(randSeed), READOPT(saveFreq), READOPT(runInternalApp),

    // ENVIRONMENT ARGS: MUST contain all 7 mentioned above (more if modified)
    READOPT(environment), READOPT(workersPerEnv), READOPT(rType),
    READOPT(senses), READOPT(nStepPappSett),
    READOPT(launchfile), READOPT(appSettings), READOPT(setupFolder)
  });
}

void Settings::check()
{
  bRecurrent = nnType=="LSTM" || nnType=="RNN" || nnType == "MGU" || nnType == "GRU";

  if(bSampleSequences && maxTotSeqNum<batchSize)
    die("Increase memory buffer size or decrease batchsize, or switch to sampling by transitions.");

  if(bTrain == false && restart == "none") {
   std::cout<<"Did not specify path for restart files, assumed current dir."<<std::endl;
   restart = ".";
  }

  if(mastersComm == MPI_COMM_NULL) die(" ");

  MPI_Comm_rank(mastersComm, &learner_rank);
  MPI_Comm_size(mastersComm, &learner_size);
  assert(workers_rank == 0);

  const Real nL = learner_size;
  // each learner computes a fraction of the batch:
  if(batchSize > 1) {
    batchSize = std::ceil(batchSize / nL) * nL;
    batchSize_loc = batchSize / learner_size;
  } else batchSize_loc = batchSize;

  // each worker collects a fraction of the initial memory buffer:
  if(minTotObsNum < 0) minTotObsNum = maxTotObsNum;
  minTotObsNum = std::ceil(minTotObsNum / nL) * nL;
  minTotObsNum_loc = minTotObsNum / learner_size;
  // each learner processes a fraction of the entire dataset:
  maxTotObsNum = std::ceil(maxTotObsNum / nL) * nL;
  maxTotObsNum_loc = maxTotObsNum / learner_size;

  obsPerStep_loc = nWorkers_own * obsPerStep / nWorkers;

  if(batchSize_loc <= 0) die(" ");
  if(obsPerStep_loc < 0) die(" ");
  if(minTotObsNum_loc < 0) die(" ");
  if(maxTotObsNum_loc <= 0) die(" ");

  if(appendedObs<0)  die("appendedObs<0");
  if(targetDelay<0)  die("targetDelay<0");
  if(splitLayers<0)  die("splitLayers<0");
  if(totNumSteps<0)  die("totNumSteps<0");
  if(sockPrefix<0)   die("sockPrefix<0");
  if(obsPerStep<0)   die("obsPerStep<0");
  if(learnrate>.1)   die("learnrate>.1");
  if(learnrate<0)    die("learnrate<0");
  if(explNoise<0)    die("explNoise<0");
  if(epsAnneal<0)    die("epsAnneal<0");
  if(batchSize<0)    die("batchSize<0");
  if(nnLambda<0)     die("nnLambda<0");
  if(nThreads<1)     die("nThreads<1");
  if(nMasters<1)     die("nMasters<1");
  if(gamma<0)        die("gamma<0");
  if(gamma>1)        die("gamma>1");
  if(nnl1<0)         die("nnl1<0");
  if(nnl2<0)         die("nnl2<0");
  if(nnl3<0)         die("nnl3<0");
  if(nnl4<0)         die("nnl4<0");
  if(nnl5<0)         die("nnl5<0");

  if(epsAnneal>0.0001 || epsAnneal<0) {
    warn("epsAnneal should be tiny. It will be set to 5e-7 for this run.");
    epsAnneal = 5e-7;
  }
}

void Settings::initRandomSeed()
{
  if(randSeed<=0) {
    std::random_device rdev;
    static constexpr long MAXINT = std::numeric_limits<int>::max();
    randSeed = std::abs(rdev() % MAXINT);

    std::cout << "Using seed " << randSeed << std::endl;
    MPI_Bcast(&randSeed, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  sockPrefix = randSeed + world_rank;
  generators.resize(0);
  generators.reserve(omp_get_max_threads());
  generators.push_back(std::mt19937(sockPrefix));
  for(int i=1; i<omp_get_max_threads(); i++) {
    const Uint seed = generators[0]();
    generators.push_back(std::mt19937(seed));
  }
}

void Settings::finalizeSeeds()
{
  const int currsize = generators.size();
  if(currsize < nThreads + nAgents) {
    generators.reserve(nThreads+nAgents);
    for(int i=currsize; i<nThreads+nAgents; i++) {
      const Uint seed = generators[0]();
      generators.push_back(std::mt19937(seed));
    }
  }
}

std::vector<int> Settings::readNetSettingsSize() const
{
  std::vector<int> ret;
  //if(nnl1<1) die("Add at least one hidden layer.\n");
  if(nnl1>0) {
    ret.push_back(nnl1);
    if (nnl2>0) {
      ret.push_back(nnl2);
      if (nnl3>0) {
        ret.push_back(nnl3);
        if (nnl4>0) {
          ret.push_back(nnl4);
          if (nnl5>0) {
            ret.push_back(nnl5);
            if (nnl6>0) {
              ret.push_back(nnl6);
            }
          }
        }
      }
    }
  }
  return ret;
}
