//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include <getopt.h>
#include "Utils/ArgParser.h"
#include "Utils/Warnings.h"
#include <functional>

class TaskQueue
{
  using cond_t = std::function<bool()>;
  using func_t = std::function<void()>;
  std::vector<std::pair<cond_t, func_t>> tasks;

public:
  inline void add(cond_t && cond, func_t && func) {
    tasks.emplace_back(std::move(cond), std::move(func));
  }

  inline void run()
  {
    // go through task list once and execute all that are ready:
    for(Uint i=0; i<tasks.size(); ++i) if( tasks[i].first() ) tasks[i].second();
  }
};

struct DistributionInfo
{
  MPI_Comm workersComm = MPI_COMM_NULL; // for workers to talk to their master
  MPI_Comm mastersComm = MPI_COMM_NULL; // for masters to talk among themselves

  int world_rank = 0;
  int world_size = 0;
  int workers_rank = 0;
  int workers_size = 0;
  int learner_rank = 0;
  int learner_size = 0;

  // number of workers (usually per master)
  int nWorkers_own = 1;
  bool bSpawnApp = false;
  int learGroupSize = -1;
  int workerCommInd = -1;
  //number of agents that:
  // in case of worker: # of agents that are contained in an environment
  // in case of master: nWorkers * # are contained in an environment
  int nAgents = -1;
  // whether Recurrent network (figured out in main)
  bool bRecurrent = false;

  int threadSafety = -1;
  bool bAsync;
  mutable std::mutex mpi_mutex;

  //random number generators (one per thread)
  mutable std::vector<std::mt19937> generators;

#define COMMENT_nThreads "Number of threads from threaded training on each \
master rank."
#define DEFAULT_nThreads 1
  int nThreads = DEFAULT_nThreads;

#define COMMENT_nMasters "Number of master ranks (policy-updating ranks)."
#define DEFAULT_nMasters 1
  int nMasters = DEFAULT_nMasters;

#define COMMENT_nWorkers "Number of worker processes (not necessarily ranks)."
#define DEFAULT_nWorkers 1
  int nWorkers = DEFAULT_nWorkers;

#define COMMENT_samplesFile "Whether to write files recording all transitions."
#define DEFAULT_samplesFile true
  bool samplesFile = DEFAULT_samplesFile;

#define COMMENT_restart "Prefix of net save files. If 'none' then no restart."
#define DEFAULT_restart "none"
  std::string restart = DEFAULT_restart;

#define COMMENT_maxTotSeqNum "DEPRECATED: Maximum number of sequences in \
training buffer"
#define DEFAULT_maxTotSeqNum 1000
  int maxTotSeqNum = DEFAULT_maxTotSeqNum;

#define COMMENT_randSeed "Random seed."
#define DEFAULT_randSeed 0
  int randSeed = DEFAULT_randSeed;

///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO ENVIRONMENT
///////////////////////////////////////////////////////////////////////////////

#define COMMENT_runInternalApp "Whether environment is linked as a library and \
requires smarties processes (=1) or we spawn an external process and comm \
through sockets (=1)."
#define DEFAULT_runInternalApp 200000
  int runInternalApp = DEFAULT_runInternalApp;

#define COMMENT_nStepPappSett "Number of time steps per appSettings file to \
use. Must be a list of positive numbers separated by semicolons. Last number \
will be overwritten to 0; i.e. last appSettings will be used til termination."
#define DEFAULT_nStepPappSett ""
  std::string nStepPappSett = DEFAULT_nStepPappSett;

#define COMMENT_appSettings "Name of file containing the command line arguments for user's application."
#define DEFAULT_appSettings ""
  std::string appSettings = DEFAULT_appSettings;

#define COMMENT_launchfile "Name of executable or launch script of user \
application. No arguments can go here. The file must be placed in the \
base run folder."
#define DEFAULT_launchfile "launchSim.sh"
  std::string launchfile = DEFAULT_launchfile;

#define COMMENT_setupFolder "The contents of this folder are copied over into the folder where the simulation is run. It can contain additional files needed to set up the simulation such as settings files, configuration files..."
#define DEFAULT_setupFolder ""
  std::string setupFolder = DEFAULT_setupFolder;
};

struct Settings
{
  Settings() {}
  ~Settings() {}

//To modify from default value any of these settings, run executable with either
//- ascii symbol of the setting (CHARARG) followed by the value (ie. -# $value)
//- the name of the setting variable followed by the value (ie. -setting $value)

///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO LEARNING ALGORITHM: lowercase LETTER
///////////////////////////////////////////////////////////////////////////////
#define COMMENT_learner "Algorithm."
#define DEFAULT_learner "RACER"
  std::string learner = DEFAULT_learner;

#define COMMENT_bTrain "Whether training a policy (=1) or evaluating (=0)."
#define DEFAULT_bTrain 1
  int bTrain = DEFAULT_bTrain;

#define COMMENT_clipImpWeight "Max importance weight for off-policy Policy \
Gradient. Algo specific."
#define DEFAULT_clipImpWeight 4
  Real clipImpWeight = DEFAULT_clipImpWeight;

#define COMMENT_targetDelay "Copy delay for target network. If >1: every \
$targetDelay grad desc steps tgt-net copies curr weigths. If <1: every \
grad desc step tgt-net does exp averaging."
#define DEFAULT_targetDelay 0
  Real targetDelay = DEFAULT_targetDelay;

#define COMMENT_explNoise "Noise added to policy. For discrete actions \
it is the probability of picking a random one (detail depend on chosen \
learning algorithm), for continuous actions it is the (initial) stdev."
#define DEFAULT_explNoise 0.5
  Real explNoise = DEFAULT_explNoise;

#define COMMENT_ERoldSeqFilter "Filter algorithm to remove old episodes from \
memory buffer. Accepts: oldest, farpolfrac, maxkldiv, minerror, or default. \
Default means oldest for ER and farpolfrac for ReFER"
#define DEFAULT_ERoldSeqFilter "default"
  std::string ERoldSeqFilter = DEFAULT_ERoldSeqFilter;

#define COMMENT_gamma "Discount factor."
#define DEFAULT_gamma 0.995
  Real gamma = DEFAULT_gamma;

#define COMMENT_dataSamplingAlgo "Algorithm for sampling the Replay Buffer."
#define DEFAULT_dataSamplingAlgo "uniform"
  std::string dataSamplingAlgo = DEFAULT_dataSamplingAlgo;

#define COMMENT_klDivConstraint "Constraint on max KL div, algo specific."
#define DEFAULT_klDivConstraint 0.01
  Real klDivConstraint = DEFAULT_klDivConstraint;

#define COMMENT_lambda "Lambda for off policy corrections."
#define DEFAULT_lambda 0.95
  Real lambda = DEFAULT_lambda;

#define COMMENT_minTotObsNum "Min number of transitions in training buffer \
before training starts. If unset we use maxTotObsNum."
#define DEFAULT_minTotObsNum -1
  int minTotObsNum = DEFAULT_minTotObsNum;

#define COMMENT_maxTotObsNum "Max number of transitions in training buffer."
#define DEFAULT_maxTotObsNum 1000000
  int maxTotObsNum = DEFAULT_maxTotObsNum;

#define COMMENT_obsPerStep "Ratio of observed *transitions* to gradient \
steps. 0.1 means that for every observation, learner does 10 gradient steps."
#define DEFAULT_obsPerStep  1
  Real obsPerStep = DEFAULT_obsPerStep;

#define COMMENT_penalTol "Tolerance used for adaptive penalization methods. \
Algo specific."
#define DEFAULT_penalTol  0.1
  Real penalTol = DEFAULT_penalTol;

#define COMMENT_epsAnneal "Annealing rate in grad steps of various \
learning-algorithm-dependent behaviors."
#define DEFAULT_epsAnneal 5e-7
  Real epsAnneal = DEFAULT_epsAnneal;

#define COMMENT_bSampleSequences "Whether to sample sequences (1) \
or observations (0) from the Replay Memory."
#define DEFAULT_bSampleSequences  0
  int bSampleSequences = DEFAULT_bSampleSequences;

#define COMMENT_saveFreq "Frequency of checkpoints of learner state. These \
checkpoints can be used to evaluate learners, but not yet to restart learning."
#define DEFAULT_saveFreq 200000
  int saveFreq = DEFAULT_saveFreq;

#define COMMENT_totNumSteps "Number of gradient steps before end of learning"
#define DEFAULT_totNumSteps 10000000
  int totNumSteps = DEFAULT_totNumSteps;

///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO NETWORK: CAPITAL LETTER
///////////////////////////////////////////////////////////////////////////////

#define COMMENT_nnLayerSizes "Sizes of non-convolutional layers \"
"(LSTM/RNN/FFNN). E.g. '128 128'."
#define DEFAULT_nnLayerSizes 0
  std::vector<int> nnLayerSizes = DEFAULT_nnLayerSizes;

#define COMMENT_batchSize "Network training batch size."
#define DEFAULT_batchSize 128
  int batchSize = DEFAULT_batchSize;

#define COMMENT_nnOutputFunc "Activation function for output layers."
#define DEFAULT_nnOutputFunc "Linear"
  std::string nnOutputFunc = DEFAULT_nnOutputFunc;

#define COMMENT_nnFunc "Activation function for non-output layers (which should\
 always be linear) which are built from settings. (Relu, Tanh, Sigm, PRelu, \
softSign, softPlus, ...)"
#define DEFAULT_nnFunc "SoftSign"
  std::string nnFunc = DEFAULT_nnFunc;

#define COMMENT_learnrate "Learning rate."
#define DEFAULT_learnrate 1e-4
  Real learnrate = DEFAULT_learnrate;

#define COMMENT_ESpopSize "Population size for ES algorithm."
#define DEFAULT_ESpopSize 1
  int ESpopSize = DEFAULT_ESpopSize;

#define COMMENT_nnType "Type of non-output layers read from settings. (RNN, \
LSTM, everything else maps to FFNN). Conv2D layers need to be built in \
environment directly."
#define DEFAULT_nnType "FFNN"
  std::string nnType = DEFAULT_nnType;

#define COMMENT_outWeightsPrefac "Output weights initialization factor (will \
be multiplied by default fan-in factor). Picking 1 leads to treating \
output layers with normal Xavier initialization."
#define DEFAULT_outWeightsPrefac 1
  Real outWeightsPrefac = DEFAULT_outWeightsPrefac;

#define COMMENT_nnLambda "Penalization factor for network weights. It will be \
multiplied by learn rate: w -= eta * nnLambda * w . L1 decay option in Bund.h"
#define DEFAULT_nnLambda std::numeric_limits<float>::epsilon()
  Real nnLambda = DEFAULT_nnLambda;

#define COMMENT_nnBPTTseq "Number of previous steps considered by RNN."
#define DEFAULT_nnBPTTseq 16
  int nnBPTTseq = DEFAULT_nnBPTTseq;

#define COMMENT_splitLayers "Number of split layers, description in Settings.h"
//"For each output required by algorithm (ie. value, policy, std, ...) " \/
//"how many non-conv layers should be devoted only to one o the outputs. " \/
//"For example if there 2 FF layers of size Z and Y and this arg is set to 1,"\/
//" then each of the outputs is connected to a separate layer of size Y. " \/
//"Each of the Y-size layers are then connected to the first layer of size Z."
#define DEFAULT_splitLayers 0
  int splitLayers = DEFAULT_splitLayers;


///////////////////////////////////////////////////////////////////////////////
//SETTINGS THAT ARE NOT READ FROM FILE
///////////////////////////////////////////////////////////////////////////////
  // rank-local data-acquisition goals
  int batchSize_loc = -1;
  Real obsPerStep_loc = -1;
  int minTotObsNum_loc = -1;
  int maxTotObsNum_loc = -1;

  void check();

  void initRandomSeed();

  void finalizeSeeds();

  std::vector<int> readNetSettingsSize() const;
};
