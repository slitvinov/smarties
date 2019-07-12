//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Settings_h
#define smarties_Settings_h

#include "Utils/Definitions.h"
#include "Utils/MPIUtilities.h"

#include <random>
#include <mutex>

namespace CLI { class App; }

namespace smarties
{

struct DistributionInfo
{
  DistributionInfo(int argc, char** argv);
  ~DistributionInfo();

  void figureOutWorkersPattern();
  void initializeOpts(CLI::App & parser);
  void initialzePRNG();
  void finalizePRNG(const Uint nAgents_local);

  Uint world_rank;
  Uint world_size;

  int threadSafety = -1;
  bool bAsyncMPI;
  mutable std::mutex mpiMutex;

  Uint nWorker_processes;
  Sint thisWorkerGroupID = -1;
  Uint nAgents;

  MPI_Comm master_workers_comm = MPI_COMM_NULL;
  MPI_Comm workerless_masters_comm = MPI_COMM_NULL;
  MPI_Comm learners_train_comm = MPI_COMM_NULL;
  MPI_Comm environment_app_comm = MPI_COMM_NULL;

  bool bIsMaster;
  Uint nOwnedEnvironments = 0;
  Uint nOwnedAgentsPerAlgo = 1;
  Uint nForkedProcesses2spawn = 0;

  //random number generators (one per thread)
  mutable std::vector<std::mt19937> generators;

#define COMMENT_nThreads "Number of threads from threaded training on each \
master rank."
#define DEFAULT_nThreads 1
  Uint nThreads = DEFAULT_nThreads;

#define COMMENT_nMasters "Number of master ranks (policy-updating ranks)."
#define DEFAULT_nMasters 1
  Uint nMasters = DEFAULT_nMasters;

#define COMMENT_nWorkers "Number of worker processes (not necessarily ranks)."
#define DEFAULT_nWorkers 1
  Uint nWorkers = DEFAULT_nWorkers;

#define COMMENT_logAllSamples "Whether to write files recording all transitions."
#define DEFAULT_logAllSamples true
  bool logAllSamples = DEFAULT_logAllSamples;

#define COMMENT_maxTotSeqNum "DEPRECATED: Maximum number of sequences in \
training buffer"
#define DEFAULT_maxTotSeqNum 1000
  Uint maxTotSeqNum = DEFAULT_maxTotSeqNum;

#define COMMENT_randSeed "Random seed."
#define DEFAULT_randSeed 0
  Uint randSeed = DEFAULT_randSeed;

#define COMMENT_learnersOnWorkers "Whether to enable hosting learning algos \
on worker processes such that workers send training data and recv parameters \
from masters. If false workers only send states and recv actions from masters."
#define DEFAULT_learnersOnWorkers true
  bool learnersOnWorkers = DEFAULT_learnersOnWorkers;

#define COMMENT_fakeMastersRanks "This options will pack master ranks in the \
first nMaster processes and put all of them except one to sleep. This is used \
for environment applications that require MPI but do not support omp threads. \
In this case we need to create one process per CPU core. To still enable fast \
training on one node we can create fake processes to fill node 0 and put all \
except for one to sleep. smarties will behave as if node 0 is the master and \
all other nodes are workers, with 1 process per CPU core. "
#define DEFAULT_fakeMastersRanks false
  bool fakeMastersRanks = DEFAULT_fakeMastersRanks;

#define COMMENT_workerProcessesPerEnv "Number of MPI ranks required by the the env \
application. It is 1 for serial/shared-memory solvers."
#define DEFAULT_workerProcessesPerEnv 1
  Uint workerProcessesPerEnv = DEFAULT_workerProcessesPerEnv;

///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO ENVIRONMENT
///////////////////////////////////////////////////////////////////////////////

#define COMMENT_runInternalApp "Whether environment is linked as a library and \
requires smarties processes (=1) or we spawn an external process and comm \
through sockets (=1)."
#define DEFAULT_runInternalApp false
  bool runInternalApp = DEFAULT_runInternalApp;

#define COMMENT_nStepPappSett "Number of time steps per appSettings file to \
use. Must be a list of positive numbers separated by semicolons. Last number \
will be overwritten to 0; i.e. last appSettings will be used til termination."
#define DEFAULT_nStepPappSett "0"
  std::string nStepPappSett = DEFAULT_nStepPappSett;

#define COMMENT_appSettings "Name of file containing the command line arguments for user's application."
#define DEFAULT_appSettings ""
  std::string appSettings = DEFAULT_appSettings;

#define COMMENT_launchFile "Name of executable or launch script of user \
application. No arguments can go here. The file must be placed in the \
base run folder."
#define DEFAULT_launchFile "launchSim.sh"
  std::string launchFile = DEFAULT_launchFile;

#define COMMENT_setupFolder "The contents of this folder are copied over into the folder where the simulation is run. It can contain additional files needed to set up the simulation such as settings files, configuration files..."
#define DEFAULT_setupFolder ""
  std::string setupFolder = DEFAULT_setupFolder;
};

struct Settings
{
  Settings();
  void check();
  void initializeOpts(CLI::App & parser);
  void defineDistributedLearning(DistributionInfo&);

///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO LEARNING ALGORITHM
///////////////////////////////////////////////////////////////////////////////
#define COMMENT_learner "Algorithm."
#define DEFAULT_learner "RACER"
  std::string learner = DEFAULT_learner;

#define COMMENT_bTrain "Whether training a policy (=1) or evaluating (=0)."
#define DEFAULT_bTrain true
  bool bTrain = DEFAULT_bTrain;

#define COMMENT_explNoise "Noise added to policy. For discrete actions \
it is the probability of picking a random one (detail depend on chosen \
learning algorithm), for continuous actions it is the (initial) stdev."
#define DEFAULT_explNoise 0.5
  Real explNoise = DEFAULT_explNoise;

#define COMMENT_gamma "Discount factor."
#define DEFAULT_gamma 0.995
  Real gamma = DEFAULT_gamma;

#define COMMENT_lambda "Lambda for off policy corrections."
#define DEFAULT_lambda 0.95
  Real lambda = DEFAULT_lambda;

#define COMMENT_obsPerStep "Ratio of observed *transitions* to gradient \
steps. 0.1 means that for every observation, learner does 10 gradient steps."
#define DEFAULT_obsPerStep  1
  Real obsPerStep = DEFAULT_obsPerStep;

#define COMMENT_clipImpWeight "Max importance weight for off-policy Policy \
Gradient. Algo specific."
#define DEFAULT_clipImpWeight 4
  Real clipImpWeight = DEFAULT_clipImpWeight;

#define COMMENT_penalTol "Tolerance used for adaptive penalization methods. \
Algo specific."
#define DEFAULT_penalTol  0.1
  Real penalTol = DEFAULT_penalTol;

#define COMMENT_klDivConstraint "Constraint on max KL div, algo specific."
#define DEFAULT_klDivConstraint 0.01
  Real klDivConstraint = DEFAULT_klDivConstraint;

#define COMMENT_targetDelay "Copy delay for target network. If >1: every \
$targetDelay grad desc steps tgt-net copies curr weigths. If <1: every \
grad desc step tgt-net does exp averaging."
#define DEFAULT_targetDelay 0
  Real targetDelay = DEFAULT_targetDelay;

#define COMMENT_epsAnneal "Annealing rate in grad steps of various \
learning-algorithm-dependent behaviors."
#define DEFAULT_epsAnneal 0
  Real epsAnneal = DEFAULT_epsAnneal;

#define COMMENT_ERoldSeqFilter "Filter algorithm to remove old episodes from \
memory buffer. Accepts: oldest, farpolfrac, maxkldiv, minerror, or default. \
Default means oldest for ER and farpolfrac for ReFER"
#define DEFAULT_ERoldSeqFilter "default"
  std::string ERoldSeqFilter = DEFAULT_ERoldSeqFilter;

#define COMMENT_dataSamplingAlgo "Algorithm for sampling the Replay Buffer."
#define DEFAULT_dataSamplingAlgo "uniform"
  std::string dataSamplingAlgo = DEFAULT_dataSamplingAlgo;

#define COMMENT_minTotObsNum "Min number of transitions in training buffer \
before training starts. If unset we use maxTotObsNum."
#define DEFAULT_minTotObsNum 0
  Uint minTotObsNum = DEFAULT_minTotObsNum;

#define COMMENT_maxTotObsNum "Max number of transitions in training buffer."
#define DEFAULT_maxTotObsNum 262144
  Uint maxTotObsNum = DEFAULT_maxTotObsNum;

#define COMMENT_totNumSteps "Number of gradient steps before end of learning"
#define DEFAULT_totNumSteps 10000000
  Uint totNumSteps = DEFAULT_totNumSteps;

#define COMMENT_bSampleSequences "Whether to sample sequences (1) \
or observations (0) from the Replay Memory."
#define DEFAULT_bSampleSequences  false
  bool bSampleSequences = DEFAULT_bSampleSequences;

#define COMMENT_saveFreq "Frequency of checkpoints of learner state. These \
checkpoints can be used to evaluate learners, but not yet to restart learning."
#define DEFAULT_saveFreq 200000
  Uint saveFreq = DEFAULT_saveFreq;

#define COMMENT_restart "Prefix of net save files. If 'none' then no restart."
#define DEFAULT_restart "none"
std::string restart = DEFAULT_restart;

///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO NETWORK: CAPITAL LETTER
///////////////////////////////////////////////////////////////////////////////

#define COMMENT_encoderLayerSizes "Sizes of non-convolutional encoder layers \
(LSTM/RNN/FFNN). E.g. '128 128'."
#define DEFAULT_encoderLayerSizes std::vector<Uint>(0)
  std::vector<Uint> encoderLayerSizes = DEFAULT_encoderLayerSizes;

#define COMMENT_nnLayerSizes "Sizes of non-convolutional layers \
(LSTM/RNN/FFNN). E.g. '128 128'."
#define DEFAULT_nnLayerSizes std::vector<Uint>(0)
  std::vector<Uint> nnLayerSizes = DEFAULT_nnLayerSizes;

#define COMMENT_batchSize "Network training batch size."
#define DEFAULT_batchSize 256
  Uint batchSize = DEFAULT_batchSize;

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
  Uint ESpopSize = DEFAULT_ESpopSize;

#define COMMENT_nnType "Type of non-output layers read from settings. (RNN, \
LSTM, everything else maps to FFNN). Conv2D layers need to be built in \
environment directly."
#define DEFAULT_nnType "FFNN"
  std::string nnType = DEFAULT_nnType;

#define COMMENT_outWeightsPrefac "Output weights initialization factor (will \
be multiplied by default fan-in factor). Picking 1 leads to treating \
output layers with normal Xavier initialization."
#define DEFAULT_outWeightsPrefac 0.1
  Real outWeightsPrefac = DEFAULT_outWeightsPrefac;

#define COMMENT_nnLambda "Penalization factor for network weights. It will be \
multiplied by learn rate: w -= eta * nnLambda * w . L1 decay option in Bund.h"
#define DEFAULT_nnLambda std::numeric_limits<float>::epsilon()
  Real nnLambda = DEFAULT_nnLambda;

#define COMMENT_nnBPTTseq "Number of previous steps considered by RNN."
#define DEFAULT_nnBPTTseq 16
  Uint nnBPTTseq = DEFAULT_nnBPTTseq;

  ///////////////////////////////////////////////////////////////////////////////
  //SETTINGS THAT ARE NOT READ FROM FILE
  ///////////////////////////////////////////////////////////////////////////////
  // rank-local data-acquisition goals:
  Uint batchSize_local = 0;
  Real obsPerStep_local = 0;
  Uint minTotObsNum_local = 0;
  Uint maxTotObsNum_local = 0;
  // whether Recurrent network (figured out in main)
  bool bRecurrent = false;
};

} // end namespace smarties
#endif // smarties_Settings_h
