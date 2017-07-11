/*
 *  Settings.h
 *  rl
 *
 *  Created by Dmitry Alexeev and extended by Guido Novati on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#pragma once
#include <getopt.h>
#include "ArgumentParser.h"
#include "Warnings.h"

struct Settings
{
	Settings() {}
	~Settings() {}

//To modify from default value any of these settings, run executable with either
//- ascii symbol of the setting (CHARARG) followed by the value (ie. -# $value)
//- the name of the setting variable followed by the value (ie. -setting $value)

///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO ENVIRONMENT: NUMBER
///////////////////////////////////////////////////////////////////////////////
#define CHARARG_rType '1'
#define COMMENT_rType "Reward type (can be defined by user in the environment)."
#define TYPEVAL_rType int
#define TYPENUM_rType INT
#define DEFAULT_rType 0
	int rType = DEFAULT_rType;

#define CHARARG_senses '2'
#define COMMENT_senses "Perceptions allowed to agent (can be defined by user in the environment)."
#define TYPEVAL_senses int
#define TYPENUM_senses INT
#define DEFAULT_senses 0
	int senses = DEFAULT_senses;

#define CHARARG_goalDY '3'
#define COMMENT_goalDY "Parameter (can be defined by user in the environment)."
#define TYPEVAL_goalDY Real
#define TYPENUM_goalDY REAL
#define DEFAULT_goalDY 0
	Real goalDY = DEFAULT_goalDY;

#define CHARARG_factory '4'
#define COMMENT_factory "Location of factory file."
#define TYPEVAL_factory string
#define TYPENUM_factory STRING
#define DEFAULT_factory "factory"
	string factory = DEFAULT_factory;

#define CHARARG_filePrefix '5'
#define COMMENT_filePrefix "Unused (?)."
#define TYPEVAL_filePrefix string
#define TYPENUM_filePrefix STRING
#define DEFAULT_filePrefix "./"
	string filePrefix = DEFAULT_filePrefix;

///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO PARALLELIZATION/COMMUNICATION: ASCII SYMBOL
///////////////////////////////////////////////////////////////////////////////
#define CHARARG_nThreads '#'
#define COMMENT_nThreads "Number of threads from threaded training on each master rank."
#define TYPEVAL_nThreads int
#define TYPENUM_nThreads INT
#define DEFAULT_nThreads 1
	int nThreads = DEFAULT_nThreads;

#define CHARARG_nMasters '$'
#define COMMENT_nMasters "Number of master ranks (policy-updating ranks)."
#define TYPEVAL_nMasters int
#define TYPENUM_nMasters INT
#define DEFAULT_nMasters 1
	int nMasters = DEFAULT_nMasters;

#define CHARARG_isServer '!'
#define COMMENT_isServer "Whether smarties launches environment app (=1) or is launched by it (=0) (then cannot train)."
#define TYPEVAL_isServer int
#define TYPENUM_isServer INT
#define DEFAULT_isServer 1
	int isServer = DEFAULT_isServer;

#define CHARARG_sockPrefix '@'
#define COMMENT_sockPrefix "Prefix for communication file over sockets."
#define TYPEVAL_sockPrefix int
#define TYPENUM_sockPrefix INT
#define DEFAULT_sockPrefix 0
	int sockPrefix = DEFAULT_sockPrefix;

///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO LEARNING ALGORITHM: lowercase LETTER
///////////////////////////////////////////////////////////////////////////////
#define CHARARG_learner 'a'
#define COMMENT_learner "Algorithm."
#define TYPEVAL_learner string
#define TYPENUM_learner STRING
#define DEFAULT_learner "NFQ"
	string learner = DEFAULT_learner;

#define CHARARG_bTrain 'b'
#define COMMENT_bTrain "Whether training a policy (=1) or evaluating (=0)."
#define TYPEVAL_bTrain int
#define TYPENUM_bTrain INT
#define DEFAULT_bTrain 1
	int bTrain = DEFAULT_bTrain;

#define CHARARG_appendedObs 'c'
#define COMMENT_appendedObs "Number of past observations to be chained together to form policy input (eg. see frames in DQN paper)."
#define TYPEVAL_appendedObs int
#define TYPENUM_appendedObs INT
#define DEFAULT_appendedObs 0
	int appendedObs = DEFAULT_appendedObs;

#define CHARARG_targetDelay 'd'
#define COMMENT_targetDelay "Copy delay for target network. If >1: every # grad desc steps tgt-net copies curr weigths. If <1: every grad desc step tgt-net does exp averaging."
#define TYPEVAL_targetDelay Real
#define TYPENUM_targetDelay REAL
#define DEFAULT_targetDelay 1000
	Real targetDelay = DEFAULT_targetDelay;

#define CHARARG_greedyEps 'e'
#define COMMENT_greedyEps "Rate of random actions (detail depend on chosen learning algorithm)."
#define TYPEVAL_greedyEps Real
#define TYPENUM_greedyEps REAL
#define DEFAULT_greedyEps 0.1
	Real greedyEps = DEFAULT_greedyEps;

#define CHARARG_saveFreq 'f'
#define COMMENT_saveFreq "Freq in # of comms with slaves for the master to save the policy."
#define TYPEVAL_saveFreq int
#define TYPENUM_saveFreq INT
#define DEFAULT_saveFreq 10000
	int saveFreq = DEFAULT_saveFreq;

#define CHARARG_gamma 'g'
#define COMMENT_gamma "Discount factor."
#define TYPEVAL_gamma Real
#define TYPENUM_gamma REAL
#define DEFAULT_gamma 0.99
	Real gamma = DEFAULT_gamma;

#define CHARARG_samplesFile 'h'
#define COMMENT_samplesFile "Name of main transition data backup file."
#define TYPEVAL_samplesFile string
#define TYPENUM_samplesFile STRING
#define DEFAULT_samplesFile "history.txt"
	string samplesFile = DEFAULT_samplesFile;

#define CHARARG_randSeed 'i'
#define COMMENT_randSeed "Random seed."
#define TYPEVAL_randSeed int
#define TYPENUM_randSeed INT
#define DEFAULT_randSeed 1
	int randSeed = DEFAULT_randSeed;

#define CHARARG_lambda 'l'
#define COMMENT_lambda "Currently unused: lambda for off policy corrections."
#define TYPEVAL_lambda Real
#define TYPENUM_lambda REAL
#define DEFAULT_lambda 0
	Real lambda = DEFAULT_lambda;

#define CHARARG_minSeqLen 'm'
#define COMMENT_minSeqLen "Minimum length of sequence, if shorter then discarded."
#define TYPEVAL_minSeqLen int
#define TYPENUM_minSeqLen INT
#define DEFAULT_minSeqLen 3
	int minSeqLen = DEFAULT_minSeqLen;

#define CHARARG_maxSeqLen 'M' //there is always an exception
#define COMMENT_maxSeqLen "Maximum length of sequence. if seq is longer it is just split into segments."
#define TYPEVAL_maxSeqLen int
#define TYPENUM_maxSeqLen INT
#define DEFAULT_maxSeqLen 9000
	int maxSeqLen = DEFAULT_maxSeqLen;

#define CHARARG_bNormalize 'n'
#define COMMENT_bNormalize "Whether state should be normalized before feeding it into network (=1). Scale and mean are read from environment or if not defined there they are computed from history buffer."
#define TYPEVAL_bNormalize int
#define TYPENUM_bNormalize INT
#define DEFAULT_bNormalize 1
	int bNormalize = DEFAULT_bNormalize;

#define CHARARG_obsPerStep 'o'
#define COMMENT_obsPerStep "Minimum ratio of observed *sequences* to gradient steps. 0.1 means that for every terminal state, learner does 10 gradient steps."
#define TYPEVAL_obsPerStep  Real
#define TYPENUM_obsPerStep  REAL
#define DEFAULT_obsPerStep  0.1
	Real obsPerStep = DEFAULT_obsPerStep;

#define CHARARG_restart 'p'
#define COMMENT_restart "File prefix of policy."
#define TYPEVAL_restart string
#define TYPENUM_restart STRING
#define DEFAULT_restart "policy"
	string restart = DEFAULT_restart;

#define CHARARG_epsAnneal 'r'
#define COMMENT_epsAnneal "Annealing rate in grad steps of various learning-algorithm-dependent behaviors."
#define TYPEVAL_epsAnneal Real
#define TYPENUM_epsAnneal REAL
#define DEFAULT_epsAnneal 1e4
	Real epsAnneal = DEFAULT_epsAnneal;

#define CHARARG_bSampleSequences 's'
#define COMMENT_bSampleSequences "Whether to sample sequences or trajectories."
#define TYPEVAL_bSampleSequences  int
#define TYPENUM_bSampleSequences  INT
#define DEFAULT_bSampleSequences  0
	int bSampleSequences = DEFAULT_bSampleSequences;

#define CHARARG_maxTotSeqNum 't'
#define COMMENT_maxTotSeqNum "Maximum number of samples in history buffer"
#define TYPEVAL_maxTotSeqNum int
#define TYPENUM_maxTotSeqNum INT
#define DEFAULT_maxTotSeqNum 5000
	int maxTotSeqNum = DEFAULT_maxTotSeqNum;

#define CHARARG_totNumSteps 'z'
#define COMMENT_totNumSteps "Number of gradient steps before end of learning"
#define TYPEVAL_totNumSteps int
#define TYPENUM_totNumSteps INT
#define DEFAULT_totNumSteps 10000000
	int totNumSteps = DEFAULT_totNumSteps;



///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO NETWORK: CAPITAL LETTER
///////////////////////////////////////////////////////////////////////////////
#define CHARARG_nnl1 'Z'
#define COMMENT_nnl1 "Size of first non-convolutional layer (LSTM/RNN/FFNN)."
#define TYPEVAL_nnl1 int
#define TYPENUM_nnl1 INT
#define DEFAULT_nnl1 0
	int nnl1 = DEFAULT_nnl1;

#define CHARARG_nnl2 'Y'
#define COMMENT_nnl2 "Size of second non-convolutional layer (LSTM/RNN/FFNN)."
#define TYPEVAL_nnl2 int
#define TYPENUM_nnl2 INT
#define DEFAULT_nnl2 0
	int nnl2 = DEFAULT_nnl2;

#define CHARARG_nnl3 'X'
#define COMMENT_nnl3 "Size of third non-convolutional layer (LSTM/RNN/FFNN)."
#define TYPEVAL_nnl3 int
#define TYPENUM_nnl3 INT
#define DEFAULT_nnl3 0
	int nnl3 = DEFAULT_nnl3;

#define CHARARG_nnl4 'W'
#define COMMENT_nnl4 "Size of fourth non-convolutional layer (LSTM/RNN/FFNN)."
#define TYPEVAL_nnl4 int
#define TYPENUM_nnl4 INT
#define DEFAULT_nnl4 0
	int nnl4 = DEFAULT_nnl4;

#define CHARARG_nnl5 'V'
#define COMMENT_nnl5 "Size of fifth non-convolutional layer (LSTM/RNN/FFNN)."
#define TYPEVAL_nnl5 int
#define TYPENUM_nnl5 INT
#define DEFAULT_nnl5 0
	int nnl5 = DEFAULT_nnl5;

#define CHARARG_nnl6 'U'
#define COMMENT_nnl6 "Size of sixth non-convolutional layer (LSTM/RNN/FFNN)."
#define TYPEVAL_nnl6 int
#define TYPENUM_nnl6 INT
#define DEFAULT_nnl6 0
	int nnl6 = DEFAULT_nnl6;

#define CHARARG_splitLayers 'S'
#define COMMENT_splitLayers "Number of split layers, description in Settings.h"
//"For each output required by algorithm (ie. value, policy, std, ...) " \/
//"how many non-conv layers should be devoted only to one o the outputs. " \/
//"For example if there 2 FF layers of size Z and Y and this arg is set to 1,"\/
//" then each of the outputs is connected to a separate layer of size Y. " \/
//"Each of the Y-size layers are then connected to the first layer of size Z."
#define TYPEVAL_splitLayers int
#define TYPENUM_splitLayers INT
#define DEFAULT_splitLayers 0
	int splitLayers = DEFAULT_splitLayers;

#define CHARARG_outWeightsPrefac 'O'
#define COMMENT_outWeightsPrefac "Output weights initialization factor (will be multiplied by default fan-in factor). Picking 1 leads to treating output layers with normal initialization."
#define TYPEVAL_outWeightsPrefac Real
#define TYPENUM_outWeightsPrefac REAL
#define DEFAULT_outWeightsPrefac 0.01
	Real outWeightsPrefac = DEFAULT_outWeightsPrefac;

#define CHARARG_batchSize 'B'
#define COMMENT_batchSize "Network training batch size."
#define TYPEVAL_batchSize int
#define TYPENUM_batchSize INT
#define DEFAULT_batchSize 32
	int batchSize = DEFAULT_batchSize;

#define CHARARG_learnrate 'L'
#define COMMENT_learnrate "Learning rate."
#define TYPEVAL_learnrate Real
#define TYPENUM_learnrate REAL
#define DEFAULT_learnrate 1e-4
	Real learnrate = DEFAULT_learnrate;

#define CHARARG_nnPdrop 'D'
#define COMMENT_nnPdrop "Unused currently (dropout)."
#define TYPEVAL_nnPdrop Real
#define TYPENUM_nnPdrop REAL
#define DEFAULT_nnPdrop 0
	Real nnPdrop = DEFAULT_nnPdrop;

#define CHARARG_nnLambda 'P'
#define COMMENT_nnLambda "Penalization factor for network weights."
#define TYPEVAL_nnLambda Real
#define TYPENUM_nnLambda REAL
#define DEFAULT_nnLambda 0
	Real nnLambda = DEFAULT_nnLambda;

#define CHARARG_nnType 'N'
#define COMMENT_nnType "Type of non-output layers read from settings. (RNN, LSTM, everything else maps to FFNN). Conv2D layers need to be built in environment directly."
#define TYPEVAL_nnType string
#define TYPENUM_nnType STRING
#define DEFAULT_nnType "FFNN"
	string nnType = DEFAULT_nnType;

#define CHARARG_nnFunc 'F'
#define COMMENT_nnFunc "Activation function for non-output layers (which are always linear) which are built from settings. (Relu, Tanh, Sigm, PRelu, softSign, softPlus, ...)"
#define TYPEVAL_nnFunc string
#define TYPENUM_nnFunc STRING
#define DEFAULT_nnFunc "PRelu"
	string nnFunc = DEFAULT_nnFunc;

///////////////////////////////////////////////////////////////////////////////
//SETTINGS THAT ARE NOT READ FROM FILE
///////////////////////////////////////////////////////////////////////////////
	int world_rank = 0;
	int world_size = 0;
	int slaves_rank = 0;
	int slaves_size = 0;
	int learner_rank = 0;
	int learner_size = 0;
	// number of slaves (usually per master)
	int nSlaves = 1;
	//number of agents that:
	// in case of slave: are contained in an environment
	// in case of master: nSlaves * are contained in an environment
	int nAgents = -1;
	// whether Recurrent network (figured out in main)
	bool bRecurrent = false;
	// number of inputs of the policy, depends on env and learning algorithm
	int nnInputs = -1;
	int nnOutputs = -1;

	//random number generators (one per thread)
	//std::mt19937* gen;
	std::vector<std::mt19937> generators;

	vector<ArgumentParser::OptionStruct> initializeOpts ()
  { //  //{ CHARARG_, "", TYPENUM_, COMMENT_, &, (TYPEVAL_) DEFAULT_ },
		//AVERT YOUR EYES!
		return vector<ArgumentParser::OptionStruct> ({
			{ CHARARG_splitLayers, "splitLayers", TYPENUM_splitLayers, COMMENT_splitLayers, &splitLayers, (TYPEVAL_splitLayers) DEFAULT_splitLayers },
			{ CHARARG_outWeightsPrefac, "outWeightsPrefac", TYPENUM_outWeightsPrefac, COMMENT_outWeightsPrefac, &outWeightsPrefac, (TYPEVAL_outWeightsPrefac) DEFAULT_outWeightsPrefac },
			{ CHARARG_rType, "rType", TYPENUM_rType, COMMENT_rType, &rType, (TYPEVAL_rType) DEFAULT_rType },
			{ CHARARG_senses, "senses", TYPENUM_senses, COMMENT_senses, &senses, (TYPEVAL_senses) DEFAULT_senses },
			{ CHARARG_goalDY, "goalDY", TYPENUM_goalDY, COMMENT_goalDY, &goalDY, (TYPEVAL_goalDY) DEFAULT_goalDY },
			{ CHARARG_factory, "factory", TYPENUM_factory, COMMENT_factory, &factory, (TYPEVAL_factory) DEFAULT_factory },
			{ CHARARG_filePrefix, "filePrefix", TYPENUM_filePrefix, COMMENT_filePrefix, &filePrefix, (TYPEVAL_filePrefix) DEFAULT_filePrefix },
			{ CHARARG_nThreads, "nThreads", TYPENUM_nThreads, COMMENT_nThreads, &nThreads, (TYPEVAL_nThreads) DEFAULT_nThreads },
			{ CHARARG_nMasters, "nMasters", TYPENUM_nMasters, COMMENT_nMasters, &nMasters, (TYPEVAL_nMasters) DEFAULT_nMasters },
			{ CHARARG_isServer, "isServer", TYPENUM_isServer, COMMENT_isServer, &isServer, (TYPEVAL_isServer) DEFAULT_isServer },
			{ CHARARG_sockPrefix, "sockPrefix", TYPENUM_sockPrefix, COMMENT_sockPrefix, &sockPrefix, (TYPEVAL_sockPrefix) DEFAULT_sockPrefix },
			{ CHARARG_appendedObs, "appendedObs", TYPENUM_appendedObs, COMMENT_appendedObs, &appendedObs, (TYPEVAL_appendedObs) DEFAULT_appendedObs },
			{ CHARARG_maxSeqLen, "maxSeqLen", TYPENUM_maxSeqLen, COMMENT_maxSeqLen, &maxSeqLen, (TYPEVAL_maxSeqLen) DEFAULT_maxSeqLen },
			{ CHARARG_minSeqLen, "minSeqLen", TYPENUM_minSeqLen, COMMENT_minSeqLen, &minSeqLen, (TYPEVAL_minSeqLen) DEFAULT_minSeqLen },
			{ CHARARG_maxTotSeqNum, "maxTotSeqNum", TYPENUM_maxTotSeqNum, COMMENT_maxTotSeqNum, &maxTotSeqNum, (TYPEVAL_maxTotSeqNum) DEFAULT_maxTotSeqNum },
			{ CHARARG_bTrain, "bTrain", TYPENUM_bTrain, COMMENT_bTrain, &bTrain, (TYPEVAL_bTrain) DEFAULT_bTrain },
			{ CHARARG_bNormalize, "bNormalize", TYPENUM_bNormalize, COMMENT_bNormalize, &bNormalize, (TYPEVAL_bNormalize) DEFAULT_bNormalize },
			{ CHARARG_saveFreq, "saveFreq", TYPENUM_saveFreq, COMMENT_saveFreq, &saveFreq, (TYPEVAL_saveFreq) DEFAULT_saveFreq },
			{ CHARARG_randSeed, "randSeed", TYPENUM_randSeed, COMMENT_randSeed, &randSeed, (TYPEVAL_randSeed) DEFAULT_randSeed },
			{ CHARARG_epsAnneal, "epsAnneal", TYPENUM_epsAnneal, COMMENT_epsAnneal, &epsAnneal, (TYPEVAL_epsAnneal) DEFAULT_epsAnneal },
			{ CHARARG_greedyEps, "greedyEps", TYPENUM_greedyEps, COMMENT_greedyEps, &greedyEps, (TYPEVAL_greedyEps) DEFAULT_greedyEps },
			{ CHARARG_targetDelay, "targetDelay", TYPENUM_targetDelay, COMMENT_targetDelay, &targetDelay, (TYPEVAL_targetDelay) DEFAULT_targetDelay },
			{ CHARARG_gamma, "gamma", TYPENUM_gamma, COMMENT_gamma, &gamma, (TYPEVAL_gamma) DEFAULT_gamma },
			{ CHARARG_lambda, "lambda", TYPENUM_lambda, COMMENT_lambda, &lambda, (TYPEVAL_lambda) DEFAULT_lambda },
			{ CHARARG_learner, "learner", TYPENUM_learner, COMMENT_learner, &learner, (TYPEVAL_learner) DEFAULT_learner },
			{ CHARARG_restart, "restart", TYPENUM_restart, COMMENT_restart, &restart, (TYPEVAL_restart) DEFAULT_restart },
			{ CHARARG_obsPerStep, "obsPerStep", TYPENUM_obsPerStep, COMMENT_obsPerStep, &obsPerStep, (TYPEVAL_obsPerStep) DEFAULT_obsPerStep },
			{ CHARARG_totNumSteps, "totNumSteps", TYPENUM_totNumSteps, COMMENT_totNumSteps, &totNumSteps, (TYPEVAL_totNumSteps) DEFAULT_totNumSteps },
			{ CHARARG_samplesFile, "samplesFile", TYPENUM_samplesFile, COMMENT_samplesFile, &samplesFile, (TYPEVAL_samplesFile) DEFAULT_samplesFile },
			{ CHARARG_bSampleSequences, "bSampleSequences", TYPENUM_bSampleSequences, COMMENT_bSampleSequences, &bSampleSequences, (TYPEVAL_bSampleSequences) DEFAULT_bSampleSequences },
			{ CHARARG_nnl1, "nnl1", TYPENUM_nnl1, COMMENT_nnl1, &nnl1, (TYPEVAL_nnl1) DEFAULT_nnl1 },
			{ CHARARG_nnl2, "nnl2", TYPENUM_nnl2, COMMENT_nnl2, &nnl2, (TYPEVAL_nnl2) DEFAULT_nnl2 },
			{ CHARARG_nnl3, "nnl3", TYPENUM_nnl3, COMMENT_nnl3, &nnl3, (TYPEVAL_nnl3) DEFAULT_nnl3 },
			{ CHARARG_nnl4, "nnl4", TYPENUM_nnl4, COMMENT_nnl4, &nnl4, (TYPEVAL_nnl4) DEFAULT_nnl4 },
			{ CHARARG_nnl5, "nnl5", TYPENUM_nnl5, COMMENT_nnl5, &nnl5, (TYPEVAL_nnl5) DEFAULT_nnl5 },
			{ CHARARG_nnl6, "nnl6", TYPENUM_nnl6, COMMENT_nnl6, &nnl6, (TYPEVAL_nnl6) DEFAULT_nnl6 },
			{ CHARARG_batchSize, "batchSize", TYPENUM_batchSize, COMMENT_batchSize, &batchSize, (TYPEVAL_batchSize) DEFAULT_batchSize },
			{ CHARARG_learnrate, "learnrate", TYPENUM_learnrate, COMMENT_learnrate, &learnrate, (TYPEVAL_learnrate) DEFAULT_learnrate },
			{ CHARARG_nnPdrop, "nnPdrop", TYPENUM_nnPdrop, COMMENT_nnPdrop, &nnPdrop, (TYPEVAL_nnPdrop) DEFAULT_nnPdrop },
			{ CHARARG_nnLambda, "nnLambda", TYPENUM_nnLambda, COMMENT_nnLambda, &nnLambda, (TYPEVAL_nnLambda) DEFAULT_nnLambda },
			{ CHARARG_nnType, "nnType", TYPENUM_nnType, COMMENT_nnType, &nnType, (TYPEVAL_nnType) DEFAULT_nnType },
			{ CHARARG_nnFunc, "nnFunc", TYPENUM_nnFunc, COMMENT_nnFunc, &nnFunc, (TYPEVAL_nnFunc) DEFAULT_nnFunc }
		});
  }

	vector<int> readNetSettingsSize()
  {
		vector<int> ret;
		if(nnl1<1) die("Add at least one hidden layer.\n");
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
		return ret;
  }
};
