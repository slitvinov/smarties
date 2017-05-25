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
#include <string>
#include <random>

#include "ArgumentParser.h"
#include "Warnings.h"

struct Settings
{
    Settings() {}
    ~Settings() {}

    //read from from settings file input
      // defined in a case by case manner in environment:
        int rewardType = 0;
        int senses = 0;
        Real goalDY = 0;
      //number of threads from threaded training on each master rank
        int nThreads = -1;
      //network shape for the non conv-2d layers
        int nnLayer1 = -1;
        int nnLayer2 = -1;
        int nnLayer3 = -1;
        int nnLayer4 = -1;
        int nnLayer5 = -1;
      //state and data handling
        // if >0 then chain # past observations together to form policy input
        // (eg. see frames in DQN paper):
        int dqnAppendS = 0;
        // batch size for policy training:
        int dqnBatch = 1;
        //maximum length of sequence, if too long BPTT might be unfeasible
        // if seq is longer it is just split into segments
        int maxSeqLen = 1000;
        //minimum length of sequence, if shorter than discarded
        int minSeqLen = 3;
        //size of history buffer for experience replay
        int maxTotSeqNum = 5000;
        // learning rate
        Real lRate = 0;
        // unused currently (dropout)
        Real nnPdrop = 0;
        // penalization factor for network weights
        Real nnLambda = 0;
        // copy delay for target network
        // if >1 then every # grad desc steps target network copies current weigths
        // if <1 then every grad desc step target network does exp averaging
        Real dqnUpdateC = 1000;
        // rate of random actions (detail depend on chosen learning algorithm)
        Real greedyEps = 0.1;
        // discount factor
        Real gamma = 0.9;
        // currently unused: lambda of off policy corrections
        Real lambda = 0;
        // annealing rate of various parameters of learning algorithm
        Real epsAnneal = 1e4;
        // algorithm
        string learner = "NFQ";
        // name prefix of policy
        string restart = "policy";
        // name of factory file
        string configFile = "factory";
        // i dont know:
        string prefix = "./";
        // name of main transition data backup file
        string samplesFile = "history.txt";
        // network type, it accepts either "RNN", or "LSTM", any other it assumes FFNN
        // conv2D layers need to be built in environment directly
        string netType = "Feedforward";
        // activation function of network
        string funcType = "PRelu";
        // whether state should be normalized before feeding it into network
        // either by contunuous update of mean and std of data
        // or if provided in environment by scale and mean defined in State struct
        bool normalizeInput = true;
        // whether training or evaluating (0)
        int bTrain = 1;

    //written automatically from environment and mpi config
      //number of agents that a master or a slave needs to deal with
      int nAgents = -1;
      // number of slaves (usually per master)
      int nSlaves = -1;
      //number of masters ranks
      int nMasters = 1;
      //whether smarties launches app or is launched by it (client script)
      int isLauncher = 1;
      //prefix for communication over sockets
      int sockPrefix = 0;
      // whether master
      bool bIsMaster = true;
      // whether Recurrent network (figured out in main)
      bool bRecurrent = false;
      // number of inputs of the policy, depends on env and learning algorithm
      int nnInputs = -1;
      int nnOutputs = -1;
      //whether have separate output layers... depends on learner algorithm
      int separateOutputs = 1;

    //misc
        //freq in # of comms with slaves for the master to save the policy
        int saveFreq = 1e4;
        int randSeed = 0;
        //random number generators (one per thread)
        //std::mt19937* gen;
        std::vector<std::mt19937> generators;

    vector<ArgumentParser::OptionStruct> initializeOpts ()
    {
      return vector<ArgumentParser::OptionStruct> ({
      		{'N',"nMasters", INT,
      			"number of masters (policy-updating ranks)",
      			&nMasters, (int)1},
      		{'g',"gamma",    REAL,
      				"Gamma discount factor parameter",
      				&gamma,     (Real)0.9},
      		{'e',"greedyeps",REAL,
      				"Fraction of actions chosen randomly",
      				&greedyEps, (Real)0.1},
      		{'E',"epsAnneal",REAL,
      				"Number of grad steps for annealing of learning-algorithm-dependent behaviors",
      				&epsAnneal, (Real)1e4},
      		{'l',"learnrate",REAL,
      				"Network learning rate",
      				&lRate,     (Real)0.001},
      		{'a',"learn",    STRING,
      				"RL algorithm",
      				&learner,   (string)"DQ"},
      		{'r',"rType",    INT,
      				"Reward: see env",
      				&rewardType,(int)-1},
      		{'i',"senses",   INT,
      				"State: see env",
      				&senses,   (int)0},
      		{'y',"goalDY",   REAL,
      				"goalDY: see env",
      				&goalDY,    (Real)0.},
      		{'t',"bTrain",   INT,
      				"Whether training (1) or evaluating a policy (0)",
      				&bTrain,    (int)1},
      		{'K',"nnL",      REAL,
      				"Network's weight decay",
      				&nnLambda,  (Real)0.0},
      		{'Z',"nnl1",     INT,
      				"NN layer 1",
      				&nnLayer1,  (int)0},
      		{'Y',"nnl2",     INT,
      				"NN layer 2",
      				&nnLayer2,  (int)0},
      		{'X',"nnl3",     INT,
      				"NN layer 3",
      				&nnLayer3,  (int)0},
      		{'W',"nnl4",     INT,
      				"NN layer 4",
      				&nnLayer4,  (int)0},
      		{'V',"nnl5",     INT,
      				"NN layer 5",
      				&nnLayer5,  (int)0},
      		{'T',"nnType",   STRING,
      				"Network Type: LSTM, RNN, any other means Feedforward",
      				&netType,  (string)"Feedforward"},
      		{'F',"funcType",   STRING,
      				"Activation (Relu, Tanh, Linear, etc) of non-ouput layers (which are always linear)",
      				&funcType, (string)"SoftSign"},
      		{'C',"dqnT",     REAL,
      				"Delay for target network weight update",
      				&dqnUpdateC,(Real)1000},
      		{'B',"dqnBatch", INT,
      				"Network update batch size",
      				&dqnBatch,  (int)10},
      		{'A',"dqnNs",    INT,
      				"Number of previous states chained together to form NN input",
      				&dqnAppendS,(int)0},
      		{'L',"dqnSeqMax",INT,
      				"Max seq length. if greater the sequence is split",
      				&maxSeqLen, (int)200},
      		{'M',"dqnSeqMin",INT,
      				"Min seq length. if less the sequence is ignored",
      				&minSeqLen, (int)2},
      		{'U',"maxTotSeqNum",INT,
      				"Maximum number of stored sequences: if exceeded the easier ones are removed",
      				&maxTotSeqNum, (int)5000},
      		{'p',"nThreads", INT,
      				"Number of threads on master ranks",
      				&nThreads,  (int)-1},
      		{'I',"isServer", INT,
      				"Whether smarties launches apps or is launched by app (then cannot train)",
      				&isLauncher,  (int)1},
      		{'P',"sockPrefix",INT,
      				"Number prefix for socket: >0 if launched by app",
      				&sockPrefix,  (int)-1},
      		{'H',"fileSamp", STRING,
      				"Location of transitions log for restart",
      				&samplesFile,(string)"obs_master.txt"},
      		{'R',"restart", STRING,
      				"Location of policy file for restart",
      				&restart,(string)"policy"}
      	});
    }
    vector<int> readNetSettingsSize()
    {
      vector<int> ret;
      assert(nnLayer1>0);
      ret.push_back(nnLayer1);
      if (nnLayer2>0) {
        ret.push_back(nnLayer2);
        if (nnLayer3>0) {
          ret.push_back(nnLayer3);
          if (nnLayer4>0) {
            ret.push_back(nnLayer4);
            if (nnLayer5>0) {
              ret.push_back(nnLayer5);
            }
          }
        }
      }
      return ret;
    }
};
