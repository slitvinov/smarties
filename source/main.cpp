/*
 *  main.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "ArgumentParser.h"
#include "Learners/Learner.h"
#include "Learners/NFQ.h"
#include "Learners/NAF.h"
#include "ObjectFactory.h"
#include "Settings.h"
#include "Scheduler.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

using namespace ErrorHandling;
using namespace ArgumentParser;
using namespace std;

void runTest(void);
void runSlave(int);
Settings settings;
int ErrorHandling::debugLvl;

void runSlave(int rank)
{
    ObjectFactory factory(settings);
    Environment* env = factory.createEnvironment(rank, 0);
    settings.nAgents = env->agents.size();
    settings.nSlaves = 1;
    Slave * simulation = new Slave(env, rank, settings);
    if (settings.restart != "none") simulation->restart(settings.restart);
    while (true) {
        simulation->run(); //if it returns, something is messed up
        env->close_Comm();
        env->setup_Comm();
    }
}

void runMaster(int nranks)
{
    ObjectFactory factory(settings);
    Environment* env = factory.createEnvironment(0,0);

    settings.nAgents = nranks*env->agents.size();
    settings.nSlaves = nranks;
    
    Learner* learner;
    if      (settings.learner == "DQ" || settings.learner == "DQN") {
        settings.nnInputs = env->sI.dimUsed;
        settings.nnOutputs = 1;
        for (int i(0); i<env->aI.dim; i++) settings.nnOutputs*=env->aI.bounds[i];
        learner = new NFQ(env, settings);
    }
    else if (settings.learner == "NA" || settings.learner == "NAF") {
        settings.nnInputs = env->sI.dimUsed;
        const int nA = env->aI.dim;
        const int nL = (nA*nA+nA)/2;
        settings.nnOutputs = 1+nL+nA;
        learner = new NAF(env,settings);
    }
    
    Master* master = new Master(learner, env, settings);
    if (settings.restart != "none") master->restart(settings.restart);
    
    if (settings.nThreads > 1) learner->TrainTasking(master);
    else master->run();
    
}

int main (int argc, char** argv)
{
    int rank(0), nranks(2);
    
#ifndef MEGADEBUG
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif
    
    struct timeval clock;
    gettimeofday(&clock, NULL);
    int seed = abs(floor(clock.tv_usec + rank));
    debugLvl=12;
    
    vector<OptionStruct> opts ({
    {'g', "gamma",    REAL,  "Gamma parameter",&settings.gamma,     (Real)0.9},
    {'e', "greedyeps",REAL,  "Greedy epsilon", &settings.greedyEps, (Real)0.1},
    {'l', "learnrate",REAL,  "Learning rate",  &settings.lRate,     (Real)0.001},
    {'b', "debug_lvl",INT,   "Debug level",    &debugLvl,           (int)4},
    {'a', "learn",    STRING,"Learner Type",   &settings.learner,   (string)"DQ"},
    {'r', "rType",    INT,   "Reward: ef,ef,y",&settings.rewardType,(int)0},
    {'y', "goalDY",   REAL,  "If r==2  goalDY",&settings.goalDY,    (Real)0.},
    {'t', "bTrain",   INT,   "am I training?", &settings.bTrain,    (int)1},
    {'i', "senses",   INT,   "top,pov,vel,pres",&settings.senses,   (int)0},
    {'K', "nnL",      REAL,  "Weight decay",   &settings.nnLambda,  (Real)0.0},
    {'D', "nnD",      REAL,  "NN's droput",    &settings.nnPdrop,   (Real)0.0},
    {'Z', "nnl1",     INT,   "NN layer 1",     &settings.nnLayer1,  (int)0},
    {'Y', "nnl2",     INT,   "NN layer 2",     &settings.nnLayer2,  (int)0},
    {'X', "nnl3",     INT,   "NN layer 3",     &settings.nnLayer3,  (int)0},
    {'W', "nnl4",     INT,   "NN layer 4",     &settings.nnLayer4,  (int)0},
    {'V', "nnl5",     INT,   "NN layer 5",     &settings.nnLayer5,  (int)0},
    {'T', "nnType",   INT,   "NNtype: LSTM,FF",&settings.nnType,    (int)1},
    {'C', "dqnT",     REAL,  "DQN update tgt", &settings.dqnUpdateC,(Real)1000},
    {'S', "dqnNs",    INT,   "appended states",&settings.dqnAppendS,(int)0},
    {'B', "dqnBatch", INT,   "batch update",   &settings.dqnBatch,  (int)10},
    {'p', "nThreads", INT,   "parallel master",&settings.nThreads,  (int)-1},
    {'H', "fileSamp", STRING,"history file",   &settings.samplesFile,(string)"../history.txt"} });
    
    Parser parser(opts);
    parser.parse(argc, argv, rank == 0);
    
    srand(settings.randSeed + rank);
    settings.gen = new mt19937(settings.randSeed);
    
    if (rank == 0) runMaster(nranks);
    else           runSlave(rank);
    
	return 0;
}
