/*
 *  main.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

#include "ArgumentParser.h"
#include "ErrorHandling.h"
#include "Learners/QLearning.h"
#include "Learners/Sarsa.h"
#include "Learners/Learner.h"
#include "Learners/NFQ.h"
#include "Learners/Explorer.h"
#include "ObjectFactory.h"
#include "Settings.h"
#include "Scheduler/Scheduler.h"
#include "Savers/AllSavers.h"

using namespace ErrorHandling;
using namespace ArgumentParser;
using namespace std;

void runTest(void);
void runSlave(int);
Settings settings;
int ErrorHandling::debugLvl;

// Runs the simulation
void runSlave(int rank)
{
    ObjectFactory factory(settings);
    Environment* env = factory.createEnvironment(rank, 0);
    Learner* learner;
    
    int nAgents = env->agents.size();
    
    settings.nAgents = nAgents;
    settings.nSlaves = 1;
    
    //TODO:
    if (settings.learner == "Q")
        learner = new QLearning(env, settings);
    else if (settings.learner == "DQ")
        learner = new NFQ(env, settings);
    Slave* simulation = new Slave(env, rank, settings);
    simulation->learner = learner;
    //TODO
    
    
    if (settings.restart != "none")
        simulation->restart(settings.restart);
    Real time = 0;
    
    while (true)
        simulation->evolve(time);
	
    exit(0);
}

void runMaster(int nranks)
{
    // TODO: No need to create a whole system, just need actInfo and sInfo
    ObjectFactory factory(settings);
    Environment* env = factory.createEnvironment(0,0);
    Learner* learner;
    
    int nAgents = (nranks)*env->agents.size();
    
    settings.nAgents = nAgents;
    settings.nSlaves = nranks;
    
    if (settings.learner == "Q")
        learner = new QLearning(env, settings);
    else if (settings.learner == "DQ")
        learner = new NFQ(env, settings);
    
    Master* master = new Master(learner, env, settings);
    
    if (settings.restart != "none")
        master->restart(settings.restart);
    
    // Save results to dir named  settings.prefix
    if (!Saver::makedir((settings.prefix+"/").c_str())) die("Unable to make a working directory!");

    RewardSaver* rsaver = new RewardSaver((ofstream*)&cout, 1000);//"reward.txt");
    master->registerSaver(rsaver);

    StateSaver* ssaver = new StateSaver("state.txt", 1000);
    
    master->run();
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
    //cout << clock.tv_usec << endl;
    int seed = abs(84967194 + floor(clock.tv_usec));
    debugLvl=12;
    vector<OptionStruct> opts ({
    {'c', "config",   STRING,"config file",    &settings.configFile,(string)"factory"},
    {'t', "dt",       REAL,  "Sim timestep",   &settings.dt,        (Real)0.001},
    {'f', "end_time", REAL,  "End time of sim",&settings.endTime,   (Real)1e9},
    {'g', "gamma",    REAL,  "Gamma parameter",&settings.gamma,     (Real)0.99},
    {'E', "endRew",   REAL,  "Terminal reward",&settings.EndR,      (Real)-9.95},
    {'e', "greedyeps",REAL,  "Greedy epsilon", &settings.greedyEps, (Real)0.001},
    {'l', "learnrate",REAL,  "Learning rate",  &settings.lRate,     (Real)0.1},
    {'d', "lambda",   REAL,  "Lambda",         &settings.lambda,    (Real)0.0},
    {'s', "rand_seed",INT,   "Random seed",    &settings.randSeed,  (int)seed},
    {'r', "restart",  STRING,"Restart",        &settings.restart,   (string)"policy"},
    {'q', "save_freq",INT,   "Save frequency", &settings.saveFreq,  (int)1000},
    {'v', "debug_lvl",INT,   "Debug level",    &debugLvl,           (int)4},
    {'p', "prefix",   STRING,"Save folder",    &settings.prefix,    (string)"./"},
    {'H', "nne",      REAL,  "NN's eta",       &settings.nnEta,     (Real)0.0001},
    {'A', "nna",      REAL,  "NN's alpha",     &settings.nnAlpha,   (Real)0.5},
    {'K', "nnL",      REAL,  "Weight decay",   &settings.nnLambda,  (Real)0.0001},
    {'D', "nnD",      REAL,  "NN's droput",    &settings.nnPdrop,   (Real)0.0},
    {'N', "nnl1",     INT,   "NN norm layer 1",&settings.nnLayer1,  (int)0},
    {'L', "nnl2",     INT,   "NN norm layer 2",&settings.nnLayer2,  (int)0},
    {'W', "nnl3",     INT,   "NN norm layer 3",&settings.nnLayer3,  (int)0},
    {'Y', "nnm1",     INT,   "NN LSTM layer 1",&settings.nnMemory1, (int)0},
    {'Z', "nnm2",     INT,   "NN LSTM layer 2",&settings.nnMemory2, (int)0},
    {'X', "nnm3",     INT,   "NN LSTM layer 3",&settings.nnMemory3, (int)0},
    {'Q', "learn",    STRING,"Learner Type",   &settings.learner,   (string)"DQ"},
    {'P', "approx",   STRING,"Approximator",   &settings.approx,    (string)"NN"},
    {'R', "rType",    INT,   "Reward: ef,ef,y",&settings.rewardType,(int)0},
    {'G', "goalDY",   REAL,  "If r==2  goalDY",&settings.goalDY,    (Real)0.0},
    {'T', "bTrain",   INT,   "am I training?", &settings.bTrain,    (int)1},
    {'S', "senses",   INT,   "top,pov,vel,pres",&settings.senses,   (int)0}
    }
    );
    
    Parser parser(opts);
    parser.parse(argc, argv, rank == 0);
	if (settings.lRate < 1e-9) settings.immortal = false;
	else                       settings.immortal = true;
    
    if  (settings.randSeed == -1 )  srand(time(NULL));
	else							srand(settings.randSeed + rank);
	
    if (rank == 0) runMaster(nranks);
    else           runSlave(rank);
    
	return 0;
}
