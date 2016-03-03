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
    ObjectFactory factory(settings.configFile);
    Environment* env = factory.createEnvironment(rank, 0);

    Slave* simulation = new Slave(env, settings.dt, rank);
    
    Real time = 0;
    
    while (true)
        simulation->evolve(time);
	
    exit(0);
}

void runMaster(int nranks)
{
    // TODO: No need to create a whole system, just need actInfo and sInfo
    ObjectFactory factory(settings.configFile);
    Environment* env = factory.createEnvironment(0,0);
    QApproximator* Qvals;
    QApproximator* Qexpl;
    Learner* learner;
    int nAgents = (nranks-1)*env->agents.size();
    
    if (settings.learner == "Q")
    {
        debug("Q\n");
        Qvals = new MultiTable(env->sI, env->aI, settings, nAgents);
        learner = new QLearning(Qvals, env->aI, settings.gamma, settings.greedyEps, settings.lRate);
    }
    else if (settings.learner == "QNN")
    {
        printf("Q learning with Network approximator.\n");
        Qvals = new NFQApproximator(env->sI, env->aI, settings, nAgents);
        learner = new QLearning(Qvals, env->aI, settings.gamma, settings.greedyEps, settings.lRate);
    }
    else if (settings.learner == "NFQ")
    {
        printf("DQN learning.\n");
        Qvals = new NFQApproximator(env->sI, env->aI, settings, nranks*env->agents.size()); //TODO fix last argument: size of agent memories (to be read from history file)
        learner = new NFQ(Qvals, env->aI, settings.gamma, settings.greedyEps, settings.lRate);
    }
    
    Master* master = new Master(learner, Qvals, env, nranks, settings.gamma*settings.lambda);
    
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
    debugLvl=0;
    vector<OptionStruct> opts ({
    {'c', "config",   STRING,"config file",    &settings.configFile,(string)"factory"},
    {'t', "dt",       REAL,  "Sim timestep",   &settings.dt,        (Real)0.01},
    {'f', "end_time", REAL,  "End time of sim",&settings.endTime,   (Real)1e9},
    {'g', "gamma",    REAL,  "Gamma parameter",&settings.gamma,     (Real)0.95},
    {'e', "greedyeps",REAL,  "Greedy epsilon", &settings.greedyEps, (Real)0.05},
    {'l', "learnrate",REAL,  "Learning rate",  &settings.lRate,     (Real)0.1},
    {'d', "lambda",   REAL,  "Lambda",         &settings.lambda,    (Real)0.0},
    {'s', "rand_seed",INT,   "Random seed",    &settings.randSeed,  (int)seed},
    {'r', "restart",  STRING,"Restart",        &settings.restart,   (string)"res/policy"},
    {'q', "save_freq",INT,   "Save frequency", &settings.saveFreq,  (int)1000},
    {'v', "debug_lvl",INT,   "Debug level",    &debugLvl,           (int)4},
    {'p', "prefix",   STRING,"Save folder",    &settings.prefix,    (string)"res/"},
    {'H', "nne",      REAL,  "NN's eta",       &settings.nnEta,     (Real)0.001},
    {'A', "nna",      REAL,  "NN's alpha",     &settings.nnAlpha,   (Real)0.5},
    {'D', "nnl",      REAL,  "NN's lambda",    &settings.nnLambda,  (Real)0.0},
    {'K', "nnk",      REAL,  "NN's kappa",     &settings.nnKappa,   (Real)0.0},
    {'S', "nnS",      REAL,  "NN's adapt rate",&settings.nnAdFac,   (Real)1e-6},
    {'F', "AL_fac",   REAL,  "Adv Learn fac",  &settings.AL_fac,    (Real)0.0},
    {'N', "nnl1",     INT,   "NN norm layer 1",&settings.nnLayer1,  (int)0},
    {'L', "nnl2",     INT,   "NN norm layer 2",&settings.nnLayer2,  (int)0},
    {'W', "nnl3",     INT,   "NN norm layer 3",&settings.nnLayer3,  (int)0},
    {'Y', "nnm1",     INT,   "NN LSTM layer 1",&settings.nnMemory1, (int)32},
    {'Z', "nnm2",     INT,   "NN LSTM layer 2",&settings.nnMemory2, (int)16},
    {'X', "nnm3",     INT,   "NN LSTM layer 3",&settings.nnMemory3, (int)8},
    {'Q', "learn",    STRING,"Learner Type",   &settings.learner,   (string)"NFQ"}
    });
    
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
