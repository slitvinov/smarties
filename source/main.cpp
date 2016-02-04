/*
 *  main.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <vector>
#include <mpi.h>
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
#include "Learners/ALearning.h"
#include "Learners/Sarsa.h"
#include "Learners/Learner.h"
#include "Learners/NFQ.h"
#include "Learners/Explorer.h"
#include "ObjectFactory.h"
#include "Settings.h"
#include "Learners/SpeedyQLearning.h"
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
        int test = simulation->evolve(time);
	
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
    
    if (settings.learner == "Q")
    {
        debug("Q\n");
        Qvals = new MultiTable(env->sI, env->aI, settings.gamma);
        learner = new QLearning(Qvals, env->aI, settings.gamma, settings.greedyEps, settings.lRate);
    }
    else if (settings.learner == "QNN")
    {
        debug("Q learning with Network approximator.\n");
        Qvals = new ANNApproximator(env->sI, env->aI, settings, nranks*env->agents.size());
        learner = new QLearning(Qvals, env->aI, settings.gamma, settings.greedyEps, settings.lRate);
    }
    else if (settings.learner == "A")
    {
        debug("A learning with Network approximator.\n");
        Qvals = new ANNApproximator(env->sI, env->aI, settings, nranks*env->agents.size());
        learner = new ALearning(Qvals, env->aI, settings.gamma, settings.greedyEps, settings.lRate);
    }
    else if (settings.learner == "NFQNN")
    {
        Qvals = new NFQApproximator(env->sI, env->aI, settings, nranks*env->agents.size()); //TODO fix last argument: size of agent memories (to be read from history file)
        learner = new NFQ(Qvals, env->aI, settings.gamma, settings.greedyEps, settings.lRate);
    }
    else if (settings.learner == "NFQtable")
    {
        Qvals = new MultiTable(env->sI, env->aI, settings.gamma);
        learner = new NFQ(Qvals, env->aI, settings.gamma, settings.greedyEps, settings.lRate);
    }
    else if (settings.learner == "SARSA")
    {
        debug("Sarsa\n");
        Qvals = new MultiTable(env->sI, env->aI, settings.gamma);
        learner = new Sarsa(Qvals, env->aI, settings.gamma, settings.greedyEps, settings.lRate, settings.lambda);
    }
    else if (settings.learner == "XP")
    {
        Qvals = new ANNApproximator(env->sI, env->aI, settings,  nranks*env->agents.size());
        Qexpl = new ANNApproximator(env->sI, env->aI, settings,  nranks*env->agents.size());
        learner = new Explorer(Qvals, Qexpl, env->aI, settings.gamma, settings.greedyEps, settings.lRate, settings.lambda);
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
    int rank, nranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    
    struct timeval clock;
    gettimeofday(&clock, NULL);
    cout << clock.tv_usec << endl;
    int seed = 84967194 + floor(clock.tv_usec);
    
    vector<OptionStruct> opts ({
		{'c', "config",     STRING, "Name of config file",    &settings.configFile, (string)"/home/novatig/smarties/factoryCart"},
		{'t', "dt",         REAL,   "Simulation timestep",    &settings.dt,         0.01},
		{'f', "end_time",   REAL,   "End time of simulaiton", &settings.endTime,    1e9},
		{'g', "gamma",      REAL,   "Gamma parameter",        &settings.gamma,      0.9},
		{'e', "greedy_eps", REAL,   "Greedy epsilon",         &settings.greedyEps,  0.05},
        {'l', "learn_rate", REAL,   "Learning rate",          &settings.lRate,      0.1},
        {'d', "lambda",     REAL,   "Lambda",                 &settings.lambda,     0.0},
		{'s', "rand_seed",  INT,    "Random seed",            &settings.randSeed,   seed},
        {'r', "restart",    STRING, "Restart",                &settings.restart,    (string)"none"},
		{'q', "save_freq",  INT,    "Save frequency",         &settings.saveFreq,   10000},
        {'v', "debug_lvl",  INT,    "Debug level",            &debugLvl,            2},
        {'p', "prefix",     STRING, "Save folder",            &settings.prefix,     (string)"res/"},
        {'H', "nne",        REAL,   "NN's eta",               &settings.nnEta,      0.05},
		{'A', "nna",        REAL,   "NN's alpha",             &settings.nnAlpha,    0.5},
        {'D', "nnl",        REAL,   "NN's lambda",            &settings.nnLambda,   0.0},
        {'K', "nnk",        REAL,   "NN's kappa",             &settings.nnKappa,    0.0},
        {'S', "nnS",        REAL,   "NN's adapt lrate fac",   &settings.nnAdFac,    1e-6},
        {'F', "AL_fac",     REAL,   "Adv Learning factor",    &settings.AL_fac,     2.0},
        {'N', "nnl1",       INT,    "NN hidden layer 1",      &settings.nnLayer1,   100},
        {'L', "nnl2",       INT,    "NN hidden layer 2",      &settings.nnLayer2,   15},
        {'W', "nnl3",       INT,    "NN hidden layer 3",      &settings.nnLayer3,   0},
        {'X', "nnl4",       INT,    "NN hidden layer 4",      &settings.nnLayer4,   0},
        {'Y', "nnm1",       INT,    "NN memory layer 1",      &settings.nnMemory1,  10},
        {'Z', "nnm2",       INT,    "NN memory layer 2",      &settings.nnMemory2,  0},
        {'O', "nnout",      INT,    "NN's outputs",           &settings.nnOuts,     1},
        {'M', "nnm1",       INT,    "NN memory layer 1",      &settings.nnMemory1,  10},
        {'T', "net",        STRING, "Network Type",           &settings.network,    (string)"ANN"},
        {'Q', "learn",      STRING, "Learner Type",           &settings.learner,    (string)"Q"}
    });
	
	Parser parser(opts);
	parser.parse(argc, argv, rank == 0);
	if (settings.lRate < 1e-9) settings.immortal = false;
	else settings.immortal = true;
    
    if  (settings.randSeed == -1 )  srand(time(NULL));
	else							srand(settings.randSeed + rank);
	
    if (rank == 0) runMaster(nranks);
    else           runSlave(rank);
    
	return 0;
}
