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
    
    double time = 0;
    
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
        Qvals = new ANNApproximator(env->sI, env->aI, settings.network, nranks*env->agents.size());
        learner = new QLearning(Qvals, env->aI, settings.gamma, settings.greedyEps, settings.lRate);
    }
    else if (settings.learner == "A")
    {
        debug("A learning with Network approximator.\n");
        Qvals = new ANNApproximator(env->sI, env->aI, "LSTM", nranks*env->agents.size());
        learner = new ALearning(Qvals, env->aI, settings.gamma, settings.greedyEps, settings.lRate);
    }
    else if (settings.learner == "NFQNN")
    {
        Qvals = new NFQApproximator(env->sI, env->aI, settings.gamma, settings.network, nranks*env->agents.size()); //TODO fix last argument: size of agent memories (to be read from history file)
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
    /*else if (settings.learner == "Speedy")
    {
        Qvals = new MultiTable(env->sI, env->aI);
        Qold = new MultiTable(env->sI, env->aI);
        learner = new SpeedyQLearning(Qvals, Qold, env->aI, settings.gamma, settings.greedyEps, settings.lRate, settings.lambda);
    }*/


    Master* master = new Master(learner, Qvals, env->aI, env->sI, env->agents.size(), nranks, settings.gamma*settings.lambda);
    
    if (settings.restart != "none")
        master->restart(settings.restart);
    
    // Save results to dir named  settings.prefix
    if (!Saver::makedir((settings.prefix+"/").c_str())) die("Unable to make a working directory!");

//     Various savers
//     TODO: Savers should be specified in factory file

    RewardSaver* rsaver = new RewardSaver((ofstream*)&cout, 1000);//"reward.txt");
    master->registerSaver(rsaver);

    StateSaver* ssaver = new StateSaver("state.txt", 1000);
    //master->registerSaver(ssaver);
    master->run();
}

int main (int argc, char** argv)
{
    int rank, nranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    
    vector<OptionStruct> opts
	({
		{'c', "config",     STRING, "Name of config file",    &settings.configFile, (string)"/home/novatig/smarties/factoryCart"},
		{'t', "dt",         DOUBLE, "Simulation timestep",    &settings.dt,         0.01},
		{'f', "end_time",   DOUBLE, "End time of simulaiton", &settings.endTime,    1e9},
		{'g', "gamma",      DOUBLE, "Gamma parameter",        &settings.gamma,      0.9},
		{'e', "greedy_eps", DOUBLE, "Greedy epsilon",         &settings.greedyEps,  0.05},
        {'l', "learn_rate", DOUBLE, "Learning rate",          &settings.lRate,      0.1},
        {'d', "lambda",     DOUBLE, "Lambda",                 &settings.lambda,     0.0},
		{'s', "rand_seed",  INT,    "Random seed",            &settings.randSeed,   84967194},
        {'r', "restart",    STRING, "Restart",                &settings.restart,    (string)"none"},
		{'q', "save_freq",  INT,    "Save frequency",         &settings.saveFreq,   10000},
        {'v', "debug_lvl",  INT,    "Debug level",            &debugLvl,            2},
        {'p', "prefix",     STRING, "Save folder",            &settings.prefix,     (string)"res/"},
        {'H', "nne",        DOUBLE, "NN's eta",               &settings.nnEta,      0.3},
		{'A', "nna",        DOUBLE, "NN's alpha",             &settings.nnAlpha,    0.1},
        {'D', "nnl",        DOUBLE, "NN's lambda",            &settings.nnLambda,    0.0},
        {'L', "nnl1",       INT,    "NN hidden layer 1",      &settings.nnLayer1,   32},
        {'M', "nnl2",       INT,    "NN hidden layer 2",      &settings.nnLayer2,   32},
        {'N', "net",        STRING, "Network Type",           &settings.network,    (string)"ANN"},
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
