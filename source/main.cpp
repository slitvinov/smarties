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

#include "ArgumentParser.h"
#include "ErrorHandling.h"
#include "Learners/QLearning.h"
#include "Learners/Sarsa.h"
#include "Learners/Learner.h"
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
    int index = 0;
    while (true) {
        // Class creating agents and environment from info in the file
        ObjectFactory factory(settings.configFile);
        Environment* env = factory.createEnvironment(rank, index);

        Slave* simulation = new Slave(env, settings.dt, rank);

        double time = 0;

        while (time < settings.endTime + settings.dt/2.0)
        {
            if(simulation->evolve(time))
            {
                delete simulation;
                index++;
                break;
            }
            time += settings.dt;
        }
    }
	
	exit(0);
}

void runMaster(int nranks)
{
    // TODO: No need to create a whole system, just need actInfo and sInfo
    ObjectFactory factory(settings.configFile);
    Environment* env = factory.createEnvironment(0,0);
    
    // Define learning algorithm
	// TODO: Make this through object factory
    QApproximator* Qvals = new MultiTable(env->sI, env->aI);
	Learner* learner = new Sarsa(Qvals, env->aI, settings.gamma, settings.greedyEps, settings.lRate, settings.lambda);
    
    Master* master = new Master(learner, env->aI, env->sI, env->agents.size(), nranks, settings.gamma*settings.lambda);
    
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
		{'c', "config",     STRING, "Name of config file",    &settings.configFile, (string)"/Users/alexeedm/Documents/Fish/smarties/factory/factoryRL_test1"},
		{'t', "dt",         DOUBLE, "Simulation timestep",    &settings.dt,         0.01},
		{'f', "end_time",   DOUBLE, "End time of simulaiton", &settings.endTime,    1e9},
		{'g', "gamma",      DOUBLE, "Gamma parameter",        &settings.gamma,      0.85},
		{'e', "greedy_eps", DOUBLE, "Greedy epsilon",         &settings.greedyEps,  0.01},
        {'l', "learn_rate", DOUBLE, "Learning rate",          &settings.lRate,      0.01},
        {'d', "lambda",     DOUBLE, "Lambda",                 &settings.lambda,     0.0},
		{'s', "rand_seed",  INT,    "Random seed",            &settings.randSeed,   11111},
        {'r', "restart",    STRING, "Restart",                &settings.restart,    (string)"none"},
		{'q', "save_freq",  INT,    "Save frequency",         &settings.saveFreq,   10000},
        {'v', "debug_lvl",  INT,    "Debug level",            &debugLvl,            2},
        {'p', "prefix",     STRING, "Save folder",            &settings.prefix,     (string)"res/"}
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
