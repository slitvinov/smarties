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
#include "Learners/DPG.h"
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

void runClient();
void runSlave(MPI_Comm slavesComm);
void runMaster(MPI_Comm slavesComm, MPI_Comm mastersComm);
Settings settings;
int ErrorHandling::debugLvl;

void runSlave(MPI_Comm slavesComm)
{
    int rank, nranks;
    MPI_Comm_rank(slavesComm, &rank);
    MPI_Comm_size(slavesComm, &nranks);
    if(rank==0) die("Slave is master?\n")
    if(nranks==1) die("Slave has no master?\n");

    //////// TODO
    int wRank, wSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wRank);
    MPI_Comm_size(MPI_COMM_WORLD, &wSize);
    if(rank!=wRank || wSize!=nranks) die("Not ready for multiple masters!\n");

    ObjectFactory factory(settings);
    Environment* env = factory.createEnvironment(rank, 0);
    settings.nAgents = env->agents.size();
    settings.nSlaves = 1;
    Slave simulation(slavesComm, env, rank, settings);
    if (settings.restart != "none")
        simulation.restart(settings.restart);

    while (true) {
        simulation.run(); //if it returns, something is messed up
        env->close_Comm();
        env->setup_Comm();
    }
}

void runClient()
{
    ObjectFactory factory(settings);
    Environment* env = factory.createEnvironment(1, 0);
    settings.nAgents = env->agents.size();
    settings.nSlaves = 1;

    Learner* learner = nullptr;
    if(settings.learner=="DQ" || settings.learner=="DQN" || settings.learner=="NFQ") {
        settings.nnInputs = env->sI.dimUsed;
        settings.nnOutputs = 1;
        for (int i(0); i<env->aI.dim; i++) settings.nnOutputs*=env->aI.bounds[i];
        learner = new NFQ(MPI_COMM_WORLD, env, settings);
    }
    else if (settings.learner == "NA" || settings.learner == "NAF") {
        settings.nnInputs = env->sI.dimUsed;
        const int nA = env->aI.dim;
        const int nL = (nA*nA+nA)/2;
        settings.nnOutputs = 1+nL+nA;
        settings.bSeparateOutputs = true; //else it does not really work
        learner = new NAF(MPI_COMM_WORLD, env, settings);
    }
    else if (settings.learner == "DP" || settings.learner == "DPG") {
        settings.nnInputs = env->sI.dimUsed + env->aI.dim;
        settings.nnOutputs = 1;
        learner = new DPG(MPI_COMM_WORLD, env, settings);
    } else die("Learning algorithm not recognized\n");
    assert(learner not_eq nullptr);

    Client simulation(learner, env, settings);
}

void runMaster(MPI_Comm slavesComm, MPI_Comm mastersComm)
{
    int masterRank, nMasters, nSlaves, isSlave;
    MPI_Comm_rank(mastersComm, &masterRank);
    MPI_Comm_size(mastersComm, &nMasters);
    MPI_Comm_rank(slavesComm, &isSlave);
    MPI_Comm_size(slavesComm, &nSlaves);
    nSlaves--; //minus master
    if(isSlave) die("Master is slave?\n")
    //if(nSlaves==0) die("Master has no slaves?\n");

    //////// TODO
    int wRank, wSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wRank);
    MPI_Comm_size(MPI_COMM_WORLD, &wSize);
    if(isSlave!=wRank || wSize!=nSlaves+1 || nMasters!=1 || masterRank)
        die("Not ready for multiple masters!\n");

    ObjectFactory factory(settings);
    Environment* env = factory.createEnvironment(0,0);

    settings.nAgents = nSlaves*env->agents.size();
    settings.nSlaves = nSlaves;

    Learner* learner = nullptr;
    if(settings.learner=="DQ" || settings.learner=="DQN" || settings.learner=="NFQ") {
        settings.nnInputs = env->sI.dimUsed;
        settings.nnOutputs = 1;
        for (int i(0); i<env->aI.dim; i++) settings.nnOutputs*=env->aI.bounds[i];
        learner = new NFQ(mastersComm, env, settings);
    }
    else if (settings.learner == "NA" || settings.learner == "NAF") {
        settings.nnInputs = env->sI.dimUsed;
        const int nA = env->aI.dim;
        const int nL = (nA*nA+nA)/2;
        settings.nnOutputs = 1+nL+nA;
        settings.bSeparateOutputs = true; //else it does not really work
        learner = new NAF(mastersComm, env, settings);
    }
    else if (settings.learner == "DP" || settings.learner == "DPG") {
        settings.nnInputs = env->sI.dimUsed + env->aI.dim;
        settings.nnOutputs = 1;
        learner = new DPG(mastersComm, env, settings);
    } else die("Learning algorithm not recognized\n");
    assert(learner not_eq nullptr);

    Master master(slavesComm, learner, env, settings);
    if (settings.restart != "none") master.restart(settings.restart);

    if (settings.nThreads > 1) learner->TrainTasking(&master);
    else master.run();
}

int main (int argc, char** argv)
{
    int rank, nranks;

    struct timeval clock;
    gettimeofday(&clock, NULL);
    debugLvl=10;

    vector<OptionStruct> opts ({
      {'N', "nMasters", INT,   "N policy ranks", &settings.nMasters,  (int)1},
      {'g', "gamma",    REAL,  "Gamma parameter",&settings.gamma,     (Real)0.9},
      {'e', "greedyeps",REAL,  "Greedy epsilon", &settings.greedyEps, (Real)0.1},
      {'l', "learnrate",REAL,  "Learning rate",  &settings.lRate,     (Real)0.001},
      {'b', "debug_lvl",INT,   "Debug level",    &debugLvl,           (int)debugLvl},
      {'a', "learn",    STRING,"Learner Type",   &settings.learner,   (string)"DQ"},
      {'r', "rType",    INT,   "Reward: ef,ef,y",&settings.rewardType,(int)-1},
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
      {'L', "dqnSeqMax",INT,   "max seq length", &settings.maxSeqLen, (int)200},
      {'B', "dqnBatch", INT,   "batch update",   &settings.dqnBatch,  (int)10},
      {'p', "nThreads", INT,   "parallel master",&settings.nThreads,  (int)-1},
      {'I', "isServer", INT,   "client=0 server=1",&settings.isLauncher,  (int)1},
      {'P',"sockPrefix",INT,   "socked id prefix",&settings.sockPrefix,  (int)-1},
      //{'H', "fileSamp", STRING,"history file",   &settings.samplesFile,(string)"../history.txt"}
      {'H', "fileSamp", STRING,"history file",   &settings.samplesFile,(string)"obs_master.txt"}
    });

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
        die("The MPI implementation does not have required thread support\n");

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    Parser parser(opts);
    parser.parse(argc, argv, rank == 0);


    if (not settings.isLauncher) {
      if (settings.sockPrefix<0) die("Not received a prefix for the socket\n");
      settings.gen = new mt19937(settings.sockPrefix);
      printf("Launching smarties as client.\n");
      if (settings.restart == "none")
        die("smarties as client works only for evaluating policies.\n");
      settings.bTrain = 0;
      runClient();
      return 0;
    }

    int runSeed;
    if (!rank) {
      runSeed = abs(clock.tv_usec % std::numeric_limits<int>::max());

      for (int i = 1; i < nranks; i++)
          MPI_Send(&runSeed, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

    } else
          MPI_Recv(&runSeed, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    settings.sockPrefix = runSeed+rank;

    settings.gen = new mt19937(settings.sockPrefix);

    const int slavesPerMaster = ceil(nranks/(double)settings.nMasters) - 1;
    const int isMaster = rank % (slavesPerMaster+1) == 0;
    const int whichMaster = rank / (slavesPerMaster+1);
    printf("Job size=%d, with %d masters, %d slaves per master. I'm %d: %s part of comm %d.\n",
    nranks,settings.nMasters,slavesPerMaster,rank,isMaster?"master":"slave",whichMaster);

    MPI_Comm slavesComm; //this communicator allows slaves to talk to their master
    MPI_Comm mastersComm; //this communicator allows masters to talk among themselves
    MPI_Comm_split(MPI_COMM_WORLD, isMaster, rank, &mastersComm);
    MPI_Comm_split(MPI_COMM_WORLD, whichMaster, rank, &slavesComm);
    if (!isMaster) MPI_Comm_free(&mastersComm);

    if (rank == 0) runMaster(slavesComm, mastersComm);
    else           runSlave(slavesComm);

    if (isMaster) MPI_Comm_free(&mastersComm);
    MPI_Comm_free(&slavesComm);
    MPI_Finalize();
    return 0;
}
