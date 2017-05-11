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
#include "Learners/RACER.h"
#include "Learners/DACER.h"
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
Learner* createLearner(MPI_Comm mastersComm, Environment*const env);
void runMaster(MPI_Comm slavesComm, MPI_Comm mastersComm);
Settings settings;
int ErrorHandling::debugLvl;

Learner* createLearner(MPI_Comm mastersComm, Environment*const env)
{
	if(settings.learner=="DQ" || settings.learner=="DQN" || settings.learner=="NFQ") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.dqnAppendS);
		settings.nnOutputs = env->aI.maxLabel;
		return new NFQ(mastersComm, env, settings);
	}
	else if (settings.learner == "RACER") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.dqnAppendS);
		const int nA = env->aI.dim;
		const int nL = (nA*nA+nA)/2;
		settings.nnOutputs = RACER::getnOutputs(nL, nA);
		settings.bSeparateOutputs = true; //else it does not really work
		return new RACER(mastersComm, env, settings);
	}
	else if (settings.learner == "DACER") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.dqnAppendS);
		const int nA = env->aI.maxLabel;
		printf("Read %d outputs\n",nA);
		settings.nnOutputs = DACER::getnOutputs(nA);
		settings.bSeparateOutputs = true; //else it does not really work
		return new DACER(mastersComm, env, settings);
	}
	else if (settings.learner == "NA" || settings.learner == "NAF") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.dqnAppendS);
		const int nA = env->aI.dim;
		const int nL = (nA*nA+nA)/2;
		settings.nnOutputs = 1+nL+nA;
		settings.bSeparateOutputs = true; //else it does not really work
		return new NAF(mastersComm, env, settings);
	}
	else if (settings.learner == "DP" || settings.learner == "DPG") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.dqnAppendS) + env->aI.dim;
		settings.nnOutputs = 1;
		return new DPG(mastersComm, env, settings);
	} else die("Learning algorithm not recognized\n");
	assert(false);
	return new NFQ(mastersComm, env, settings); //fake, to silence warnings
}

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
	//if(rank!=wRank || wSize!=nranks)
	//die("Not ready for multiple masters!\n");

	settings.nSlaves = 1;
	ObjectFactory factory(settings);
	Environment* env = factory.createEnvironment(rank, 0);
	settings.nAgents = env->agents.size();

	const bool isSpawner = true;
	const bool verbose = 0;
	const int sdim = env->sI.dim;
	const int adim = env->aI.dim;
	const int socket = settings.sockPrefix;
	const std::string exec = env->execpath;
	const std::string flog = "log_"+std::to_string(wRank)+"_";
	int available_ranks = nranks-1; //one is the master
	const int ranks_per = env->mpi_ranks_per_env;

	if(ranks_per) // if 0 then it's supposed to be run as a forked process
	{
		if(available_ranks%ranks_per)
			die("Number of ranks does not match app\n");
		int slaveGroup = (rank-1) / ranks_per;
		MPI_Comm app_com;
		MPI_Comm_split(slavesComm, slaveGroup, rank, &app_com);

		Communicator comm(sdim,adim,slavesComm,app_com,slaveGroup,exec,env->paramsfile,flog,verbose);
		do
			comm.ext_app_run();
		while (settings.bTrain);
	}
	else
	{
		Communicator comm(socket,sdim,adim,isSpawner,exec,slavesComm,flog,verbose);
		//if (settings.restart != "none") comm.restart(settings.restart);

		Slave simulation(&comm, env, settings);
		simulation.run();
	}
	die("Slave returning?\n");
}

void runClient()
{
	settings.nSlaves = 1;
	ObjectFactory factory(settings);
	Environment* env = factory.createEnvironment(1, 0);
	settings.nAgents = env->agents.size();

	Learner* learner = createLearner(MPI_COMM_WORLD, env);

	const bool isSpawner = false;
	const bool verbose = 0;
	const int sdim = env->sI.dim;
	const int adim = env->aI.dim;
	const int socket = settings.sockPrefix;
	const std::string exec = env->execpath;
	const std::string flog = "log_0_";

	Communicator comm(socket,sdim,adim,isSpawner,exec,MPI_COMM_WORLD,flog,verbose);
	if (settings.restart != "none") {
		learner->restart(settings.restart);
		//comm.restart(settings.restart);
	}

	Client simulation(learner, &comm, env, settings);
	simulation.run();
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
	//if(isSlave!=wRank || wSize!=nSlaves+1 || nMasters!=1 || masterRank)
	//    die("Not ready for multiple masters!\n");

	settings.nSlaves = nSlaves;

	ObjectFactory factory(settings);
	Environment* env = factory.createEnvironment(0,0);

	settings.nAgents = env->agents.size();

	if(env->mpi_ranks_per_env)
	{ //unblock creation of app comm if needed
		MPI_Comm tmp_com;
		MPI_Comm_split(slavesComm, MPI_UNDEFINED, 0, &tmp_com);
		//no need to free this
	}

	Learner* learner = createLearner(mastersComm, env);

	Master master(slavesComm, learner, env, settings);
	master.restart(settings.restart);
	printf("nthreads %d\n",settings.nThreads); fflush(0);

#if 1
	if (settings.restart != "none" && !nSlaves) {
		printf("No slaves, just dumping the policy\n");
		vector<int> nbins(env->stateDumpNBins());
		vector<Real> lower(env->stateDumpLowerBound()), upper(env->stateDumpUpperBound());
		learner->dumpPolicy(lower, upper, nbins);
		abort();
	}
#endif

	if (settings.nThreads > 1) learner->TrainTasking(&master);
	else master.run();
	die("Master returning?\n");
}

int main (int argc, char** argv)
{
	struct timeval clock;
	gettimeofday(&clock, NULL);
	debugLvl=10;

	vector<OptionStruct> opts ({
		{'N',"nMasters", INT,
			"number of masters (policy-updating ranks)",
			&settings.nMasters, (int)1},
		{'g',"gamma",    REAL,
				"Gamma parameter",
				&settings.gamma,     (Real)0.9},
		{'e',"greedyeps",REAL,
				"fraction of actions chosen randomly",
				&settings.greedyEps, (Real)0.1},
		{'E',"epsAnneal",REAL,
				"number of grad steps over which eps is annealed to 0",
				&settings.epsAnneal, (Real)1e4},
		{'l',"learnrate",REAL,
				"Networks learning rate",
				&settings.lRate,     (Real)0.001},
		{'a',"learn",    STRING,
				"RL algorithm",
				&settings.learner,   (string)"DQ"},
		{'r',"rType",    INT,
				"Reward: see env",
				&settings.rewardType,(int)-1},
		{'i',"senses",   INT,
				"State: see env",
				&settings.senses,   (int)0},
		{'y',"goalDY",   REAL,
				"goalDY: see env",
				&settings.goalDY,    (Real)0.},
		{'t',"bTrain",   INT,
				"Whether training (1) or evaluating a policy (0)",
				&settings.bTrain,    (int)1},
		{'K',"nnL",      REAL,
				"Network's weight decay",
				&settings.nnLambda,  (Real)0.0},
		{'Z',"nnl1",     INT,
				"NN layer 1",
				&settings.nnLayer1,  (int)0},
		{'Y',"nnl2",     INT,
				"NN layer 2",
				&settings.nnLayer2,  (int)0},
		{'X',"nnl3",     INT,
				"NN layer 3",
				&settings.nnLayer3,  (int)0},
		{'W',"nnl4",     INT,
				"NN layer 4",
				&settings.nnLayer4,  (int)0},
		{'V',"nnl5",     INT,
				"NN layer 5",
				&settings.nnLayer5,  (int)0},
		{'T',"nnType",   INT,
				"Network Type: LSTM (1) Feed forward (0)",
				&settings.nnType,    (int)1},
		{'C',"dqnT",     REAL,
				"Delay for target network weight update",
				&settings.dqnUpdateC,(Real)1000},
		{'B',"dqnBatch", INT,
				"Network update batch size",
				&settings.dqnBatch,  (int)10},
		{'A',"dqnNs",    INT,
				"Number of previous states chained together to form NN input",
				&settings.dqnAppendS,(int)0},
		{'L',"dqnSeqMax",INT,
				"max seq length. if greater the sequence is cut",
				&settings.maxSeqLen, (int)200},
		{'M',"dqnSeqMin",INT,
				"min seq length. if less the sequence is ignored",
				&settings.minSeqLen, (int)4},
		{'U',"maxTotSeqNum",INT,
				"maximum number of stored sequences: if exceeded the easier ones are removed",
				&settings.maxTotSeqNum, (int)5000},
		{'p',"nThreads", INT,
				"Number of threads on master ranks",
				&settings.nThreads,  (int)-1},
		{'I',"isServer", INT,
				"Whether smarties launches apps or is launched by app (then cannot train)",
				&settings.isLauncher,  (int)1},
		{'P',"sockPrefix",INT,
				"Number prefix for socket: >0 if launched by app",
				&settings.sockPrefix,  (int)-1},
		{'H',"fileSamp", STRING,
				"Location of transitions log for restart",
				&settings.samplesFile,(string)"obs_master.txt"},
		{'R',"restart", STRING,
				"Location of policy file for restart",
				&settings.restart,(string)"policy"}
	});

	int provided, rank, nranks;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
	if (provided < MPI_THREAD_FUNNELED)
		die("The MPI implementation does not have required thread support\n");

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);

	Parser parser(opts);
	parser.parse(argc, argv, rank == 0);


	if (not settings.isLauncher) {
		if (settings.sockPrefix<0)
			die("Not received a prefix for the socket\n");
		settings.generators.push_back(mt19937(settings.sockPrefix));
		printf("Launching smarties as client.\n");
		if (settings.restart == "none")
			die("smarties as client works only for evaluating policies.\n");
		settings.bTrain = 0;
		runClient();
		MPI_Finalize();
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
	settings.generators.reserve(settings.nThreads);
	settings.generators.push_back(mt19937(settings.sockPrefix));
	for(int i=1; i<settings.nThreads; i++)
		settings.generators.push_back(mt19937(settings.generators[0]));

	const int slavesPerMaster = ceil(nranks/(double)settings.nMasters) - 1;
	const int isMaster = rank % (slavesPerMaster+1) == 0;
	settings.bIsMaster = isMaster;
	const int whichMaster = rank / (slavesPerMaster+1);
	printf("Job size=%d, with %d masters, %d slaves per master. I'm %d: %s part of comm %d.\n",
			nranks,settings.nMasters,slavesPerMaster,rank,isMaster?"master":"slave",whichMaster);

	MPI_Comm slavesComm; //this communicator allows slaves to talk to their master
	MPI_Comm mastersComm; //this communicator allows masters to talk among themselves
	MPI_Comm_split(MPI_COMM_WORLD, isMaster, rank, &mastersComm);
	MPI_Comm_split(MPI_COMM_WORLD, whichMaster, rank, &slavesComm);
	if (!isMaster) MPI_Comm_free(&mastersComm);

	if (isMaster) runMaster(slavesComm, mastersComm);
	else          runSlave(slavesComm);

	if (isMaster) MPI_Comm_free(&mastersComm);
	MPI_Comm_free(&slavesComm);
	MPI_Finalize();
	return 0;
}
