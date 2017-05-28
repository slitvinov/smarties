/*
 *  main.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "allLearners.h"
#include "Scheduler.h"
#include "ObjectFactory.h"

using namespace ErrorHandling;
using namespace std;

void runClient();
void runSlave(MPI_Comm slavesComm);
void runMaster(MPI_Comm slavesComm, MPI_Comm mastersComm);

Settings settings;
Uint ErrorHandling::debugLvl;

void runSlave(MPI_Comm slavesComm)
{
	int rank, nranks;
	MPI_Comm_rank(slavesComm, &rank);
	MPI_Comm_size(slavesComm, &nranks);
	if(rank==0) die("Slave is master?\n")
	if(nranks<=1) die("Slave has no master?\n");

	int wRank, wSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &wRank);
	MPI_Comm_size(MPI_COMM_WORLD, &wSize);

	settings.nSlaves = 1;
	ObjectFactory factory(settings);
	Environment* env = factory.createEnvironment(rank, 0);
	settings.nAgents = env->agents.size();
	Communicator comm = env->create_communicator(slavesComm, settings.sockPrefix, true);

	Slave simulation(&comm, env, settings);
	simulation.run();
}

void runClient()
{
	settings.nSlaves = 1;
	ObjectFactory factory(settings);
	Environment* env = factory.createEnvironment(1, 0);
	settings.nAgents = env->agents.size();

	Learner* learner = createLearner(MPI_COMM_WORLD, env, settings);
	Communicator comm = env->create_communicator(MPI_COMM_NULL, settings.sockPrefix, false);
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
	assert(nSlaves>=0);
	if(isSlave) die("Master is slave?\n")
    		//if(nSlaves==0) die("Master has no slaves?\n");

    		int wRank, wSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &wRank);
	MPI_Comm_size(MPI_COMM_WORLD, &wSize);

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

	Learner* learner = createLearner(mastersComm, env, settings);

	Master master(slavesComm, learner, env, settings);
	master.restart(settings.restart);
	printf("nthreads %d\n",settings.nThreads); fflush(0);

	#if 1
		if (settings.restart != "none" && !nSlaves && !learner->nData()) {
			printf("No slaves, just dumping the policy\n");
			vector<Uint> nbins(env->stateDumpNBins());
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

	vector<ArgumentParser::OptionStruct> opts = settings.initializeOpts();

	int provided, rank, nranks;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
	if (provided < MPI_THREAD_FUNNELED)
		die("The MPI implementation does not have required thread support\n");

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);

	ArgumentParser::Parser parser(opts);
	parser.parse(argc, argv, rank == 0);

	settings.bRecurrent = settings.nnType=="LSTM" || settings.nnType=="RNN";

	if (not settings.isServer) {
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
	if(settings.nThreads<1) die("Error: nThreads\n");
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
