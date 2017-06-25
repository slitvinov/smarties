/*
 *  main.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "AllLearners.h"
#include "Scheduler.h"
#include "ObjectFactory.h"
using namespace std;

void runClient();
void runSlave(MPI_Comm slavesComm);
void runMaster(MPI_Comm slavesComm, MPI_Comm mastersComm);
Settings settings;

void runSlave(MPI_Comm slavesComm)
{
	MPI_Comm_rank(slavesComm, &settings.slaves_rank);
	MPI_Comm_size(slavesComm, &settings.slaves_size);
	if(settings.slaves_rank==0) die("Slave is master?\n")
	if(settings.slaves_size<=1) die("Slave has no master?\n");
	settings.nSlaves = 1;
	ObjectFactory factory(settings);
	Environment* env = factory.createEnvironment();
	Communicator comm = env->create_communicator(slavesComm, settings.sockPrefix, true);

	Slave simulation(&comm, env, settings);
	simulation.run();
}

void runClient()
{
	settings.nSlaves = 1;
	ObjectFactory factory(settings);
	Environment* env = factory.createEnvironment();
	Communicator comm = env->create_communicator(MPI_COMM_NULL, settings.sockPrefix, false);

	Learner* learner = createLearner(MPI_COMM_WORLD, env, settings);
	if (settings.restart != "none") {
		learner->restart(settings.restart);
		//comm.restart(settings.restart);
	}
	Client simulation(learner, &comm, env, settings);
	simulation.run();
}

void runMaster(MPI_Comm slavesComm, MPI_Comm mastersComm)
{
	MPI_Comm_rank(slavesComm, &settings.slaves_rank);
	MPI_Comm_size(slavesComm, &settings.slaves_size);
	MPI_Comm_rank(mastersComm, &settings.learner_rank);
	MPI_Comm_size(mastersComm, &settings.learner_size);
	settings.nSlaves = settings.slaves_size-1; //minus master
	assert(settings.nSlaves>=0 && settings.slaves_rank == 0);

	ObjectFactory factory(settings);
	Environment* env = factory.createEnvironment();
	Communicator comm = env->create_communicator(slavesComm, settings.sockPrefix, true);

	if(env->mpi_ranks_per_env) { //unblock creation of app comm if needed
		MPI_Comm tmp_com;
		MPI_Comm_split(slavesComm, MPI_UNDEFINED, 0, &tmp_com);
		//no need to free this
	}

	Learner* learner = createLearner(mastersComm, env, settings);

	Master master(slavesComm, learner, env, settings);
	master.restart(settings.restart);
	printf("nthreads %d\n",settings.nThreads); fflush(0);

#if 1
	if (settings.restart != "none" && !settings.nSlaves && !learner->nData())
	{
		printf("No slaves, just dumping the policy\n");
		learner->dumpPolicy();
		abort();
	}
#endif
	assert(settings.nThreads > 1);
	learner->run(&master);
	die("Master returning?\n");
}

int main (int argc, char** argv)
{
	struct timeval clock;
	gettimeofday(&clock, NULL);

	vector<ArgumentParser::OptionStruct> opts = settings.initializeOpts();

	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
	if (provided < MPI_THREAD_FUNNELED)
		die("The MPI implementation does not have required thread support\n");

	MPI_Comm_rank(MPI_COMM_WORLD, &settings.world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &settings.world_size);

	ArgumentParser::Parser parser(opts);
	parser.parse(argc, argv, settings.world_rank == 0);
	settings.bRecurrent = settings.nnType=="LSTM" || settings.nnType=="RNN";
	MPI_Barrier(MPI_COMM_WORLD);

	if (not settings.isServer)
	{
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
	if (!settings.world_rank) {
		runSeed = abs(clock.tv_usec % std::numeric_limits<int>::max());
		for (int i = 1; i < settings.world_size; i++)
			MPI_Send(&runSeed, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
	} else
		MPI_Recv(&runSeed, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	settings.sockPrefix = runSeed+settings.world_rank;
	settings.generators.reserve(settings.nThreads);
	settings.generators.push_back(mt19937(settings.sockPrefix));
	if(settings.nThreads<1) die("Error: nThreads\n");
	for(int i=1; i<settings.nThreads; i++)
		settings.generators.push_back(mt19937(settings.generators[0]));

	const int slavesPerMaster = ceil(settings.world_size/(double)settings.nMasters)-1;
	const int isMaster = settings.world_rank % (slavesPerMaster+1) == 0;
	const int whichMaster = settings.world_rank / (slavesPerMaster+1);
	printf("Job size=%d, with %d masters, %d slaves per master. I'm %d: %s part of comm %d.\n",
			settings.world_size,settings.nMasters,slavesPerMaster,settings.world_rank,
			isMaster?"master":"slave",whichMaster);

	MPI_Comm slavesComm; //this communicator allows slaves to talk to their master
	MPI_Comm mastersComm; //this communicator allows masters to talk among themselves
	MPI_Comm_split(MPI_COMM_WORLD, isMaster, settings.world_rank, &mastersComm);
	MPI_Comm_split(MPI_COMM_WORLD, whichMaster, settings.world_rank, &slavesComm);
	if (!isMaster) MPI_Comm_free(&mastersComm);

	MPI_Barrier(MPI_COMM_WORLD);
	if (isMaster) runMaster(slavesComm, mastersComm);
	else          runSlave(slavesComm);

	if (isMaster) MPI_Comm_free(&mastersComm);
	MPI_Comm_free(&slavesComm);
	MPI_Finalize();
	return 0;
}
