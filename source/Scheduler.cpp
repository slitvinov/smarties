//
//  Scheduler.cpp
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#include "Misc.h"
#include "Scheduler.h"

#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cassert>

static int doublePtrToInt(const double* const ptr);

Master::Master(MPI_Comm _c, Learner*const _l, Environment*const _e, Settings&_s):
		  slavesComm(_c), learner(_l), env(_e), aI(_e->aI), sI(_e->sI),
		  agents(_e->agents), bTrain(_s.bTrain), nPerRank(_e->nAgentsPerRank),
		  nSlaves(_s.nSlaves), saveFreq(_s.saveFreq), nThreads(_s.nThreads),
		  inSize((3+_e->sI.dim)*sizeof(double)), outSize(_e->aI.dim*sizeof(double)),
		  inbuf(_alloc(inSize)), outbuf(_alloc(outSize)),
		  sOld(_e->sI), sNew(_e->sI), aOld(_e->aI,&_s.generators[0]), aNew(_e->aI,&_s.generators[0]),
		  meanR(0), varR(0),  iter(0), status(_e->agents.size(),1)
{
	//the following Irecv will be sent after sending the action
	MPI_Irecv(inbuf, inSize, MPI_BYTE, MPI_ANY_SOURCE, 1, slavesComm, &request);
}

void Master::restart(string fname)
{
	learner->restart(fname);
	if (fname == "none") return;
	FILE * f = fopen("master.status", "r");
	unsigned long iter_fake = 0;
	if (f == NULL) return;
	fscanf(f, "master iter: %lu\n", &iter_fake);
	if(iter_fake) iter = iter_fake;
	printf("master iter: %lu\n", iter);
	fclose(f);
}

void Master::save()
{
	ofstream filestats;
	filestats.open("master_rewards.dat", ios::app);
	filestats<<iter<<" "<<meanR<<" "<<varR<<endl;
	filestats.close();
	printf("Iter %lu, Mean reward: %f variance:%f \n", iter, meanR, varR);

	FILE * f = fopen("master.status", "w");
	if (f != NULL)
		fprintf(f, "master iter: %lu\n", iter);
	fclose(f);

	learner->save("policy");
}

void Master::run()
{
	MPI_Status mpistatus;
	int completed=0, agentStatus=0, agent;
	double reward;
	while (true) {
		while (true) {
			//if threaded: check on the guys, synchronize, apply gradient
			if (nThreads > 1 && learner->checkBatch(iter)) return;

			MPI_Test(&request, &completed, &mpistatus);
			if (completed) break;

			//if single thread master: process a batch
			if (nThreads == 1) learner->TrainBatch();
		}
		//printf("Master receives from %d\n", mpistatus.MPI_SOURCE);
		const int slave = mpistatus.MPI_SOURCE;
		recvState(slave, agent, agentStatus, reward);

		if (agentStatus == _AGENT_FAILCOMM) {
			learner->clearFailedSim((slave-1)*nPerRank, slave*nPerRank);
			continue;
		}

		learner->select(agent, sNew, aNew, sOld, aOld, agentStatus, reward);
		#if 0
		printf("To learner %d: %s --> %s with %s rewarded with %f going to %s\n",
				agent, sOld.print().c_str(), sNew.print().c_str(),
				aOld.print().c_str(), reward, aNew.print().c_str());
		fflush(0);
		#endif
		if (agentStatus != _AGENT_FIRSTCOMM) {
			const Real alpha = 1./saveFreq;// + std::min(0.,1-iter/(Real)saveFreq);
			const Real oldMean = meanR;
			meanR = (1.-alpha)*meanR + alpha*reward;
			varR = (1.-alpha)*varR + alpha*(reward-meanR)*(reward-oldMean);
		}
		if (agentStatus != _AGENT_LASTCOMM)  {
			sendAction(slave, agent);
		} else { //if terminal, no action required
			if(env->resetAll)
				learner->pushBackEndedSim((slave-1)*nPerRank, slave*nPerRank);
		}
		if (++iter % saveFreq == 0) save();
	}
	die("How on earth could you possibly get here? \n");
}

Slave::Slave(Communicator*const _c, Environment*const _e, Settings& _s):
		comm(_c), env(_e), bTrain(_s.bTrain), status(_e->agents.size(),1) {}

void Slave::run()
{
	vector<double> state(env->sI.dim);
	int iAgent, agentStatus;
	double reward;

	while(true)
	{
		comm->launch();
		while(true)
		{
			if (comm->recvStateFromApp()) break; //sim crashed
			comm->unpackState(iAgent, agentStatus, state, reward);
			comm->sendStateMPI();

			status[iAgent] = agentStatus;
			if(agentStatus != _AGENT_LASTCOMM)
			{
				comm->recvActionMPI();
				comm->sendActionToApp();
			} else {
				bool bDone = true; //did all agents reach terminal state?
				for (int i=0; i<status.size(); i++)
					bDone = bDone && status[i] == _AGENT_LASTCOMM;
				bDone = bDone || env->resetAll; //does env end is any terminates?
				/*
          if(bDone && !bTrain) {
            comm->answerTerminateReq(-1);
            return;
          }
          else
				 */
				comm->answerTerminateReq(1);
			}
		}
		//if here, a crash happened:
		//if we are training, then launch again, otherwise exit
		//if (!bTrain) return;
	}
}

Client::Client(Learner*const _l, Communicator*const _c, Environment*const _e,
		Settings& _s):
		  learner(_l), comm(_c), env(_e), aI(_e->aI), sI(_e->sI), agents(_e->agents),
		  sOld(_e->sI), sNew(_e->sI), aOld(_e->aI, &_s.generators[0]), aNew(_e->aI, &_s.generators[0]),
		  status(_e->agents.size(),1)
{}

void Client::run()
{
	vector<double> state(env->sI.dim);
	int iAgent, agentStatus;
	double reward;

	comm->launch();

	while(true)
	{
		if (comm->recvStateFromApp()) break; //sim crashed

		prepareState(iAgent, agentStatus, reward);
		learner->select(iAgent, sNew, aNew, sOld, aOld, agentStatus, reward);

		printf("To learner %d: %s --> %s with %s rewarded with %f going to %s\n",
				iAgent, sOld.print().c_str(), sNew.print().c_str(),
				aOld.print().c_str(), reward, aNew.print().c_str());
		status[iAgent] = agentStatus;

		if(agentStatus != _AGENT_LASTCOMM) {
			prepareAction(iAgent);
			comm->sendActionToApp();
		} else {
			bool bDone = true; //did all agents reach terminal state?
			for (int i=0; i<status.size(); i++)
				bDone = bDone && status[i] == _AGENT_LASTCOMM;
			bDone = bDone || env->resetAll; //or does env end is any terminates?

			if(bDone) {
				comm->answerTerminateReq(-1);
				return;
			}
			else comm->answerTerminateReq(1);
		}
	}
}

void Client::prepareState(int& iAgent, int& istatus, Real& reward)
{
	const double*const buf = comm->getDataout();
	iAgent = doublePtrToInt(buf+0);
	assert(iAgent >= 0 && iAgent < agents.size());

	istatus = doublePtrToInt(buf+1);
	agents[iAgent]->Status = istatus;

	sNew.unpack(buf+2);

	//agent's s is stored in sOld
	agents[iAgent]->swapStates();
	agents[iAgent]->setState(sNew);
	agents[iAgent]->getOldState(sOld);
	agents[iAgent]->getAction(aOld);

	reward = buf[env->sI.dim+2];
	agents[iAgent]->r = reward;
}

void Master::recvState(const int slave, int& iAgent, int& istatus, Real& reward)
{
	const double*const buf = inbuf;

	const int recv_iAgent = doublePtrToInt(buf+0);
	iAgent = (slave-1) * nPerRank + recv_iAgent;
	assert(iAgent >= 0 && iAgent < agents.size());

	istatus = doublePtrToInt(buf+1);
	agents[iAgent]->Status = istatus;

	sNew.unpack(buf+2);

	//agent's s is stored in sOld
	agents[iAgent]->swapStates();
	agents[iAgent]->setState(sNew);
	agents[iAgent]->getOldState(sOld);
	agents[iAgent]->getAction(aOld);

	reward = buf[env->sI.dim+2];
	agents[iAgent]->r = reward;

	MPI_Irecv(inbuf, inSize, MPI_BYTE, MPI_ANY_SOURCE, 1, slavesComm, &request);
}

void Master::sendAction(const int slave, const int iAgent)
{
	assert(iAgent >= 0 && iAgent < agents.size());
	agents[iAgent]->act(aNew);
	aNew.pack(outbuf);
	MPI_Send(outbuf, outSize, MPI_BYTE, slave, 0, slavesComm);
}

void Client::prepareAction(const int iAgent)
{
	assert(iAgent >= 0 && iAgent < agents.size());
	agents[iAgent]->act(aNew);
	aNew.pack(comm->getDatain());
}

static int doublePtrToInt(const double*const ptr)
{
	return (int)*ptr;//*((int*)ptr);
}
