//
//  Scheduler.cpp
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#include "Scheduler.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

static void unpackState(double* const data, int& agent, _AGENT_STATUS& info,
		std::vector<double>& state, double& reward);

Master::Master(MPI_Comm _c, Learner*const _l, Environment*const _e, Settings&_s):
				  slavesComm(_c), learner(_l), env(_e), aI(_e->aI), sI(_e->sI),
				  agents(_e->agents), bTrain(_s.bTrain), nPerRank(_e->nAgentsPerRank),
				  nSlaves(_s.nSlaves), saveFreq(_s.saveFreq), nThreads(_s.nThreads),
				  inSize((3+_e->sI.dim)*sizeof(double)), outSize(_e->aI.dim*sizeof(double)),
				  inbuf(_alloc(inSize)), outbuf(_alloc(outSize)), sOld(_e->sI),sNew(_e->sI),
				  aOld(_e->aI,&_s.generators[0]), aNew(_e->aI,&_s.generators[0]), status(_e->agents.size(),1)
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
		debugS("Master receives from %d\n", mpistatus.MPI_SOURCE);
		const int slave = mpistatus.MPI_SOURCE;
		recvState(slave, agent, agentStatus, reward);

		if (agentStatus == _AGENT_FAILCOMM) {
			learner->clearFailedSim((slave-1)*nPerRank, slave*nPerRank);
			continue;
		}

		learner->select(agent, sNew, aNew, sOld, aOld, agentStatus, reward);
		debugS("Agent %d: [%s] -> [%s] with [%s] rewarded with %f going to [%s]\n",
				agent, sOld._print().c_str(), sNew._print().c_str(),
				aOld._print().c_str(), reward, aNew._print().c_str());

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

	while(true) {

		while(true) {
			if (comm->recvStateFromApp()) break; //sim crashed
			unpackState(comm->getDataState(), iAgent, agentStatus, state, reward);

			status[iAgent] = agentStatus;
			if(agentStatus != _AGENT_LASTCOMM)
			{
				comm->sendActionToApp();
			} else {
				bool bDone = true; //did all agents reach terminal state?
				for (Uint i=0; i<status.size(); i++)
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
		comm->launch();
	}
}

Client::Client(Learner*const _l, Communicator*const _c, Environment*const _e,
		Settings& _s):
			  learner(_l), comm(_c), env(_e), aI(_e->aI), sI(_e->sI), agents(_e->agents),
			  sOld(_e->sI), sNew(_e->sI), aOld(_e->aI, &_s.generators[0]),
			  aNew(_e->aI, &_s.generators[0]), status(_e->agents.size(),1)
{}

void Client::run()
{
	vector<double> state(env->sI.dim);
	int iAgent, agentStatus;
	double reward;

	while(true)
	{
		if (comm->recvStateFromApp()) break; //sim crashed

		prepareState(iAgent, agentStatus, reward);
		learner->select(iAgent, sNew, aNew, sOld, aOld, agentStatus, reward);

		debugS("Agent %d: [%s] -> [%s] with [%s] rewarded with %f going to [%s]\n",
				iAgent, sOld._print().c_str(), sNew._print().c_str(),
				aOld._print().c_str(), reward, aNew._print().c_str());
		status[iAgent] = agentStatus;

		if(agentStatus != _AGENT_LASTCOMM) {
			prepareAction(iAgent);
			comm->sendActionToApp();
		} else {
			bool bDone = true; //did all agents reach terminal state?
			for (Uint i=0; i<status.size(); i++)
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
	vector<Real> recv_state(sNew.sInfo.dim);

	unpackState(comm->getDataState(), iAgent, istatus, recv_state, reward);
	assert(iAgent>=0 && iAgent<static_cast<int>(agents.size()));

	sNew.set(recv_state);
	//agent's s is stored in sOld
	agents[iAgent]->Status = istatus;
	agents[iAgent]->swapStates();
	agents[iAgent]->setState(sNew);
	agents[iAgent]->getOldState(sOld);
	agents[iAgent]->getAction(aOld);
	agents[iAgent]->r = reward;
}

void Master::recvState(const int slave, int& iAgent, int& istatus, Real& reward)
{
	vector<Real> recv_state(sNew.sInfo.dim);
	int recv_iAgent = -1;
	unpackState(inbuf, recv_iAgent, istatus, recv_state, reward);
	assert(recv_iAgent>=0);
	iAgent = (slave-1) * nPerRank + recv_iAgent;
	//printf("%d %d %d, %lu\n",recv_iAgent,slave,iAgent,agents.size()); fflush(0);
	assert(iAgent>=0);
	assert(iAgent<static_cast<int>(agents.size()));

	sNew.set(recv_state);

	//agent's s is stored in sOld
	agents[iAgent]->Status = istatus;
	agents[iAgent]->swapStates();
	agents[iAgent]->setState(sNew);
	agents[iAgent]->getOldState(sOld);
	agents[iAgent]->getAction(aOld);
	agents[iAgent]->r = reward;

	MPI_Irecv(inbuf, inSize, MPI_BYTE, MPI_ANY_SOURCE, 1, slavesComm, &request);
}

void Master::sendAction(const int slave, const int iAgent)
{
	if(iAgent<0) die("Error in iAgent number in Master::sendAction\n");
	assert(iAgent >= 0 && iAgent < static_cast<int>(agents.size()));
	agents[iAgent]->act(aNew);
	for (Uint i=0; i<aI.dim; i++) outbuf[i] = aNew.vals[i];
	MPI_Send(outbuf, outSize, MPI_BYTE, slave, 0, slavesComm);
}

void Client::prepareAction(const int iAgent)
{
	if(iAgent<0) die("Error in iAgent number in Client::prepareAction\n");
	assert(iAgent >= 0 && iAgent < static_cast<int>(agents.size()));
	agents[iAgent]->act(aNew);
	double* const buf = comm->getDataAction();
	for (Uint i=0; i<aI.dim; i++) buf[i] = aNew.vals[i];
}

static void unpackState(double* const data, int& agent, _AGENT_STATUS& info,
		std::vector<double>& state, double& reward)
{
	assert(data not_eq nullptr);
	agent = doublePtrToInt(data+0);
	info  = doublePtrToInt(data+1);
	for (unsigned j=0; j<state.size(); j++) {
		state[j] = data[j+2];
		assert(not std::isnan(state[j]));
		assert(not std::isinf(state[j]));
	}
	reward = data[state.size()+2];
	assert(not std::isnan(reward));
	assert(not std::isinf(reward));
}
