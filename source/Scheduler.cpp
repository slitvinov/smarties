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

Master::Master(MPI_Comm _c,Learner*const _l, Environment*const _e, Settings&_s):
  slavesComm(_c),learner(_l),env(_e),aI(_e->aI),sI(_e->sI),agents(_e->agents),
	bTrain(_s.bTrain), nPerRank(_e->nAgentsPerRank), saveFreq(_s.saveFreq),
	nSlaves(_s.nSlaves), nThreads(_s.nThreads), learn_rank(_s.learner_rank),
	learn_size(_s.learner_size), totNumSteps(_s.totNumSteps),
	outSize(_e->aI.dim*sizeof(double)), inSize((3+_e->sI.dim)*sizeof(double)),
	inpBufs(alloc_bufs(inSize,nSlaves)), outBufs(alloc_bufs(outSize,nSlaves)),
	cumulative_rewards(_e->agents.size(),0), openIrecv(nSlaves, false)
{
	//the following Irecv will be sent after sending the action
	requests.resize(nSlaves, MPI_REQUEST_NULL);
	for(int i=1; i<=nSlaves; i++) {
		MPI_Irecv(inpBufs[i-1], inSize, MPI_BYTE, i, 1, slavesComm, &requests[i-1]);
		openIrecv[i-1] = true;
	}
}

void Master::recvState(const int slave, int& iAgent, int& istatus, Real& reward)
{
	vector<Real> recv_state(sI.dim);
	int recv_iAgent = -1;
	unpackState(inpBufs[slave-1], recv_iAgent, istatus, recv_state, reward);
	assert(recv_iAgent>=0);
	iAgent = (slave-1) * nPerRank + recv_iAgent;
	//printf("%d %d %d, %lu\n",recv_iAgent,slave,iAgent,agents.size()); fflush(0);
	assert(iAgent>=0);
	assert(iAgent<static_cast<int>(agents.size()));

	State s(sI);
	s.set(recv_state);
	agents[iAgent]->Status = istatus;
	agents[iAgent]->swapStates(); //swap sold and snew
	agents[iAgent]->setState(s);
	agents[iAgent]->r = reward;
}

void Master::sendAction(const int slave, const int iAgent)
{
	if(iAgent<0) die("Error in iAgent number in Master::sendAction\n");
	assert(iAgent >= 0 && iAgent < static_cast<int>(agents.size()));
	for (Uint i=0; i<aI.dim; i++)
		outBufs[slave-1][i] = agents[iAgent]->a->vals[i];

	MPI_Send(outBufs[slave-1], outSize, MPI_BYTE, slave, 0, slavesComm);
}

void Master::restart(string fname)
{
	char path[256];
	learner->restart(fname);
	if (fname == "none") return;
	sprintf(path, "master_rank%02d.status", learn_rank);
	FILE * f = fopen(path, "r");
	if (f == NULL) return;

	unsigned long iter_fake = 0;
	fscanf(f, "master iter: %lu\n", &iter_fake);
	if(iter_fake) iter = iter_fake;
	printf("master iter: %lu\n", iter);
	fclose(f);
}

void Master::save()
{
	std::ofstream fout;
	char filepath[256];
	/*
	{
		sprintf(filepath, "rewards_rank%02d.dat", learn_rank);
		fout.open(filepath, ios::app);
		fout<<iter<<" "<<meanR<<" "<<varR<<endl;
		fout.close();
		printf("Iter %lu, Mean reward: %f variance:%f \n", iter, meanR, varR);
	}
	*/
	if(!bTrain) return;
	{
		sprintf(filepath, "master_rank%02d.status", learn_rank);
		fout.open(filepath, ios::trunc);
		fout<<"master iter: "<<iter<<endl;
		fout.close();
	}
	learner->save("policy");
}

int Master::run()
{
	while (true) {
		if(!bTrain && stepNum==totNumSteps) return 1; //used to terminate
		MPI_Status mpistatus;
		int completed=0;
		while (true) {
			//check on the threads, synchronize, apply gradient
			if (learner->checkBatch(iter)) return 0;
			for(int i=0; i<nSlaves && !completed; i++)
				if(openIrecv[i]) //otherwise, slave is being 'served'
					MPI_Test(&requests[i], &completed, &mpistatus);
			if (completed) break;
		}
		const int slave = mpistatus.MPI_SOURCE;
		debugS("Master receives from %d\n", slave);
		openIrecv[slave-1] = false;

#pragma omp task firstprivate(slave)
		{
			int agent, agentStatus;
			double reward;
			recvState(slave, agent, agentStatus, reward);

			if (agentStatus == _AGENT_FAILCOMM)
			{
				learner->clearFailedSim((slave-1)*nPerRank, slave*nPerRank);
				for (int i = (slave-1)*nPerRank; i<slave*nPerRank; i++)
					cumulative_rewards[i] = 0;
			}
			else
			{
				learner->select(agent, *agents[agent]);
				debugS("Agent %d (%d): [%s] -> [%s] rewarded with %f going to [%s]\n", agent, agents[agent]->Status, agents[agent]->sOld->_print().c_str(), agents[agent]->s->_print().c_str(), reward, agents[agent]->a->_print().c_str()); fflush(0);

				//track performance of agents:
				trackAgentsPerformance(agentStatus, agent, reward);

				if (agentStatus != _AGENT_LASTCOMM)  {
					sendAction(slave, agent);
				} else { //if terminal, no action required
					if(env->resetAll)
						learner->pushBackEndedSim((slave-1)*nPerRank, slave*nPerRank);

#pragma omp atomic
					++stepNum; //used to terminate
				}
			}
			MPI_Irecv(inpBufs[slave-1], inSize, MPI_BYTE, slave, 1, slavesComm, &requests[slave-1]);
			openIrecv[slave-1] = true;
		}
		//if (++iter % saveFreq == 0) save();
	}
	die("How on earth could you possibly get here? \n");
	return 0;
}
/*
int Master::run()
{
	MPI_Status mpistatus;
	int completed=0, agentStatus=0, agent;
	double reward;
	while (true) {
		while (true) {
			//check on the threads, synchronize, apply gradient
			if (learner->checkBatch(iter)) return 0;

			MPI_Test(&request, &completed, &mpistatus);
			if (completed) break;
		}
		debugS("Master receives from %d\n", mpistatus.MPI_SOURCE);
		const int slave = mpistatus.MPI_SOURCE;
		recvState(slave, agent, agentStatus, reward);

		if (agentStatus == _AGENT_FAILCOMM) {
			learner->clearFailedSim((slave-1)*nPerRank, slave*nPerRank);
			for (int i = (slave-1)*nPerRank; i<slave*nPerRank; i++) {
				status[i] = 1; cumulative_rewards[i] = 0;
			}
			continue;
		}

		learner->select(agent, sNew, aNew, sOld, aOld, agentStatus, reward);
		debugS("Agent %d: [%s] -> [%s] with [%s] rewarded with %f going to [%s]\n",
				agent, sOld._print().c_str(), sNew._print().c_str(),
				aOld._print().c_str(), reward, aNew._print().c_str());

		//track performance of agents:
		trackAgentsPerformance(agentStatus, agent, reward);
		if (agentStatus != _AGENT_FIRSTCOMM) {
			const Real alpha = 1./saveFreq;// + std::min(0.,1-iter/(Real)saveFreq);
			const Real oldMean = meanR;
			cumulative_rewards[agent] += reward;
			meanR = (1.-alpha)*meanR + alpha*reward;
			varR = (1.-alpha)*varR + alpha*(reward-meanR)*(reward-oldMean);
		} else cumulative_rewards[agent] = 0;

		if (agentStatus != _AGENT_LASTCOMM)  {
			sendAction(slave, agent);
		} else { //if terminal, no action required
			if(env->resetAll)
				learner->pushBackEndedSim((slave-1)*nPerRank, slave*nPerRank);

			if(!bTrain)
				if(++stepNum == totNumSteps) return 1; //used to terminate
		}
		if (++iter % saveFreq == 0) save();
	}
	die("How on earth could you possibly get here? \n");
	return 0;
}
*/
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
				if (comm->sendActionToApp()) {
					printf("Slave exiting\n");
					fflush(0);
					return;
				}
			} else {
				/*
					bool bDone = true; //did all agents reach terminal state?
					for (Uint i=0; i<status.size(); i++)
						bDone = bDone && status[i] == _AGENT_LASTCOMM;
					bDone = bDone || env->resetAll; //does env end is any terminates?
          if(bDone && !bTrain) {
            comm->answerTerminateReq(-1);
            return;
          }
          else
				 */
				comm->answerTerminateReq(1.);
			}
		}
		//if here, a crash happened:
		//if we are training, then launch again, otherwise exit
		//if (!bTrain) return;
		comm->launch();
	}
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

void Master::trackAgentsPerformance(const _AGENT_STATUS agentStatus, const int agent, const Real reward)
{
	if (agentStatus != _AGENT_FIRSTCOMM) {
		//const Real alpha = 1./saveFreq;// + std::min(0.,1-iter/(Real)saveFreq);
		//const Real oldMean = meanR;
		cumulative_rewards[agent] += reward;
		//meanR = (1.-alpha)*meanR + alpha*reward;
		//varR = (1.-alpha)*varR + alpha*(reward-meanR)*(reward-oldMean);
	}

#pragma omp critical
	if (agentStatus == _AGENT_LASTCOMM) {
		char path[256];
		sprintf(path, "cumulative_rewards_rank%02d.dat", learn_rank);
		std::ofstream outf(path, ios::app);
		outf<<learner->iter()<<" "<<agent<<" "<<cumulative_rewards[agent]<<endl;
		cumulative_rewards[agent] = 0;
		outf.close();
	}
}

/*
Client::Client(Learner*const _l, Communicator*const _c, Environment*const _e,
		Settings& _s):
	  learner(_l), comm(_c), env(_e), agents(_e->agents), aI(_e->aI), sI(_e->sI),
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

void Client::prepareAction(const int iAgent)
{
	if(iAgent<0) die("Error in iAgent number in Client::prepareAction\n");
	assert(iAgent >= 0 && iAgent < static_cast<int>(agents.size()));
	agents[iAgent]->act(aNew);
	double* const buf = comm->getDataAction();
	for (Uint i=0; i<aI.dim; i++) buf[i] = aNew.vals[i];
}
*/
