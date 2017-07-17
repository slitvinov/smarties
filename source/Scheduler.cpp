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

Master::Master(MPI_Comm _c,Learner*const _l, Environment*const _e, Settings&_s):
  slavesComm(_c),learner(_l),env(_e),aI(_e->aI),sI(_e->sI), agents(_e->agents),
	bTrain(_s.bTrain), nPerRank(_e->nAgentsPerRank), saveFreq(_s.saveFreq),
	nSlaves(_s.nSlaves), nThreads(_s.nThreads), learn_rank(_s.learner_rank),
	learn_size(_s.learner_size), totNumSteps(_s.totNumSteps),
	outSize(_e->aI.dim*sizeof(double)), inSize((3+_e->sI.dim)*sizeof(double)),
	inpBufs(alloc_bufs(inSize,nSlaves)), outBufs(alloc_bufs(outSize,nSlaves)),
	slaveIrecvStatus(nSlaves, EMPTY), agentSortingCheck(agents.size(), 0), requests(nSlaves, MPI_REQUEST_NULL), profiler(_l->profiler)
{
	if(nSlaves*nPerRank != static_cast<int>(agents.size()))
		die("FATAL: Mismatch in master's nSlaves nPerRank nAgents.\n")
	//the following Irecv will be sent after sending the action
	for(int i=1; i<=nSlaves; i++) {
		MPI_Irecv(inpBufs[i-1], inSize, MPI_BYTE, i, 1, slavesComm, &requests[i-1]);
		slaveIrecvStatus[i-1] = OPEN;
	}
}

int Master::recvState(const int slave)
{
	vector<Real> recv_state(sI.dim);
	int recv_iAgent = -1, istatus;
	double reward;
	unpackState(inpBufs[slave-1], recv_iAgent, istatus, recv_state, reward);
	assert(recv_iAgent>=0);
	const int iAgent = (slave-1) * nPerRank + recv_iAgent;
	//printf("%d %d %d, %lu\n",recv_iAgent,slave,iAgent,agents.size()); fflush(0);
	assert(iAgent>=0);
	assert(iAgent<static_cast<int>(agents.size()));
	agents[iAgent]->update(istatus, recv_state, reward);
	if (istatus == _AGENT_LASTCOMM) {
		char path[256];
		sprintf(path, "cumulative_rewards_rank%02d.dat", learn_rank);
		std::ofstream outf(path, ios::app);
		outf<<learner->iter()<<" "<<iAgent<<" "<<agents[iAgent]->transitionID<<" "<<agents[iAgent]->cumulative_rewards<<endl;
		outf.close();
	}
	return iAgent;
}

void Master::restart(string fname)
{
	learner->restart(fname);
	/*
	if (fname == "none") return;
	char path[256];
	sprintf(path, "master_rank%02d.status", learn_rank);
	FILE * f = fopen(path, "r");
	if (f == NULL) return;

	unsigned long iter_fake = 0;
	fscanf(f, "master iter: %lu\n", &iter_fake);
	if(iter_fake) iter = iter_fake;
	printf("master iter: %lu\n", iter);
	fclose(f);
	*/
}

void Master::save()
{
	/*
	std::ofstream fout;
	char filepath[256];
	{
		sprintf(filepath, "rewards_rank%02d.dat", learn_rank);
		fout.open(filepath, ios::app);
		fout<<iter<<" "<<meanR<<" "<<varR<<endl;
		fout.close();
		printf("Iter %lu, Mean reward: %f variance:%f \n", iter, meanR, varR);
	}
	if(!bTrain) return;
	{
		sprintf(filepath, "master_rank%02d.status", learn_rank);
		fout.open(filepath, ios::trunc);
		fout<<"master iter: "<<iter<<endl;
		fout.close();
	}
	*/
	learner->save("policy");
}

int Master::run()
{
	while (true)
	{
		if(!bTrain && stepNum >= totNumSteps) return 1;
		if( bTrain && learner->reachedMaxGradStep()) return 1;

		learner->prepareData(); //sync data, make sure we can sample

		#pragma omp parallel num_threads(nThreads)
		#pragma omp master
		while (not learner->batchGradientReady())
		{
			//nSlaves tasks are reserved to handle slaves, if comm queue is empty
			const int availTasks = nThreads -learner->nTasks - (postponed_queue.size() ? 0 : nSlaves);
			learner->spawnTrainTasks(availTasks);

			for(int i=0; i<nSlaves; i++) //check all slaves
			{
				int completed=0;
				MPI_Status mpistatus;
				{
					lock_guard<mutex> lock(mpi_mutex);
					if(slaveIrecvStatus[i] == OPEN) //otherwise, Irecv not sent
					{
						MPI_Test(&requests[i], &completed, &mpistatus);
					}
					else if (slaveIrecvStatus[i] == SEND)
					{
						vector<Real> _a(outBufs[i], outBufs[i]+aI.dim);
						MPI_Send(outBufs[i], outSize, MPI_BYTE, i+1, 0, slavesComm);
						debugS("Sent action to slave %d: [%s]\n", i+1, print(_a).c_str());
						slaveIrecvStatus[i] = OVER;
					}
					else
					{
						if(slaveIrecvStatus[i] != OVER && slaveIrecvStatus[i] != DOING)
						_die("slave status is %d\n",slaveIrecvStatus[i]);
					}

					if(slaveIrecvStatus[i] == OVER)
					{
						MPI_Irecv(inpBufs[i], inSize, MPI_BYTE, i+1, 1, slavesComm, &requests[i]);
						slaveIrecvStatus[i] = OPEN;
					}
				}

				if(completed)
				{
					int slave = mpistatus.MPI_SOURCE;
					assert(slaveIrecvStatus[i] == OPEN && slave==i+1);
					debugS("Master receives from %d\n", slave);
					slaveIrecvStatus[slave-1] = DOING; //slave will be 'served' by task
					const int agent = recvState(slave); //unpack buffer

					if(learnerReadyForAgent(slave, agent))
					{
						{
							lock_guard<mutex> lock(learner->task_mutex);
							learner->nTasks++;
						}
						#pragma omp task firstprivate(slave, agent) if(learner->nTasks<nThreads)
						{
							processRequest(slave, agent);
							lock_guard<mutex> lock(learner->task_mutex);
							learner->nTasks--;
						}
					}
					else //never triggered for off-policy algorithms:
					{
						die("Not supposed to be here yet\n");
						postponed_queue.push_back(make_pair(slave, agent));
					}
					debugS("number of tasks %d\n", learner->nTasks);
					assert(learner->nTasks<nThreads && learner->nTasks>=0);
				}
			}
		}

		//for(int i=0; i<nSlaves; i++)
		//	assert(slaveIrecvStatus[i] == OPEN);
		learner->applyGradient(); //tasks have finished, update is ready

		if(postponed_queue.size()) //never triggered for off-policy algorithms
		{
			debugS("postponed_queue.size(): %lu\n", postponed_queue.size());
			#pragma omp parallel num_threads(nThreads)
			#pragma omp master
			for (const auto& w : postponed_queue) {
				const int slave = w.first, agent = w.second;
				#pragma omp task firstprivate(slave, agent)
					processRequest(slave, agent);
			}
			postponed_queue.clear();
		}
		profiler->stop_all();
	}
	die("FATAL MASTER::run.\n");
	return 0;
}

void Master::processRequest(const int slave, const int agent)
{
	assert(agent >= 0 && agent < static_cast<int>(agents.size()));
	if (agents[agent]->Status == _AGENT_FAILCOMM) //it was a crash :sadface:
	{ //TODO fix for on-pol: if crash clear unfinished workspace assigned to slave
		learner->clearFailedSim((slave-1)*nPerRank, slave*nPerRank);
		for (int i=(slave-1)*nPerRank; i<slave*nPerRank; i++) agents[i]->reset();
		printf("Received an _AGENT_FAILCOMM\n");
		slaveIrecvStatus[slave-1] = OVER;
	}
	else
	{
		//pick next action and ...do a bunch of other stuff with the data:
		learner->select(agent, *agents[agent]);
		debugS("Agent %d (%d): [%s] -> [%s] rewarded with %f going to [%s]\n", agent, agents[agent]->Status, agents[agent]->sOld->_print().c_str(), agents[agent]->s->_print().c_str(), agents[agent]->r, agents[agent]->a->_print().c_str());

		if (agents[agent]->Status != _AGENT_LASTCOMM)
		{
			for(Uint i=0; i<aI.dim; i++)
				outBufs[slave-1][i] = agents[agent]->a->vals[i];

			lock_guard<mutex> lock(mpi_mutex);
			slaveIrecvStatus[slave-1] = SEND;
		}
		else
		{ //if terminal, no action required
			lock_guard<mutex> lock(mpi_mutex);
			slaveIrecvStatus[slave-1] = OVER;
			//if(env->resetAll) TODO
			//	learner->pushBackEndedSim((slave-1)*nPerRank, slave*nPerRank);
			#pragma omp atomic
			++stepNum; //sequence counter: used to terminate if not training
		}
	}
}

int Master::learnerReadyForAgent(const int slave, const int agent) const
{
	//Return whether we need more data from this agent:
	//generally will return true. Except when on-policy algorithms (ie. GAE).
	//  For example: if batch is almost ready and waiting from the last agents to
	//  finish sequence, getting more data would then be a waste because then the
	//  gradient will be applied, and data would become off-policy and unusable
	//However, if I receive data of a brand new seq from agent B on slave S while
	//waiting for terminal state of agent C, also on slave S, then user is NOT
	//using correct algorithm for the problem or has implemented something wrong.
	//There is a check on on-policy algo to verify that when new seq from agent
	//on a slave S begins, all other slave S's agents must have sent term state.
	const int ready = learner->readyForAgent(slave, agent);
	assert(ready>=0 || agents[agent]->Status == _AGENT_FIRSTCOMM);
	return ready>=0;
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
