//
//  Scheduler.h
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#pragma once

class Learner;
#include "Communicator.h"
#include "Learners/Learner.h"

class Master
{
private:
	const MPI_Comm slavesComm;
	Learner* const learner;
	Environment* const env;
	const ActionInfo aI;
	const StateInfo  sI;
	const vector<Agent*> agents;
	const int bTrain, nPerRank, saveFreq, nSlaves, nThreads, learn_rank, learn_size, totNumSteps, outSize, inSize;
	const vector<double*> inpBufs;
	const vector<double*> outBufs;
	unsigned long iter = 0;
	long int stepNum = 0;
	//vector<int> status;
	vector<Real> cumulative_rewards;
	vector<MPI_Request> requests;
	vector<bool> openIrecv;

	void trackAgentsPerformance(const _AGENT_STATUS agentStatus, const int agent, const Real reward);
	void recvState(const int slave, int& iAgent, int& istatus, Real& reward);
	void sendAction(const int slave, const int iAgent);
	void save();

	static inline vector<double*> alloc_bufs(const int size, const int num)
	{
		vector<double*> ret(size, nullptr);
		for(int i=0; i<num; i++) ret[i] = _alloc(size);
		return ret;
	}

public:
	Master(MPI_Comm comm, Learner*const learner, Environment*const env, Settings& settings);
	~Master()
	{
		_dispose_object(env);
		for(int i=0; i<nSlaves; i++) _dealloc(inpBufs[i]);
		for(int i=0; i<nSlaves; i++) _dealloc(outBufs[i]);
		_dispose_object(learner);
	}
	void sendTerminateReq(const double msg = -256)
	{
		//it's ugly, i send -256 to kill the slaves... but...
		//what are the chances that learner sends action -256.(+/- eps) to clients?
		printf("nslaves %d\n",nSlaves);
		for (int slave=1; slave<=nSlaves; slave++) {
			outBufs[slave-1][0] =  msg;
			MPI_Ssend(outBufs[slave-1], outSize, MPI_BYTE, slave, 0, slavesComm);
		}
	}
	int run();
	void restart(string fname);
};

class Slave
{
private:
	Communicator* const comm;
	Environment* const env;
	const bool bTrain;
	vector<int> status;

public:
	Slave(Communicator*const c, Environment*const e, Settings& s);
	~Slave()
	{
		_dispose_object(env);
	}
	void run();
};

/*
class Client
{
private:
	Learner* const learner;
	Communicator* const comm;
	Environment* const env;
	vector<Agent*> agents;
	const ActionInfo aI;
	const StateInfo  sI;
	vector<int> status;
	void prepareState(int& iAgent, int& istatus, Real& reward);
	void prepareAction(const int iAgent);

public:
	Client(Learner*const l,Communicator*const c,Environment*const e,Settings&s);
	~Client()
	{
		_dispose_object(env);
		_dispose_object(learner);
	}
	void run();
};
*/
